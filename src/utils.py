# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import typing
import gym
import threading
from torch import multiprocessing as mp
import logging
import traceback
import os
import numpy as np

from src.core import prof
from src.env_utils import FrameStack, Environment, Minigrid2Image
from src import atari_wrappers as atari_wrappers

from gym_minigrid import wrappers as wrappers

import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT

#import vizdoomgym


COMPLETE_MOVEMENT = [
    ['NOOP'],
    ['up'],
    ['down'],
    ['left'],
    ['left', 'A'],
    ['left', 'B'],
    ['left', 'A', 'B'],
    ['right'],
    ['right', 'A'],
    ['right', 'B'],
    ['right', 'A', 'B'],
    ['A'],
    ['B'],
    ['A', 'B'],
]

shandle = logging.StreamHandler()
shandle.setFormatter(
    logging.Formatter(
        '[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] '
        '%(message)s'))
log = logging.getLogger('torchbeast')
log.propagate = False
log.addHandler(shandle)
log.setLevel(logging.INFO)

Buffers = typing.Dict[str, typing.List[torch.Tensor]]


def create_env(flags):
    if 'MiniGrid' in flags.env:
        return Minigrid2Image(wrappers.FullyObsWrapper(gym.make(flags.env)))
    elif 'Mario' in flags.env:
        env = atari_wrappers.wrap_pytorch(
            atari_wrappers.wrap_deepmind(
                atari_wrappers.make_atari(flags.env, noop=True),
                clip_rewards=False,
                frame_stack=True,
                scale=False,
                fire=True))
        env = JoypadSpace(env, COMPLETE_MOVEMENT)
        return env
    else:
        env = atari_wrappers.wrap_pytorch(
            atari_wrappers.wrap_deepmind(
                atari_wrappers.make_atari(flags.env, noop=False),
                clip_rewards=False,
                frame_stack=True,
                scale=False,
                fire=False))
        return env

import torch.nn as nn

class ActorPool(nn.Module):
    def __init__(self, policy, flags):
        super().__init__()

        self.batch_size = flags.batch_size
        self.unroll_length = flags.unroll_length

        from gym_minigrid.fast_minigrid import BatchMinigrid
        env_seeds = torch.randint(0, 2**32 - 1, (self.batch_size,)).tolist()
        self.batch_env = BatchMinigrid(seeds=env_seeds)
        self.policy = policy

        self.unroll_partial_obs = []
        self.unroll_reward = []
        self.unroll_done = []
        self.unroll_action = []
        self.unroll_policy_logits = []
        self.unroll_episode_return = []

    @torch.no_grad()
    def prepare(self):
        # env_output frame
        # ! env_output reward
        # ! env_output done
        # ! env_output episode_return
        # env_output episode_step
        # env_output episode_win
        # env_output carried_col
        # env_output carried_obj
        # ! env_output partial_obs
        # ! agent_output policy_logits
        # agent_output baseline
        # ! agent_output action

        device = next(self.policy.parameters()).device
        done = torch.zeros(self.batch_size, dtype=torch.bool, device=device)
        reward = torch.zeros(self.batch_size, dtype=torch.float32, device=device)

        self.agent_state = self.policy.initial_state(batch_size=self.batch_size)

        partial_obs = self.batch_env.render_fpv()[None, ...]
        agent_output, unused_state = self.policy({
            'partial_obs': partial_obs,
            'done': done,
        }, self.agent_state)

        self.unroll_partial_obs.append(partial_obs)
        self.unroll_done.append(done)
        self.unroll_reward.append(reward)
        self.unroll_action.append(agent_output['action'])
        self.unroll_policy_logits.append(agent_output['policy_logits'])
        self.unroll_episode_return.append(reward)

    @torch.no_grad()
    def forward(self):
        device = next(self.policy.parameters()).device

        partial_obs = self.unroll_partial_obs[-1]
        done = self.unroll_done[-1]

        for t in range(self.unroll_length):
            agent_output, self.agent_state = self.policy({
                'partial_obs': partial_obs,
                'done': done,
            }, self.agent_state)

            partial_obs, reward, done, _ = self.batch_env.step(agent_output['action'][0])
            partial_obs = partial_obs[None, ...]
            # if done we need to reset that environment

            self.unroll_partial_obs.append(partial_obs)
            self.unroll_reward.append(reward)
            self.unroll_done.append(done)
            self.unroll_action.append(agent_output['action'])
            self.unroll_policy_logits.append(agent_output['policy_logits'])
            self.unroll_episode_return.append(reward) # ok for sparse rewards

        batch = {
            'partial_obs': torch.cat(self.unroll_partial_obs, dim=0),
            'reward': torch.stack(self.unroll_reward),
            'done': torch.stack(self.unroll_done),
            'action': torch.cat(self.unroll_action, dim=0),
            'policy_logits': torch.cat(self.unroll_policy_logits, dim=0),
            'episode_return': torch.stack(self.unroll_episode_return),
        }

        # remove all but the last frame in the unroll
        self.unroll_partial_obs = self.unroll_partial_obs[-1:]
        self.unroll_reward = self.unroll_reward[-1:]
        self.unroll_done = self.unroll_done[-1:]
        self.unroll_action = self.unroll_action[-1:]
        self.unroll_policy_logits = self.unroll_policy_logits[-1:]
        self.unroll_episode_return = self.unroll_episode_return[-1:]

        episode_returns = batch['episode_return'][batch['done']]
        print(torch.where(batch['done']))
        return batch, self.agent_state





def get_batch(free_queue: mp.SimpleQueue,
              full_queue: mp.SimpleQueue,
              buffers: Buffers,
              initial_agent_state_buffers,
              flags,
              timings,
              lock=threading.Lock()):
    with lock:
        timings.time('lock')
        indices = [full_queue.get() for _ in range(flags.batch_size)]
        timings.time('dequeue')
    batch = {
        key: torch.stack([buffers[key][m] for m in indices], dim=1)
        for key in buffers
    }
    initial_agent_state = (
        torch.cat(ts, dim=1)
        for ts in zip(*[initial_agent_state_buffers[m] for m in indices])
    )
    timings.time('batch')
    for m in indices:
        free_queue.put(m)
    timings.time('enqueue')
    batch = {
        k: t.to(device=flags.device, non_blocking=True)
        for k, t in batch.items()
    }
    initial_agent_state = tuple(t.to(device=flags.device, non_blocking=True)
                                for t in initial_agent_state)
    timings.time('device')
    return batch, initial_agent_state

def create_buffers(obs_shape, num_actions, flags) -> Buffers:
    T = flags.unroll_length
    specs = dict(
        frame=dict(size=(T + 1, *obs_shape), dtype=torch.int32),
        reward=dict(size=(T + 1,), dtype=torch.float32),
        done=dict(size=(T + 1,), dtype=torch.bool),
        episode_return=dict(size=(T + 1,), dtype=torch.float32),
        episode_step=dict(size=(T + 1,), dtype=torch.int32),
        last_action=dict(size=(T + 1,), dtype=torch.int64),
        policy_logits=dict(size=(T + 1, num_actions), dtype=torch.float32),
        baseline=dict(size=(T + 1,), dtype=torch.float32),
        action=dict(size=(T + 1,), dtype=torch.int64),
        episode_win=dict(size=(T + 1,), dtype=torch.int32),
        carried_obj=dict(size=(T + 1,), dtype=torch.int32),
        carried_col=dict(size=(T + 1,), dtype=torch.int32),
        partial_obs=dict(size=(T + 1, 7, 7, 3), dtype=torch.int32),
        episode_state_count=dict(size=(T + 1,), dtype=torch.float32),
        train_state_count=dict(size=(T + 1,), dtype=torch.float32),
    )

    buffers: Buffers = {key: [] for key in specs}
    for _ in range(flags.num_buffers):
        for key in buffers:
            buffers[key].append(torch.empty(**specs[key]).share_memory_())
    return buffers

def cat_buffers(*args):
    if args[0] is None:
        return cat_buffers(*args[1:])
    return {key: torch.cat([arg[key].cpu() for arg in args], dim=1)
            for key in args[0]}

def act(i: int, free_queue: mp.SimpleQueue, full_queue: mp.SimpleQueue,
        model: torch.nn.Module, buffers: Buffers, 
        episode_state_count_dict: dict, train_state_count_dict: dict,
        initial_agent_state_buffers, flags):
    try:
        log.info('Actor %i started.', i)
        timings = prof.Timings()  

        gym_env = create_env(flags)
        seed = i ^ int.from_bytes(os.urandom(4), byteorder='little')
        gym_env.seed(seed)
        
        if flags.num_input_frames > 1:
            gym_env = FrameStack(gym_env, flags.num_input_frames)  

        env = Environment(gym_env, fix_seed=flags.fix_seed, env_seed=flags.env_seed)

        env_output = env.initial()
        device = next(model.parameters()).device

        agent_state = model.initial_state(batch_size=1)
        agent_output, unused_state = model({
            'partial_obs': env_output['partial_obs'].cuda(device, non_blocking=True),
            'done': env_output['done'].cuda(device, non_blocking=True),
        }, agent_state)

        while True:
            index = free_queue.get()
            if index is None:
                break

            # initial_obs = env_output
            # Write old rollout end.
            for key in env_output:
                buffers[key][index][0, ...] = env_output[key]
            for key in agent_output:
                buffers[key][index][0, ...] = agent_output[key]
            for i, tensor in enumerate(agent_state):
                initial_agent_state_buffers[index][i][...] = tensor


            # Update the episodic state counts
            episode_state_key = tuple(env_output['frame'].view(-1).tolist())
            if episode_state_key in episode_state_count_dict:
                episode_state_count_dict[episode_state_key] += 1
            else:
                episode_state_count_dict.update({episode_state_key: 1})
            buffers['episode_state_count'][index][0, ...] = \
                torch.tensor(1 / np.sqrt(episode_state_count_dict.get(episode_state_key)))
            
            # Reset the episode state counts when the episode is over
            if env_output['done'][0][0]:
                for episode_state_key in episode_state_count_dict:
                    episode_state_count_dict = dict()

            # Update the training state counts if you're doing count-based exploration
            if flags.model == 'count':
                train_state_key = tuple(env_output['frame'].view(-1).tolist())
                if train_state_key in train_state_count_dict:
                    train_state_count_dict[train_state_key] += 1
                else:
                    train_state_count_dict.update({train_state_key: 1})
                buffers['train_state_count'][index][0, ...] = \
                    torch.tensor(1 / np.sqrt(train_state_count_dict.get(train_state_key)))

            # Do new rollout
            for t in range(flags.unroll_length):
                timings.reset()

                with torch.no_grad():
                    agent_output, agent_state = model({
                        'partial_obs': env_output['partial_obs'].cuda(device, non_blocking=True),
                        'done': env_output['done'].cuda(device, non_blocking=True),
                    }, agent_state)

                timings.time('model')

                env_output = env.step(agent_output['action'])

                timings.time('step')
                # TODO: make it a length 4 trajectories

                for key in env_output:
                    buffers[key][index][t + 1, ...] = env_output[key]
    
                for key in agent_output:
                    buffers[key][index][t + 1, ...] = agent_output[key]
                
                # Update the episodic state counts
                episode_state_key = tuple(env_output['frame'].view(-1).tolist())
                if episode_state_key in episode_state_count_dict:
                   episode_state_count_dict[episode_state_key] += 1
                else:
                    episode_state_count_dict.update({episode_state_key: 1})
                buffers['episode_state_count'][index][t + 1, ...] = \
                    torch.tensor(1 / np.sqrt(episode_state_count_dict.get(episode_state_key)))

                # Reset the episode state counts when the episode is over
                if env_output['done'][0][0]:
                    episode_state_count_dict = dict()

                # Update the training state counts if you're doing count-based exploration
                if flags.model == 'count':
                    train_state_key = tuple(env_output['frame'].view(-1).tolist())
                    if train_state_key in train_state_count_dict:
                        train_state_count_dict[train_state_key] += 1
                    else:
                        train_state_count_dict.update({train_state_key: 1})
                    buffers['train_state_count'][index][t + 1, ...] = \
                        torch.tensor(1 / np.sqrt(train_state_count_dict.get(train_state_key)))

                timings.time('write')
            full_queue.put(index)

        if i == 0:
            log.info('Actor %i: %s', i, timings.summary())

    except KeyboardInterrupt:
        pass  
    except Exception as e:
        logging.error('Exception in worker process %i', i)
        traceback.print_exc()
        print()
        raise e

