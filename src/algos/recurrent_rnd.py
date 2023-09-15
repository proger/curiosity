# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys
import threading
import time
import timeit
import pprint

from celluloid import Camera
import matplotlib.pyplot as plt
import numpy as np
import wandb

import torch
from torch import multiprocessing as mp
from torch import nn
from torch.nn import functional as F

from src.core import file_writer
from src.core import prof
from src.core import vtrace

import src.models as models
import src.losses as losses

from src.env_utils import FrameStack
from src.utils import get_batch, log, create_env, create_buffers, act

MarioDoomPolicyNet = models.MarioDoomPolicyNet
MarioDoomStateEmbeddingNet = models.MarioDoomStateEmbeddingNet

FullObsMinigridPolicyNet = models.FullObsMinigridPolicyNet
FullObsMinigridStateEmbeddingNet = models.FullObsMinigridStateEmbeddingNet

def learn(actor_model,
          model: models.MinigridPolicyNet,
          random_target_network,
          predictor_network,
          batch,
          initial_agent_state, 
          optimizer,
          predictor_optimizer,
          scheduler,
          flags,
          frames=None,
          lock=threading.Lock()):
    """Performs a learning (optimization) step."""
    with lock:
        if flags.use_fullobs_intrinsic:
            random_embedding = random_target_network(batch, next_state=True)\
                    .reshape(flags.unroll_length, flags.batch_size, 128)        
            predicted_embedding = predictor_network(batch, next_state=True)\
                    .reshape(flags.unroll_length, flags.batch_size, 128)
        else:
            random_embedding = random_target_network(batch['partial_obs'][1:].to(device=flags.device))
            if flags.rnd_autoregressive is not None:
                # shift random embedding by 1 step to the right
                shifted_targets = F.pad(random_embedding, (0, 0, 0, 0, 1, 0))[:-1]

                predicted_embedding = predictor_network(
                    inputs=batch['partial_obs'][1:].to(device=flags.device),
                    shifted_targets=shifted_targets,
                )
            else:
                predicted_embedding = predictor_network(
                    inputs=batch['partial_obs'][1:].to(device=flags.device)
                )

        intrinsic_rewards = torch.norm(predicted_embedding.detach() - random_embedding.detach(), dim=2, p=2)

        intrinsic_reward_coef = flags.intrinsic_reward_coef
        intrinsic_rewards *= intrinsic_reward_coef 
        
        num_samples = flags.unroll_length * flags.batch_size
        actions_flat = batch['action'][1:].reshape(num_samples).cpu().detach().numpy()
        intrinsic_rewards_flat = intrinsic_rewards.reshape(num_samples).cpu().detach().numpy()

        rnd_loss = flags.rnd_loss_coef * \
                losses.compute_forward_dynamics_loss(predicted_embedding, random_embedding.detach()) 
            
        learner_outputs, unused_state = model(batch, initial_agent_state)

        bootstrap_value = learner_outputs['baseline'][-1]

        batch = {key: tensor[1:] for key, tensor in batch.items()}
        learner_outputs = {
            key: tensor[:-1]
            for key, tensor in learner_outputs.items()
        }
        
        rewards = batch['reward']
            
        if flags.no_reward:
            total_rewards = intrinsic_rewards
        else:            
            total_rewards = rewards + intrinsic_rewards
        clipped_rewards = torch.clamp(total_rewards, -1, 1)
        
        discounts = (~batch['done']).float() * flags.discounting

        vtrace_returns = vtrace.from_logits(
            behavior_policy_logits=batch['policy_logits'],
            target_policy_logits=learner_outputs['policy_logits'],
            actions=batch['action'],
            discounts=discounts,
            rewards=clipped_rewards,
            values=learner_outputs['baseline'],
            bootstrap_value=bootstrap_value)

        pg_loss = losses.compute_policy_gradient_loss(learner_outputs['policy_logits'],
                                               batch['action'],
                                               vtrace_returns.pg_advantages)
        baseline_loss = flags.baseline_cost * losses.compute_baseline_loss(
            vtrace_returns.vs - learner_outputs['baseline'])
        entropy_loss = flags.entropy_cost * losses.compute_entropy_loss(
            learner_outputs['policy_logits'])

        total_loss = pg_loss + baseline_loss + entropy_loss + rnd_loss

        episode_returns = batch['episode_return'][batch['done']]

        scheduler.step()
        optimizer.zero_grad()
        predictor_optimizer.zero_grad()
        total_loss.backward()
        grad_norm_policy = nn.utils.clip_grad_norm_(model.parameters(), flags.max_grad_norm)
        grad_norm_rnd_predictor = nn.utils.clip_grad_norm_(predictor_network.parameters(), flags.max_grad_norm)
        optimizer.step()
        predictor_optimizer.step()

        actor_model.load_state_dict(model.state_dict())

        # When adding keys here, do it again in stat_keys below in this file.
        stats = {
            'mean_episode_return': torch.mean(episode_returns).item(),
            'total_loss': total_loss.item(),
            'pg_loss': pg_loss.item(),
            'baseline_loss': baseline_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'rnd_loss': rnd_loss.item(),
            'mean_rewards': torch.mean(rewards).item(),
            'mean_intrinsic_rewards': torch.mean(intrinsic_rewards).item(),
            'mean_total_rewards': torch.mean(total_rewards).item(),
            'grad_norm_policy': grad_norm_policy.item(),
            'grad_norm_rnd_predictor': grad_norm_rnd_predictor.item(),
            'lr_policy': scheduler.get_lr()[0],
        }
        return stats


def train(flags):  
    if flags.xpid is None:
        flags.xpid = 'curiosity-%s' % time.strftime('%Y%m%d-%H%M%S')

    wandb.init(config=flags)

    plogger = file_writer.FileWriter(
        xpid=flags.xpid,
        xp_args=flags.__dict__,
        rootdir=flags.savedir,
    )

    checkpointpath = os.path.expandvars(
        os.path.expanduser('%s/%s/%s' % (flags.savedir, flags.xpid,
                                         'model.tar')))

    T = flags.unroll_length
    B = flags.batch_size

    flags.device = None
    if not flags.disable_cuda and torch.cuda.is_available():
        log.info('Using CUDA.')
        flags.device = torch.device('cuda')
    else:
        log.info('Not using CUDA.')
        flags.device = torch.device('cpu')

    env = create_env(flags)
    if flags.num_input_frames > 1:
        env = FrameStack(env, flags.num_input_frames)  

    if 'MiniGrid' in flags.env: 
        if flags.use_fullobs_policy:
            model = FullObsMinigridPolicyNet(env.observation_space.shape, env.action_space.n).to(device=flags.device)
        else:
            model = models.MinigridPolicyNet(env.observation_space.shape, env.action_space.n).to(device=flags.device)
        if flags.use_fullobs_intrinsic:
            raise NotImplementedError('use_fullobs_intrinsic is not implemented yet')
            random_target_network = FullObsMinigridStateEmbeddingNet(env.observation_space.shape).to(device=flags.device)
            predictor_network = FullObsMinigridStateEmbeddingNet(env.observation_space.shape).to(device=flags.device)
        else:
            random_target_network = models.MinigridStateEmbeddingNet(env.observation_space.shape).to(device=flags.device)
            predictor_network = models.MinigridStateSequenceNet(
                env.observation_space.shape,
                history=flags.rnd_history,
                autoregressive=flags.rnd_autoregressive,
            ).to(device=flags.device)
    else:
        raise NotImplementedError('Only MiniGrid environments are supported at the moment.')
        model = MarioDoomPolicyNet(env.observation_space.shape, env.action_space.n)
        random_target_network = MarioDoomStateEmbeddingNet(env.observation_space.shape).to(device=flags.device)
        predictor_network = MarioDoomStateEmbeddingNet(env.observation_space.shape).to(device=flags.device)

    buffers = create_buffers(env.observation_space.shape, model.num_actions, flags)
    
    model.share_memory()
    
    initial_agent_state_buffers = []
    for _ in range(flags.num_buffers):
        state = model.initial_state(batch_size=1)
        for t in state:
            t.share_memory_()
        initial_agent_state_buffers.append(state)

    actor_processes = []
    ctx = mp.get_context('spawn')
    free_queue = ctx.SimpleQueue()
    full_queue = ctx.SimpleQueue()

    episode_state_count_dict = dict()
    train_state_count_dict = dict()
    for i in range(flags.num_actors):
        actor = ctx.Process(
            target=act,
            args=(i, free_queue, full_queue, model, buffers, 
                episode_state_count_dict, train_state_count_dict, 
                initial_agent_state_buffers, flags))
        actor.start()
        actor_processes.append(actor)

    if 'MiniGrid' in flags.env: 
        if flags.use_fullobs_policy:
            learner_model = FullObsMinigridPolicyNet(env.observation_space.shape, env.action_space.n)\
                .to(device=flags.device)
        else:
            learner_model = models.MinigridPolicyNet(env.observation_space.shape, env.action_space.n)\
                .to(device=flags.device)
    else:
        raise NotImplementedError('Only MiniGrid environments are supported at the moment.')
        learner_model = MarioDoomPolicyNet(env.observation_space.shape, env.action_space.n)\
            .to(device=flags.device)

    optimizer = torch.optim.RMSprop(
        learner_model.parameters(),
        lr=flags.policy_learning_rate,
        momentum=flags.momentum,
        eps=flags.epsilon,
        alpha=flags.alpha)
    
    predictor_optimizer = torch.optim.RMSprop(
        predictor_network.parameters(),
        lr=flags.learning_rate,
        momentum=flags.momentum,
        eps=flags.epsilon,
        alpha=flags.alpha)
    

    def lr_lambda(epoch):
        return 1 - min(epoch * T * B, flags.total_frames) / flags.total_frames

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    logger = logging.getLogger('logfile')
    stat_keys = [
        'total_loss',
        'mean_episode_return',
        'pg_loss',
        'baseline_loss',
        'entropy_loss',
        'rnd_loss',
        'mean_rewards',
        'mean_intrinsic_rewards',
        'mean_total_rewards',
        'grad_norm_policy',
        'grad_norm_rnd_predictor',
        'lr_policy',
    ]

    logger.info('# Step\t%s', '\t'.join(stat_keys))

    frames, stats = 0, {}


    def batch_and_learn(i, lock=threading.Lock()):
        """Thread target for the learning process."""
        nonlocal frames, stats
        timings = prof.Timings()
        while frames < flags.total_frames:
            timings.reset()
            batch, agent_state = get_batch(free_queue, full_queue, buffers, 
                initial_agent_state_buffers, flags, timings)
            stats = learn(model, learner_model, random_target_network, predictor_network,
                          batch, agent_state, optimizer, predictor_optimizer, scheduler, 
                          flags, frames=frames)
            timings.time('learn')
            with lock:
                to_log = dict(frames=frames)
                to_log.update({k: stats[k] for k in stat_keys})
                if wandb.run is not None:
                    wandb.log(to_log)
                plogger.log(to_log)
                frames += T * B

        if i == 0:
            log.info('Batch and learn: %s', timings.summary())

    for m in range(flags.num_buffers):
        free_queue.put(m)

    threads = []    
    for i in range(flags.num_threads):
        thread = threading.Thread(
            target=batch_and_learn, name='batch-and-learn-%d' % i, args=(i,))
        thread.start()
        threads.append(thread)


    def checkpoint(frames):
        if flags.disable_checkpoint:
            return
        log.info('Saving checkpoint to %s', checkpointpath)
        torch.save({
            'model_state_dict': model.state_dict(),
            'random_target_network_state_dict': random_target_network.state_dict(),
            'predictor_network_state_dict': predictor_network.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'predictor_optimizer_state_dict': predictor_optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'flags': vars(flags),
        }, checkpointpath)

    timer = timeit.default_timer
    try:
        last_checkpoint_time = timer()
        while frames < flags.total_frames:
            start_frames = frames
            start_time = timer()
            time.sleep(5)

            if timer() - last_checkpoint_time > flags.save_interval * 60:  
                checkpoint(frames)
                last_checkpoint_time = timer()

            fps = (frames - start_frames) / (timer() - start_time)
            
            if stats.get('episode_returns', None):
                mean_return = 'Return per episode: %.1f. ' % stats[
                    'mean_episode_return']
            else:
                mean_return = ''

            total_loss = stats.get('total_loss', float('inf'))
            if stats:
                log.info('After %i frames: loss %f @ %.1f fps. Mean Return %.1f. \n Stats \n %s', \
                        frames, total_loss, fps, stats['mean_episode_return'], pprint.pformat(stats))

    except KeyboardInterrupt:
        return  
    else:
        for thread in threads:
            thread.join()
        log.info('Learning finished after %d frames.', frames)
    finally:
        for _ in range(flags.num_actors):
            free_queue.put(None)
        for actor in actor_processes:
            actor.join(timeout=1)

    checkpoint(frames)
    plogger.close()

    videopath = os.path.expandvars(
        os.path.expanduser('%s/%s/%s' % (flags.savedir, flags.xpid,
                                         'animation.mp4')))
    model.load_state_dict(torch.load(checkpointpath)['model_state_dict'])
    test(model, env=None, flags=flags, videopath=videopath)
    wandb.log({'demo': wandb.Video(videopath)})


def test(model, *, env, flags, videopath='animation.mp4'):
    flags.num_buffers = 1
    flags.fix_seed = True

    if env is None:
        env = create_env(flags)

    buffers = create_buffers(env.observation_space.shape, model.num_actions, flags)

    initial_agent_state_buffers = []
    for _ in range(flags.num_buffers):
        state = model.initial_state(batch_size=1)
        for t in state:
            t.share_memory_()
        initial_agent_state_buffers.append(state)

    ctx = mp.get_context('spawn')
    free_queue = ctx.SimpleQueue()
    full_queue = ctx.SimpleQueue()

    for m in range(flags.num_buffers):
        free_queue.put(m)
    free_queue.put(None)

    episode_state_count_dict = dict()
    train_state_count_dict = dict()
    act(0, free_queue, full_queue, model, buffers,
        episode_state_count_dict, train_state_count_dict,
        initial_agent_state_buffers, flags)

    fig = plt.figure()
    ax = plt.gca()
    plt.axis('off')
    camera = Camera(fig)
    env.seed(flags.env_seed)
    obs = env.reset()
    img = env.render('rgb_array', tile_size=32)
    plt.imshow(img)
    camera.snap()
    all_done = False
    for action in buffers['action'][0].tolist():
        obs, reward, done, info = env.step(action)
        all_done = all_done or done
        img = env.render('rgb_array', tile_size=32)
        plt.imshow(img)
        if all_done:
            ax.text(0.5, 1.01, 'beam me up pls', transform=ax.transAxes)
        camera.snap()
    animation = camera.animate()
    animation.save(videopath)


if __name__ == '__main__':
    from src.arguments import parser
    flags = parser.parse_args()
    flags.model = 'recurrent-rnd'

    if flags.test:
        checkpoint = flags.test
        env = create_env(flags)

        model = models.MinigridPolicyNet(env.observation_space.shape, env.action_space.n).to(flags.device)
        model.load_state_dict(torch.load(checkpoint)['model_state_dict'])
        model.share_memory()

        test(model, env=env, flags=flags, videopath='animation.mp4')
    else:
        train(flags)