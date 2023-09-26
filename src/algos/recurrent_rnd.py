# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
import os
from pathlib import Path
import sys
import threading
import time
import timeit

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
from src.utils import get_batch, log, create_env, create_buffers, act, cat_buffers

MarioDoomPolicyNet = models.MarioDoomPolicyNet
MarioDoomStateEmbeddingNet = models.MarioDoomStateEmbeddingNet

FullObsMinigridPolicyNet = models.FullObsMinigridPolicyNet
FullObsMinigridStateEmbeddingNet = models.FullObsMinigridStateEmbeddingNet

def learn(actor_model,
          model: models.MinigridPolicyNet,
          random_target_network,
          grouped_random_target_network,
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
        done = batch['done'][1:].to(device=flags.device)
        with torch.no_grad():
            global_random_embedding = random_target_network(batch['partial_obs'][1:].to(device=flags.device))
            seed = torch.randint(low=20, high=32788, size=(1,)).item()
            models.reinit_conv2d_(grouped_random_target_network, seed=seed)
            local_random_embedding = grouped_random_target_network(batch['partial_obs'][1:].to(device=flags.device))

        if flags.rnd_autoregressive != 'no':
            global_predicted_embedding, global_rnd_loss = predictor_network(
                inputs=batch['partial_obs'][1:].to(device=flags.device),
                done=done,
                targets=global_random_embedding,
            )

            local_predicted_embedding, local_rnd_loss = predictor_network(
                inputs=batch['partial_obs'][1:].to(device=flags.device),
                done=done,
                targets=local_random_embedding,
            )
            rnd_loss = flags.rnd_global_loss_weight * global_rnd_loss + (1-flags.rnd_global_loss_weight) * local_rnd_loss
        else:
            global_predicted_embedding = predictor_network(
                inputs=batch['partial_obs'][1:].to(device=flags.device),
                done=done,
            )

            # norm over hidden (2), mean over batch (1), sum over time (0)
            rnd_loss = (torch.norm(global_predicted_embedding - global_random_embedding.detach(), dim=2, p=2)).mean(dim=1).sum()

        global_intrinsic_rewards = torch.norm(global_predicted_embedding.detach() - global_random_embedding.detach(), dim=2, p=2)
        local_intrinsic_rewards = torch.norm(local_predicted_embedding.detach() - local_random_embedding.detach(), dim=2, p=2)
        intrinsic_rewards = flags.rnd_global_reward_weight * global_intrinsic_rewards + flags.rnd_local_reward_weight * local_intrinsic_rewards
        intrinsic_rewards *= flags.intrinsic_reward_coef

        rnd_loss = flags.rnd_loss_coef * rnd_loss

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

        stats = {
            'mean_episode_return': torch.mean(episode_returns).item(),
            'total_loss': total_loss.item(),
            'pg_loss': pg_loss.item(),
            'baseline_loss': baseline_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'rnd_loss': rnd_loss.item(),
            'mean_rewards': torch.mean(rewards).item(),
            'mean_intrinsic_rewards': torch.mean(intrinsic_rewards).item(),
            'mean_global_intrinsic_rewards': flags.rnd_global_reward_weight * torch.mean(global_intrinsic_rewards).item(),
            'mean_local_intrinsic_rewards': flags.rnd_local_reward_weight * torch.mean(local_intrinsic_rewards).item(),
            'mean_total_rewards': torch.mean(total_rewards).item(),
            'grad_norm_policy': grad_norm_policy.item(),
            'grad_norm_rnd_predictor': grad_norm_rnd_predictor.item(),
            'lr_policy': scheduler.get_lr()[0],
        } | {f'stepwise/intrinsic_rewards_{i:02d}': r for i, r in enumerate(intrinsic_rewards[:, 0].cpu().tolist())}
        return stats


def train(flags):  
    if flags.xpid is None:
        flags.xpid = f"{flags.model}-{time.strftime('%Y%m%d-%H%M%S')}"

    wandb.init(config=flags)

    plogger = file_writer.FileWriter(
        xpid=flags.xpid,
        xp_args=flags.__dict__,
        rootdir=flags.savedir,
    )

    exproot = Path(flags.savedir) / flags.xpid
    exproot.mkdir(parents=True, exist_ok=True)
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
            random_target_network = models.MinigridStateEmbeddingNet(
                env.observation_space.shape,
                final_activation=False,
            ).to(device=flags.device)
            models.reinit_conv2d_(random_target_network, seed=flags.rnd_seed)
            grouped_random_target_network = models.GroupedStateEmbeddingNet(
                flags.batch_size,
                env.observation_space.shape,
                final_activation=False,
            ).to(device=flags.device)
            predictor_network = models.MinigridStateSequenceNet(
                env.observation_space.shape,
                history=flags.rnd_history,
                autoregressive=flags.rnd_autoregressive,
                hidden_size=flags.rnd_lstm_width,
                supervise_everything=flags.rnd_supervise_everything,
                supervise_early=flags.rnd_supervise_early,
            ).to(device=flags.device)
    else:
        raise NotImplementedError('Only MiniGrid environments are supported at the moment.')
        model = MarioDoomPolicyNet(env.observation_space.shape, env.action_space.n)
        random_target_network = MarioDoomStateEmbeddingNet(env.observation_space.shape).to(device=flags.device)
        predictor_network = MarioDoomStateEmbeddingNet(env.observation_space.shape).to(device=flags.device)

    wandb.config.params_policy = sum(p.numel() for p in model.parameters() if p.requires_grad)
    wandb.config.params_rnd_predictor = sum(p.numel() for p in predictor_network.parameters() if p.requires_grad)
    wandb.config.params_rnd_target = sum(p.numel() for p in random_target_network.parameters() if p.requires_grad)
    wandb.config.params_grouped_rnd_target = sum(p.numel() for p in grouped_random_target_network.parameters() if p.requires_grad)
    log.info('Number of parameters: policy %d, rnd_predictor %d, rnd_target %d, grouped_rnd_target %d',
                wandb.config.params_policy,
                wandb.config.params_rnd_predictor,
                wandb.config.params_rnd_target,
                wandb.config.params_grouped_rnd_target)

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

    if flags.rnd_optimizer == 'adamw':
        predictor_optimizer = torch.optim.AdamW(
            predictor_network.parameters(),
            lr=flags.learning_rate,
            weight_decay=flags.rnd_weight_decay)
    else:
        predictor_optimizer = torch.optim.RMSprop(
            predictor_network.parameters(),
            lr=flags.learning_rate,
            momentum=flags.momentum,
            eps=flags.epsilon,
            alpha=flags.alpha,
            weight_decay=flags.rnd_weight_decay)

    def lr_lambda(epoch):
        return 1 - min(epoch * T * B, flags.total_frames) / flags.total_frames

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    frames, stats = 0, {}


    def batch_and_learn(i, lock=threading.Lock()):
        """Thread target for the learning process."""
        nonlocal frames, stats
        timings = prof.Timings()
        megabuffer = None
        while frames < flags.total_frames:
            timings.reset()
            batch, agent_state = get_batch(free_queue, full_queue, buffers, 
                initial_agent_state_buffers, flags, timings)
            if flags.megabuffer:
                megabuffer = cat_buffers(megabuffer, batch)
            stats = learn(model, learner_model, random_target_network, grouped_random_target_network, predictor_network,
                          batch, agent_state, optimizer, predictor_optimizer, scheduler, 
                          flags, frames=frames)
            timings.time('learn')
            with lock:
                to_log = dict(frames=frames)
                to_log.update({k: stats[k] for k in stats})
                if wandb.run is not None:
                    wandb.log(to_log)
                plogger.log(to_log)
                frames += T * B

            # write megabuffer to disk after T*B*100 frames
            if frames % (T * B * 100) == 0 and megabuffer is not None:
                from safetensors.torch import save_file
                save_file(megabuffer, exproot / f'{frames:010d}.megabuffer')
                megabuffer = None

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
        if flags.save_all_checkpoints:
            path = Path(checkpointpath).parent / f'{frames}.pt'
        else:
            path = Path(checkpointpath)
        log.info('Saving checkpoint to %s', str(path))
        models.reinit_conv2d_(random_target_network, seed=0)
        torch.save({
            'model_state_dict': model.state_dict(),
            'random_target_network_state_dict': random_target_network.state_dict(),
            'predictor_network_state_dict': predictor_network.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'predictor_optimizer_state_dict': predictor_optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'flags': vars(flags),
        }, str(path))
        test(model, random_target_network, predictor_network,
             env=None, flags=flags, videoroot=path.parent,
             seeds=[3])

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
                log.info('After %i frames: loss %f @ %.1f fps. Mean Return %.1f.', \
                        frames, total_loss, fps, stats['mean_episode_return'])

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

    last_checkpoint = torch.load(checkpointpath)
    model.load_state_dict(last_checkpoint['model_state_dict'])
    random_target_network.load_state_dict(last_checkpoint['random_target_network_state_dict'])
    predictor_network.load_state_dict(last_checkpoint['predictor_network_state_dict'])
    test(model, random_target_network, predictor_network,
         env=None, flags=flags, videoroot=Path(checkpointpath).parent, seeds=[3,5,8,13,21,34])


def test(
    model,
    random_target_network,
    predictor_network,
    *,
    env,
    flags,
    videoroot=Path('.'),
    seeds=[3,5,8,13,21,34],
):
    flags = copy.deepcopy(flags)
    flags.num_buffers = 1
    flags.fix_seed = True
    flags.batch_size = 1 # for get_batch

    if env is None:
        env = create_env(flags)

    stat = {}

    videoroot.mkdir(exist_ok=True, parents=True)

    for seed in seeds:
        flags.env_seed = seed
        env.seed(flags.env_seed)
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

        timings = prof.Timings()  

        batch, agent_state = get_batch(free_queue, full_queue, buffers, 
            initial_agent_state_buffers, flags, timings)

        done = batch['done'][1:].to(device=flags.device)
        models.reinit_conv2d_(random_target_network, seed=16384)
        random_embedding = random_target_network(batch['partial_obs'][1:].to(device=flags.device))
        if flags.rnd_autoregressive != 'no':
            predicted_embedding, _ = predictor_network(
                inputs=batch['partial_obs'][1:].to(device=flags.device),
                done=done,
                targets=random_embedding.detach(),
            )
        else:
            predicted_embedding = predictor_network(
                inputs=batch['partial_obs'][1:].to(device=flags.device),
                done=done,
            )

        intrinsic_rewards = flags.intrinsic_reward_coef * torch.norm(predicted_embedding.detach() - random_embedding.detach(), dim=2, p=2)
        intrinsic_rewards_list = intrinsic_rewards.view(-1).cpu().numpy().tolist()
        (videoroot / f'{seed}.rewards').write_text('\n'.join(map(str, intrinsic_rewards_list)))

        # wandb.Table of rewards
        reward_table = wandb.Table(columns=['step', 'reward'], data=[[i, r] for i, r in enumerate(intrinsic_rewards.view(-1).cpu().tolist())])
        stat[f'test/rewards-per-step-{seed}'] = wandb.plot.line(reward_table, 'step', 'reward', title=f'rewards per step for seed {seed}')

        # returns
        episode_returns = batch['episode_return'][batch['done']]
        stat[f'test/returns-{seed}'] = torch.mean(episode_returns).item()
        stat[f'test/ext-rewards-{seed}'] = batch['reward'].sum().item()

        actions = { # MiniGrid
            0: 'left',
            1: 'right',
            2: 'forward',
            3: 'pickup UNUSED',
            4: 'drop UNUSED',
            5: 'toggle',
            6: 'done UNUSED',
        }
        if flags.video:
            fig, (axl, axr) = plt.subplots(2,1,gridspec_kw={'height_ratios': [2, 1]}, figsize=(8,16))
            #plt.axis('off')
            plt.tight_layout()
            camera = Camera(fig)
            env.seed(flags.env_seed)
            obs = env.reset()
            img = env.render('rgb_array', tile_size=32)
            axl.imshow(img)
            axr.set_title('dopamine')
            camera.snap()
            #for action in buffers['action'][0].tolist():
            for i, action in enumerate(batch['action'][1:,0].tolist()):
                obs, reward, done, info = env.step(action)
                if done:
                    break
                img = env.render('rgb_array', tile_size=32)
                axl.imshow(img)
                axr.text(0.1, 1.0, actions[action], transform=axr.transAxes, fontsize='large')
                axr.plot(intrinsic_rewards_list[:i+1], marker='x')
                camera.snap()
            animation = camera.animate()
            animation.save(str(videoroot / f'{seed}.mp4'))
            stat[f'test/video-{seed}'] = wandb.Video(str(videoroot / f'{seed}.mp4'))

            print('saved', videoroot / f'{seed}.mp4', videoroot / f'{seed}.rewards')
        else:
            print('saved', videoroot / f'{seed}.rewards')

    if wandb.run is not None:
        wandb.log(stat)


if __name__ == '__main__':
    from src.arguments import parser
    flags = parser.parse_args()
    flags.model = 'recurrent-rnd'

    if flags.test:
        wandb.init(config=flags)
        env = create_env(flags)

        checkpoint = torch.load(str(flags.test))

        model = models.MinigridPolicyNet(env.observation_space.shape, env.action_space.n).to(flags.device)
        model.load_state_dict(checkpoint['model_state_dict'])

        if flags.test_rnd:
            # use a different checkpoint for the reward model
            checkpoint = torch.load(str(flags.test_rnd))

        random_target_network = models.MinigridStateEmbeddingNet(
            env.observation_space.shape,
            final_activation=False,
        ).to(device=flags.device)
        random_target_network.load_state_dict(checkpoint['random_target_network_state_dict'])

        predictor_network = models.MinigridStateSequenceNet(
            env.observation_space.shape,
            history=flags.rnd_history,
            autoregressive=flags.rnd_autoregressive,
            hidden_size=flags.rnd_lstm_width,
            supervise_everything=flags.rnd_supervise_everything,
            supervise_early=flags.rnd_supervise_early,
        ).to(device=flags.device)
        predictor_network.load_state_dict(checkpoint['predictor_network_state_dict'])

        test(model, random_target_network, predictor_network, env=None, flags=flags, videoroot=flags.test.parent)
    else:
        train(flags)
