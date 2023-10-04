# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
import os
from pathlib import Path
import threading
import time
import timeit

import wandb

import torch
from torch import nn
from torch.nn import functional as F

from src.core import prof

import src.models as models
from src.utils import log

from .recurrent_rnd import meta_predictor_step


class Dataset(torch.utils.data.Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)*3200

    def __getitem__(self, index):
        path = self.paths[index // 3200]
        index = index % 3200
        from safetensors.torch import load_file
        data = load_file(path)
        data = {k: data[k][:, index, ...] for k in ['partial_obs', 'done', 'episode_state_count']}
        return data

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
    done = batch['done'][1:].to(device=flags.device)

    if flags.rnd_autoregressive != 'no':
        intrinsic_rewards, rnd_loss = meta_predictor_step(
            random_target_network=random_target_network,
            grouped_random_target_network=grouped_random_target_network,
            predictor_network=predictor_network,
            batch=batch,
            done=done,
            flags=flags,
            generator=None,
            ignore_global=True
        )
    else:
        with torch.no_grad():
            models.reinit_conv2d_(random_target_network, seed=flags.rnd_seed)
            global_random_embedding = random_target_network(batch['partial_obs'][1:].to(device=flags.device))

        global_predicted_embedding = predictor_network(
            inputs=batch['partial_obs'][1:].to(device=flags.device),
            done=done,
        )

        # norm over hidden (2), mean over batch (1), sum over time (0)
        rnd_loss = (torch.norm(global_predicted_embedding - global_random_embedding.detach(), dim=2, p=2)).mean(dim=1).sum()

        intrinsic_rewards = flags.intrinsic_reward_coef * torch.norm(global_predicted_embedding.detach() - global_random_embedding.detach(), dim=2, p=2)

    total_loss = flags.rnd_loss_coef * rnd_loss

    predictor_optimizer.zero_grad()
    total_loss.backward()
    grad_norm_rnd_predictor = nn.utils.clip_grad_norm_(predictor_network.parameters(), flags.max_grad_norm)
    predictor_optimizer.step()
    scheduler.step()

    stats = {
        'rnd_loss': total_loss.item(),
        'grad_norm_rnd_predictor': grad_norm_rnd_predictor.item(),
        'lr': scheduler.get_last_lr()[0],
    } | {f'train-stepwise/intrinsic_rewards_{i:02d}': r for i, r in enumerate(intrinsic_rewards[:, 0].cpu().tolist())}
    return stats


def train(flags):  
    if flags.xpid is None:
        flags.xpid = f"{flags.model}-{time.strftime('%Y%m%d-%H%M%S')}"

    wandb.init(config=flags)

    env_observation_space_shape = (7, 7, 3)

    exproot = Path(flags.savedir) / flags.xpid
    exproot.mkdir(parents=True, exist_ok=True)
    checkpointpath = os.path.expandvars(
        os.path.expanduser('%s/%s/%s' % (flags.savedir, flags.xpid,
                                         'model.tar')))

    T = flags.unroll_length
    B = flags.batch_size

    flags.device = torch.device('cuda')

    random_target_network = models.MinigridStateEmbeddingNet(
        env_observation_space_shape,
        final_activation=False,
    ).to(device=flags.device)
    grouped_random_target_network = models.GroupedStateEmbeddingNet(
        flags.batch_size,
        env_observation_space_shape,
        final_activation=False,
    ).to(device=flags.device)
    predictor_network = models.MinigridStateSequenceNet(
        env_observation_space_shape,
        history=flags.rnd_history,
        autoregressive=flags.rnd_autoregressive,
        hidden_size=flags.rnd_lstm_width,
        supervise_everything=flags.rnd_supervise_everything,
        supervise_early=flags.rnd_supervise_early,
    ).to(device=flags.device)

    wandb.config.params_rnd_predictor = sum(p.numel() for p in predictor_network.parameters() if p.requires_grad)
    wandb.config.params_rnd_target = sum(p.numel() for p in random_target_network.parameters() if p.requires_grad)
    log.info('Number of parameters: rnd_predictor %d, rnd_target %d',
                wandb.config.params_rnd_predictor,
                wandb.config.params_rnd_target)

    predictor_optimizer = torch.optim.AdamW(
        predictor_network.parameters(),
        lr=flags.learning_rate,
        weight_decay=flags.rnd_weight_decay)

    def lr_lambda(epoch):
        #return 1 - min(epoch * T * B, flags.total_frames) / flags.total_frames
        return 1

    scheduler = torch.optim.lr_scheduler.LambdaLR(predictor_optimizer, lr_lambda)

    frames, stats = 0, {}

    loader = torch.utils.data.DataLoader(
        Dataset(sorted(Path(flags.train).glob('*.megabuffer'))),
        batch_size=flags.batch_size,
        shuffle=True,
        num_workers=24,
        pin_memory=True,
        drop_last=True,
    )
    iter_loader = iter(loader)

    timings = prof.Timings()

    def checkpoint(frames):
        if flags.disable_checkpoint:
            return
        if flags.save_all_checkpoints:
            path = Path(checkpointpath).parent / f'{frames}.pt'
        else:
            path = Path(checkpointpath)
        log.info('Saving checkpoint to %s', str(path))
        torch.save({
            'random_target_network_state_dict': random_target_network.state_dict(),
            'predictor_network_state_dict': predictor_network.state_dict(),
            'predictor_optimizer_state_dict': predictor_optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'flags': vars(flags),
        }, str(path))

    timer = timeit.default_timer
    start_frames = 0
    start_time = timer()
    while frames < flags.total_frames:
        timings.reset()
        try:
            batch = next(iter_loader)
        except StopIteration:
            iter_loader = iter(loader)
            batch = next(iter_loader)
            log.info('Resetting dataloader')

        batch['done'] = batch['done'].T # (B, T) -> (T, B)
        batch['partial_obs'] = batch['partial_obs'].transpose(0,1) # (B, T, 7, 7, 3) -> (T, B, 7, 7, 3)

        stats = learn(None, None, random_target_network, grouped_random_target_network, predictor_network,
                        batch, None, None, predictor_optimizer, scheduler, 
                        flags, frames=frames)
        timings.time('learn')

        frames += T * B

        to_log = dict(frames=frames)
        to_log.update(stats)

        if frames % (T * B * 100) == 0:
            fps = (frames - start_frames) / (timer() - start_time)
            log.info('After %i frames (%i updates): loss %f @ %.1f fps. LR %.6f.', \
                    frames, frames // (T*B), stats['rnd_loss'], fps, scheduler.get_last_lr()[0])
            start_frames = frames
            start_time = timer()
            to_log.update(evaluate(None, random_target_network, predictor_network,
             env=None, flags=flags,
             seeds=[3, flags.rnd_seed, 20]))

        if frames % (T * B * 1000) == 0:
            checkpoint(frames)

        if wandb.run is not None:
            wandb.log(to_log)

    last_checkpoint = torch.load(checkpointpath)
    random_target_network.load_state_dict(last_checkpoint['random_target_network_state_dict'])
    predictor_network.load_state_dict(last_checkpoint['predictor_network_state_dict'])
    evaluate(None, random_target_network, predictor_network,
         env=None, flags=flags, seeds=[3,5,8,13,21,34])
    checkpoint(frames)



@torch.no_grad()
def evaluate(
    model,
    random_target_network,
    predictor_network,
    *,
    env,
    flags,
    seeds=[3,0]
):
    flags = copy.deepcopy(flags)
    flags.device = torch.device('cuda')

    stats = {}

    for filename in ["0000320000.megabuffer", "0006400000.megabuffer"]:        
        loc = Path(flags.valid) / filename
        loader = torch.utils.data.DataLoader(
            Dataset([loc]),
            batch_size=flags.batch_size,
            shuffle=False,
            num_workers=24,
            pin_memory=True,
            drop_last=False,
        )

        for seed in seeds:
            models.reinit_conv2d_(random_target_network, seed=seed)

            stats[f'valid/rnd_loss_{loc.stem}_{seed}'] = 0.

            for i, batch in enumerate(loader):
                batch['done'] = batch['done'].T # (B, T) -> (T, B)
                batch['partial_obs'] = batch['partial_obs'].transpose(0,1) # (B, T, 7, 7, 3) -> (T, B, 7, 7, 3)
                batch['episode_state_count'] = batch['episode_state_count'].transpose(0,1) # (B, T, C) -> (T, B, C)

                done = batch['done'][1:].to(device=flags.device)
                random_embedding = random_target_network(batch['partial_obs'][1:].to(device=flags.device))

                if flags.rnd_autoregressive != 'no':
                    predicted_embedding, rnd_loss = predictor_network(
                        inputs=batch['partial_obs'][1:].to(device=flags.device),
                        done=done,
                        targets=random_embedding.detach(),
                    )
                else:
                    predicted_embedding = predictor_network(
                        inputs=batch['partial_obs'][1:].to(device=flags.device),
                        done=done,
                    )

                    # norm over hidden (2), mean over batch (1), sum over time (0)
                    rnd_loss = (torch.norm(predicted_embedding - random_embedding.detach(), dim=2, p=2)).mean(dim=1).sum()

                intrinsic_rewards = flags.intrinsic_reward_coef * torch.norm(predicted_embedding.detach() - random_embedding.detach(), dim=2, p=2)

                stats[f'valid/rnd_loss_{loc.stem}_{seed}'] += flags.rnd_loss_coef * rnd_loss.item()

                if i == 0:
                    #reward_table = wandb.Table(columns=['step', 'reward'], data=[[step, r] for step, r in enumerate(intrinsic_rewards[:, i].view(-1).cpu().tolist())])
                    #stats[f'valid/rewards_per_step_{loc.stem}-0_{seed}'] = wandb.plot.line(reward_table, 'step', 'reward', title=f'rewards per step for seed {seed}')

                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots(1,1,figsize=(7,5))
                    plt.tight_layout()
                    ax.plot(intrinsic_rewards[:, i].view(-1).cpu().tolist(), label='intrinsic_rewards')
                    ax.plot(batch['episode_state_count'][1:, i]*10, label='episode_state_count*10')
                    plt.legend()
                    # write to wandb
                    stats[f'valid/rewards_per_step_{loc.stem}-0_{seed}+çounts'] = wandb.Image(fig)
                    plt.close(fig)

            stats[f'valid/rnd_loss_{loc.stem}_{seed}'] /= len(loader)

    return stats

if __name__ == '__main__':
    from src.arguments import parser
    parser.add_argument('--train', type=Path, required=True, help='Path to the train dataset.')
    parser.add_argument('--valid', type=Path, required=True, help='Path to the validation dataset.')
    flags = parser.parse_args()
    flags.model = 'pretrain'

    train(flags)
