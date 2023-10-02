# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from pathlib import Path

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(
    description='PyTorch Scalable Agent',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

# General Settings.
parser.add_argument('--env', type=str, default='MiniGrid-MultiRoom-N7-S4-v0',
                    help='Gym environment. Other options are: MiniGrid-ObstructedMaze-2Dlh-v0, \
                    SuperMarioBros-1-1-v0, VizdoomMyWayHomeDense-v0 etc.')
parser.add_argument('--xpid', default=None,
                    help='Experiment id (default: None).')
parser.add_argument('--num_input_frames', default=1, type=int,
                    help='Number of input frames to the model and state embedding including the current frame \
                    When num_input_frames > 1, it will also take the previous num_input_frames - 1 frames as input.')
parser.add_argument('--run_id', default=0, type=int,
                    help='Run id used for running multiple instances of the same HP set \
                    (instead of a different random seed since torchbeast does not accept this).')
parser.add_argument('--seed', default=0, type=int,
                    help='Environment seed.')
parser.add_argument('--save_interval', default=10, type=int, metavar='N',
                    help='Time interval (in minutes) at which to save the model.')    
parser.add_argument('--checkpoint_num_frames', default=10000000, type=int,
                    help='Number of frames for checkpoint to load.')

# Training settings.
parser.add_argument('--disable_checkpoint', action='store_true',
                    help='Disable saving checkpoint.')
parser.add_argument('--save_all_checkpoints', action='store_true',
                    help='Save all checkpoints under unique names instead of just the last one.')
parser.add_argument('--savedir', default='exp/',
                    help='Root dir where experiment data will be saved.')
parser.add_argument('--total_frames', default=30000000, type=int, metavar='T',
                    help='Total environment frames to train for.')
parser.add_argument('--batch_size', default=32, type=int, metavar='B',
                    help='Learner batch size.')
parser.add_argument('--unroll_length', default=100, type=int, metavar='T',
                    help='The unroll length (time dimension).')
parser.add_argument('--queue_timeout', default=1, type=int,
                    metavar='S', help='Error timeout for queue.')
parser.add_argument('--num_buffers', default=80, type=int,
                    metavar='N', help='Number of shared-memory buffers.')
parser.add_argument('--num_actors', default=31, type=int, metavar='N',
                    help='Number of actors.')
parser.add_argument('--num_threads', default=1, type=int,
                    metavar='N', help='Number learner threads.')
parser.add_argument('--disable_cuda', action='store_true',
                    help='Disable CUDA.')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--max_grad_norm', default=40., type=float,
                    metavar='MGN', help='Max norm of gradients.')
parser.add_argument('--megabuffer', action='store_true',
                    help='Store megabuffers with all experience.')

# Loss settings.
parser.add_argument('--entropy_cost', default=0.0001, type=float,
                    help='Entropy cost/multiplier.')
parser.add_argument('--baseline_cost', default=0.5, type=float,
                    help='Baseline cost/multiplier.')
parser.add_argument('--discounting', default=0.99, type=float,
                    help='Discounting factor.')

# Optimizer settings.
parser.add_argument('--learning_rate', default=0.0001, type=float,
                    metavar='LR', help='Learning rate.')
parser.add_argument('--policy_learning_rate', default=0.0001, type=float,
                    metavar='LR', help='Learning rate.')
parser.add_argument('--alpha', default=0.99, type=float,
                    help='RMSProp smoothing constant.')
parser.add_argument('--momentum', default=0, type=float,
                    help='RMSProp momentum.')
parser.add_argument('--epsilon', default=1e-5, type=float,
                    help='RMSProp epsilon.')
parser.add_argument('--rnd_weight_decay', default=0.0001, type=float,
                    help='Weight decay for RND.')
parser.add_argument('--rnd_optimizer', choices=['rmsprop', 'adamw'], default='rmsprop',
                    help='Optimizer to use for RND. AdamW ignores --epsilon, --momentum and --alpha flags.')

# Exploration Settings.
parser.add_argument('--forward_loss_coef', default=10.0, type=float,
                    help='Coefficient for the forward dynamics loss. \
                    This weighs the inverse model loss agains the forward model loss. \
                    Should be between 0 and 1.')
parser.add_argument('--inverse_loss_coef', default=0.1, type=float,
                    help='Coefficient for the forward dynamics loss. \
                    This weighs the inverse model loss agains the forward model loss. \
                    Should be between 0 and 1.')
parser.add_argument('--intrinsic_reward_coef', default=0.01, type=float,
                    help='Coefficient for the intrinsic reward. \
                    This weighs the intrinsic reaward against the extrinsic one. \
                    Should be larger than 0.')
parser.add_argument('--rnd_loss_coef', default=0.1, type=float,
                    help='Coefficient for the RND loss coefficient relative to the IMPALA one.')
parser.add_argument('--rnd_history', default=16, type=int,
                    help='Number of history frames to use for computing the Recurrent RND prediction. 0 means use all available history, no padding.')
parser.add_argument('--rnd_autoregressive', choices=['no', 'forward-target', 'forward-target-difference'],
                    default='forward-target',
                    help='Use past targets as inputs for Recurrent RND.')
parser.add_argument('--rnd_lstm_width', default=128, type=int,
                    help='Width of the LSTM used for Recurrent RND.')
parser.add_argument('--rnd_supervise_everything', type=str2bool, default=True,
                    help='Supervise all intermediary windowed LSTM outputs with the ground truth.')
parser.add_argument('--rnd_supervise_early', type=str2bool,
                    help='Supervise LSTM inputs with the ground truth (assumes some autoregression).')
parser.add_argument('--rnd_global_loss_weight', default=0.1, type=float,
                    help='Weight (0..1) of the global random embedding predictor for Recurrent RND.')
parser.add_argument('--rnd_local_reward_weight', default=0.5, type=float,
                    help='Weight of the local random embedding predictor for Recurrent RND.')
parser.add_argument('--rnd_global_reward_weight', default=0.5, type=float,
                    help='Weight of the local random embedding predictor for Recurrent RND.')
parser.add_argument('--rnd_seed', default=0, type=int,
                    help='Seed for the RND network.')
parser.add_argument('--rnd_meta_seed', default=1516516984916, type=int,
                    help='Seed for the Meta RND task randomizer.')
parser.add_argument('--rnd_extra_steps', default=0, type=int,
                    help='Extra steps with more random inits of the RND network.')
parser.add_argument('--rnd_init', type=str, help='Initialize the RND network with this checkpoint for training.')

# Singleton Environments.
parser.add_argument('--fix_seed', action='store_true',
                    help='Fix the environment seed so that it is \
                    no longer procedurally generated but rather the same layout every episode.')
parser.add_argument('--env_seed', default=1, type=int,
                    help='The seed used to generate the environment if we are using a \
                    singleton (i.e. not procedurally generated) environment.')
parser.add_argument('--no_reward', action='store_true',
                    help='No extrinsic reward. The agent uses only intrinsic reward to learn.')
parser.add_argument('--test', type=Path, help='Test the agent using this policy checkpoint.')
parser.add_argument('--test_rnd', type=Path, help='Test the agent using this RND checkpoint (allows to use reward model and a policy from different checkpoints).')
parser.add_argument('--video', action='store_true', help='Record a video of the agent.')

# Training Models.
parser.add_argument('--model', default='vanilla',
                    choices=['vanilla', 'count', 'curiosity', 'recurrent-rnd', 'rnd', 'ride', 'no-episodic-counts', 'only-episodic-counts'],
                    help='Model used for training the agent.')

# Baselines for AMIGo paper.
parser.add_argument('--use_fullobs_policy', action='store_true',
                    help='Use a full view of the environment as input to the policy network.')
parser.add_argument('--use_fullobs_intrinsic', action='store_true',
                    help='Use a full view of the environment for computing the intrinsic reward.')