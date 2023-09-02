# Agents with intrinsic rewards

Read more about curiosity in https://people.idsia.ch/~juergen/artificial-curiosity-since-1990.html

Environment setup.

```bash
python -m venv .venv
.venv/bin/pip install wandb celluloid
.venv/bin/pip install virtualenv
.venv/bin/pip install -r new_requirements.txt -I
.venv/bin/pip install -e ./gym-minigrid
```

Train [RND](https://openai.com/research/reinforcement-learning-with-prediction-based-rewards) agent.

```bash
ulimit -n 64000 # prevents RuntimeError: unable to open shared memory object </torch_13695_1684771047_984> in read-write mode: Too many open files (24)
env OMP_NUM_THREADS=1 .venv/bin/python main.py --model rnd --env MiniGrid-MultiRoom-N7-S4-v0 --total_frames 100000000 --intrinsic_reward_coef 0.1 --entropy_cost 0.0001
```

Try the environment with manual control. Use `ssh -X` with XQuartz running on a Mac if needed.

```
.venv/bin/python gym-minigrid/manual_control.py # whole map view
.venv/bin/python gym-minigrid/manual_control.py --agent_view
```

Profile the environment. `py-spy` that write flamegraph SVGs on disk. This revealed that the original experience collection agents were spending most of their time running their policies on CPU. Another use let me know that actor workers are waiting a lot of time synchronous cuda transfer of useless tensors.

```
pip install py-spy
py-spy record --pid 3081044 -r 33 -s
```

Perform a rollout from a trained policy. This writes `animation.mp4` on disk.

```
python -m src.algos.rnd exp/curiosity-20230901-121847/model.tar --fix_seed
```

Original README contents below:

## RIDE: Rewarding Impact-Driven Exploration for Procedurally-Generated Environments

This is an implementation of the method proposed in 

<a href="https://openreview.net/pdf?id=rkg-TJBFPB">RIDE: Rewarding Impact-Driven Exploration for Procedurally-Generated Environments</a> 

by Roberta Raileanu and Tim Rocktäschel, published at ICLR 2020. 

We propose a novel type of intrinsic reward which encourges the agent to take actions that result in significant changes to its representation of the environment state.

The code includes all the baselines and ablations used in the paper. 

The code was also used to run the baselines in [Learning with AMIGO:
Adversarially Motivated Intrinsic Goals](https://arxiv.org/pdf/2006.12122.pdf). 
See [the associated repo](https://github.com/facebookresearch/adversarially-motivated-intrinsic-goals) for instructions on how to reproduce the results from that paper.

## Citation
If you use this code in your own work, please cite our paper:
```
@inproceedings{
  Raileanu2020RIDE:,
  title={{RIDE: Rewarding Impact-Driven Exploration for Procedurally-Generated Environments}},
  author={Roberta Raileanu and Tim Rockt{\"{a}}schel},
  booktitle={International Conference on Learning Representations},
  year={2020},
  url={https://openreview.net/forum?id=rkg-TJBFPB}
}
```

## Installation

```
# create a new conda environment
conda create -n ride python=3.7
conda activate ride 

# install dependencies
git clone git@github.com:facebookresearch/impact-driven-exploration.git
cd impact-driven-exploration
pip install -r requirements.txt

# install MiniGrid
cd gym-minigrid
python setup.py install
```

## Train RIDE on MiniGrid
```
cd impact-driven-exploration

OMP_NUM_THREADS=1 python main.py --model ride --env MiniGrid-MultiRoom-N7-S4-v0 --total_frames 30000000 --intrinsic_reward_coef 0.1 --entropy_cost 0.0005 --num_actors 3

OMP_NUM_THREADS=1 python main.py --model ride --env MiniGrid-MultiRoomNoisyTV-N7-S4-v0 --total_frames 30000000 --intrinsic_reward_coef 0.1 --entropy_cost 0.0005

OMP_NUM_THREADS=1 python main.py --model ride --env MiniGrid-MultiRoom-N7-S8-v0 --total_frames 30000000 --intrinsic_reward_coef 0.5 --entropy_cost 0.001

OMP_NUM_THREADS=1 python main.py --model ride --env MiniGrid-MultiRoom-N10-S4-v0 --total_frames 30000000 --intrinsic_reward_coef 0.1 --entropy_cost 0.0005

OMP_NUM_THREADS=1 python main.py --model ride --env MiniGrid-KeyCorridor-S3-R3-v0 --total_frames 30000000 --intrinsic_reward_coef 0.1 --entropy_cost 0.0005

OMP_NUM_THREADS=1 python main.py --model ride --env MiniGrid-ObstructedMaze-2Dlh-v0 --total_frames 100000000 --intrinsic_reward_coef 0.5 --entropy_cost 0.001

OMP_NUM_THREADS=1 python main.py --model ride --env MiniGrid-MultiRoom-N10-S10-v0 --total_frames 100000000 --intrinsic_reward_coef 0.5 --entropy_cost 0.001

OMP_NUM_THREADS=1 python main.py --model ride --env MiniGrid-MultiRoom-N12-S10-v0 --total_frames 100000000 --intrinsic_reward_coef 0.5 --entropy_cost 0.001

```
To train RIDE on the other MiniGrid environments used in our paper, replace the ```--env``` argument above with each of the following:
```
MiniGrid-MultiRoom-N7-S4-v0
MiniGrid-MultiRoomNoisyTV-N7-S4-v0
MiniGrid-MultiRoom-N7-S8-v0
MiniGrid-MultiRoom-N10-S4-v0
MiniGrid-MultiRoom-N10-S10-v0
MiniGrid-MultiRoom-N12-S10-v0
MiniGrid-ObstructedMaze-2Dlh-v0 
MiniGrid-KeyCorridorS3R3-v0
```
Make sure to use the best hyperparameters for each environment, as listed in the paper. 

To run different seeds for a model, change the ```--run_id``` argument.

## Overview of RIDE
![RIDE Overview](/figures/ride_overview.png)

## Results on MiniGrid
![MiniGrid Results](/figures/ride_results.png)

## Analysis of RIDE
![Intrinsic Reward Heatmaps](/figures/ride_analysis.png)

![State Visitation Heatmaps](/figures/ride_analysis_counts.png)

## Acknowledgements
Our vanilla RL algorithm is based on [Torchbeast](https://github.com/facebookresearch/torchbeast), which is an open source implementation of IMPALA.

## License
This code is under the CC-BY-NC 4.0 (Attribution-NonCommercial 4.0 International) license.
