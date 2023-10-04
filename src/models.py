# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn 
from torch.nn import functional as F
import numpy as np 


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def reinit_conv2d_(network, seed=0):
    state = torch.get_rng_state()
    torch.manual_seed(seed)
    from src.models import init
    init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                           constant_(x, 0), nn.init.calculate_gain('relu'))

    for name, mod in network.named_modules():
        if isinstance(mod, nn.Conv2d):
            init_(mod)
    torch.set_rng_state(state)


class FullObsMinigridPolicyNet(nn.Module):
    def __init__(self, observation_shape, num_actions):
        super(FullObsMinigridPolicyNet, self).__init__()
        self.observation_shape = observation_shape
        self.num_actions = num_actions

        self.use_index_select = True
        self.obj_dim = 5
        self.col_dim = 3
        self.con_dim = 2
        self.agent_loc_dim = 10
        self.num_channels = (self.obj_dim + self.col_dim + self.con_dim)
                
        self.embed_object = nn.Embedding(11, self.obj_dim)
        self.embed_color = nn.Embedding(6, self.col_dim)
        self.embed_contains = nn.Embedding(4, self.con_dim)
        self.embed_agent_loc = nn.Embedding(self.observation_shape[0]*self.observation_shape[1] + 1, self.agent_loc_dim)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                            constant_(x, 0), nn.init.calculate_gain('relu'))

        ##Because Fully_observed
        self.feat_extract = nn.Sequential(
            init_(nn.Conv2d(in_channels=self.num_channels, out_channels=32, kernel_size=(3, 3), stride=2, padding=1)),
                nn.ELU(),
            init_(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2, padding=1)),
                nn.ELU(),
            init_(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2, padding=1)),
                nn.ELU(),
            init_(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2, padding=1)),
                nn.ELU(),
            init_(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2, padding=1)),
                nn.ELU(),
            )
        

        self.fc = nn.Sequential(
            init_(nn.Linear(32 + self.agent_loc_dim + self.obj_dim + self.col_dim, 1024)),
            nn.ReLU(),
            init_(nn.Linear(1024, 1024)),
            nn.ReLU(),
        )

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                            constant_(x, 0))

        self.policy = init_(nn.Linear(1024, self.num_actions))
        self.baseline = init_(nn.Linear(1024, 1))


    def initial_state(self, batch_size):
        return tuple()
  
    def _select(self, embed, x):
        if self.use_index_select:
            out = embed.weight.index_select(0, x.reshape(-1))
            # handle reshaping x to 1-d and output back to N-d
            return out.reshape(x.shape +(-1,))
        else:
            return embed(x) 

    def create_embeddings(self, x, id):
        #indices = torch.tensor([i for i in range(x.shape[3]) if i%3==id])
        #object_ids = torch.index_select(x, 3, indices)
        if id == 0:
            objects_emb = self._select(self.embed_object, x[:,:,:,id::3])
        elif id == 1:
            objects_emb = self._select(self.embed_color, x[:,:,:,id::3])
        elif id == 2:
            objects_emb = self._select(self.embed_contains, x[:,:,:,id::3])
        embeddings = torch.flatten(objects_emb, 3, 4)
        return embeddings

    def agent_loc(self, frames):
        T, B, *_ = frames.shape
        agent_location = torch.flatten(frames, 2, 3)
        agent_location = agent_location[:,:,:,0] 
        agent_location = (agent_location == 10).nonzero() #select object id
        agent_location = agent_location[:,2]
        agent_location = agent_location.view(T,B,1)
        return agent_location 

    def forward(self, inputs, core_state=()):
        # -- [unroll_length x batch_size x height x width x channels]
        x = inputs["frame"]
        T, B, *_ = x.shape
       
        # -- [unroll_length*batch_size x height x width x channels]
        x = torch.flatten(x, 0, 1)  # Merge time and batch.
        agent_loc = self.agent_loc(inputs["frame"])
        carried_col = inputs["carried_col"]
        carried_obj = inputs["carried_obj"]

        x = x.long()
        agent_loc = agent_loc.long()
        carried_obj = carried_obj.long()
        carried_col = carried_col.long()
        # -- [B x H x W x K]
        x = torch.cat([self.create_embeddings(x, 0), self.create_embeddings(x, 1), self.create_embeddings(x, 2)], dim = 3)
        agent_loc_emb = self._select(self.embed_agent_loc, agent_loc)
        carried_obj_emb = self._select(self.embed_object, carried_obj)
        carried_col_emb = self._select(self.embed_color, carried_col)

        # -- [unroll_length*batch_size x channels x width x height]
        x = x.transpose(1, 3)
        # -- [B x K x W x H]

        agent_loc_emb = agent_loc_emb.view(T * B, -1)
        carried_obj_emb = carried_obj_emb.view(T * B, -1)
        carried_col_emb = carried_col_emb.view(T * B, -1) 

        x = self.feat_extract(x)
        x = x.view(T * B, -1)
        union = torch.cat([x, agent_loc_emb, carried_obj_emb, carried_col_emb], dim=1)
        core_input = self.fc(union)
        
        core_output = core_input
        core_state = tuple()

        policy_logits = self.policy(core_output)
        baseline = self.baseline(core_output)
        

        if self.training:
            action = torch.multinomial(F.softmax(policy_logits, dim=1), num_samples=1)
        else:
            # Don't sample when testing.
            action = torch.argmax(policy_logits, dim=1)

        policy_logits = policy_logits.view(T, B, self.num_actions)
        baseline = baseline.view(T, B)
        action = action.view(T, B)

        return dict(policy_logits=policy_logits, baseline=baseline, action=action), core_state

class FullObsMinigridStateEmbeddingNet(nn.Module):
    def __init__(self, observation_shape):
        super(FullObsMinigridStateEmbeddingNet, self).__init__()
        self.observation_shape = observation_shape

        self.use_index_select = True
        self.obj_dim = 5
        self.col_dim = 3
        self.con_dim = 2
        self.agent_loc_dim = 10
        self.num_channels = (self.obj_dim + self.col_dim + self.con_dim) 
        
        self.embed_object = nn.Embedding(11, self.obj_dim)
        self.embed_color = nn.Embedding(6, self.col_dim)
        self.embed_contains = nn.Embedding(4, self.con_dim)
        self.embed_agent_loc = nn.Embedding(self.observation_shape[0]*self.observation_shape[1] + 1, self.agent_loc_dim)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                            constant_(x, 0), nn.init.calculate_gain('relu'))

        ##Because Fully_observed
        self.feat_extract = nn.Sequential(
            init_(nn.Conv2d(in_channels=self.num_channels, out_channels=32, kernel_size=(3, 3), stride=2, padding=1)),
                nn.ELU(),
            init_(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2, padding=1)),
                nn.ELU(),
            init_(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2, padding=1)),
                nn.ELU(),
            init_(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2, padding=1)),
                nn.ELU(),
            init_(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2, padding=1)),
                nn.ELU(),
            )
        
        self.fc = nn.Sequential(
            init_(nn.Linear(32 + self.agent_loc_dim + self.obj_dim + self.col_dim, 128)),
            nn.ReLU(),
            init_(nn.Linear(128, 128)),
            nn.ReLU(),
        )
        
    def _select(self, embed, x):
        if self.use_index_select:
            out = embed.weight.index_select(0, x.reshape(-1))
            # handle reshaping x to 1-d and output back to N-d
            return out.reshape(x.shape +(-1,))
        else:
            return embed(x) 

    def create_embeddings(self, x, id):
        #indices = torch.tensor([i for i in range(x.shape[3]) if i%3==id])
        #object_ids = torch.index_select(x, 3, indices)
        if id == 0:
            objects_emb = self._select(self.embed_object, x[:,:,:,id::3])
        elif id == 1:
            objects_emb = self._select(self.embed_color, x[:,:,:,id::3])
        elif id == 2:
            objects_emb = self._select(self.embed_contains, x[:,:,:,id::3])
        embeddings = torch.flatten(objects_emb, 3, 4)
        return embeddings

    def agent_loc(self, frames):
        T, B, *_ = frames.shape
        agent_location = torch.flatten(frames, 2, 3)
        agent_location = agent_location[:,:,:,0] 
        agent_location = (agent_location == 10).nonzero() #select object id
        agent_location = agent_location[:,2]
        agent_location = agent_location.view(T,B,1)
        return agent_location 

    def forward(self, inputs, next_state=False):
        # -- [unroll_length x batch_size x height x width x channels]
        if next_state:
            x = inputs["frame"][1:]
        else:
            x = inputs["frame"][:-1]
        T, B, *_ = x.shape
       
        # -- [unroll_length*batch_size x height x width x channels]
        x = torch.flatten(x, 0, 1)  # Merge time and batch.
        if next_state:
            agent_loc = self.agent_loc(inputs["frame"][1:])
            carried_col = inputs["carried_col"][1:]
            carried_obj = inputs["carried_obj"][1:]
        else:
            agent_loc = self.agent_loc(inputs["frame"][:-1])
            carried_col = inputs["carried_col"][:-1]
            carried_obj = inputs["carried_obj"][:-1]

        x = x.long()
        agent_loc = agent_loc.long()
        carried_obj = carried_obj.long()
        carried_col = carried_col.long()
        # -- [B x H x W x K]
        x = torch.cat([self.create_embeddings(x, 0), self.create_embeddings(x, 1), self.create_embeddings(x, 2)], dim = 3)
        agent_loc_emb = self._select(self.embed_agent_loc, agent_loc)
        carried_obj_emb = self._select(self.embed_object, carried_obj)
        carried_col_emb = self._select(self.embed_color, carried_col)

        # -- [unroll_length*batch_size x channels x width x height]
        x = x.transpose(1, 3)
        # -- [B x K x W x H]

        agent_loc_emb = agent_loc_emb.view(T * B, -1)
        carried_obj_emb = carried_obj_emb.view(T * B, -1)
        carried_col_emb = carried_col_emb.view(T * B, -1) 

        x = self.feat_extract(x)
        x = x.view(T * B, -1)
        union = torch.cat([x, agent_loc_emb, carried_obj_emb, carried_col_emb], dim=1)
        core_input = self.fc(union)

        return core_input


class MinigridPolicyNet(nn.Module):
    def __init__(self, observation_shape, num_actions):
        super(MinigridPolicyNet, self).__init__()
        self.observation_shape = observation_shape
        self.num_actions = num_actions

        init_ = lambda m: init(m, nn.init.orthogonal_, 
            lambda x: nn.init.constant_(x, 0), 
            nn.init.calculate_gain('relu'))
        
        self.feat_extract = nn.Sequential(
            init_(nn.Conv2d(in_channels=self.observation_shape[2], out_channels=32, kernel_size=(3, 3), stride=2, padding=1)),
            nn.ELU(),
            init_(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2, padding=1)),
            nn.ELU(),
            init_(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2, padding=1)),
            nn.ELU(),
        )
    
        self.fc = nn.Sequential(
            init_(nn.Linear(32, 1024)),
            nn.ReLU(),
            init_(nn.Linear(1024, 1024)),
            nn.ReLU(),
        )

        self.core = nn.LSTM(1024, 1024, 2)

        init_ = lambda m: init(m, nn.init.orthogonal_, 
            lambda x: nn.init.constant_(x, 0))

        self.policy = init_(nn.Linear(1024, self.num_actions))
        self.baseline = init_(nn.Linear(1024, 1))


    def initial_state(self, batch_size):
        device = next(self.parameters()).device
        return tuple(torch.zeros(self.core.num_layers, batch_size, 
                                 self.core.hidden_size, device=device) for _ in range(2))


    def forward(self, inputs, core_state=()):
        # -- [unroll_length x batch_size x height x width x channels]
        x = inputs['partial_obs']
        T, B, *_ = x.shape

        # -- [unroll_length*batch_size x height x width x channels]
        x = torch.flatten(x, 0, 1)  # Merge time and batch.

        x = x.float() #/ 255.0
        
        # -- [unroll_length*batch_size x channels x width x height]
        x = x.transpose(1, 3)
        x = self.feat_extract(x)
        x = x.view(T * B, -1)
        core_input = self.fc(x)

        core_input = core_input.view(T, B, -1)
        core_output_list = []
        notdone = (~inputs['done']).float()
        for input, nd in zip(core_input.unbind(), notdone.unbind()):
            nd = nd.view(1, -1, 1)
            core_state = tuple(nd * s for s in core_state)
            output, core_state = self.core(input.unsqueeze(0), core_state)
            core_output_list.append(output)
        core_output = torch.flatten(torch.cat(core_output_list), 0, 1)
        
        policy_logits = self.policy(core_output)
        baseline = self.baseline(core_output)

        if self.training:
            action = torch.multinomial(
                F.softmax(policy_logits, dim=1), num_samples=1)
        else:
            action = torch.argmax(policy_logits, dim=1)

        policy_logits = policy_logits.view(T, B, self.num_actions)
        baseline = baseline.view(T, B)
        action = action.view(T, B)

        return dict(policy_logits=policy_logits, baseline=baseline, 
                    action=action), core_state


def make_feat_extract(in_channels, out_channels, final_activation=True):
    init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                        constant_(x, 0), nn.init.calculate_gain('relu'))

    if final_activation:
        return nn.Sequential(
            init_(nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=(3, 3), stride=2, padding=1)),
            nn.ELU(),
            init_(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2, padding=1)),
            nn.ELU(),
            init_(nn.Conv2d(in_channels=32, out_channels=out_channels, kernel_size=(3, 3), stride=2, padding=1)),
            nn.ELU(),
        )
    else:
        return nn.Sequential(
            init_(nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=(3, 3), stride=2, padding=1)),
            nn.ELU(),
            init_(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2, padding=1)),
            nn.ELU(),
            init_(nn.Conv2d(in_channels=32, out_channels=out_channels, kernel_size=(3, 3), stride=2, padding=1)),
        )


class GroupedStateEmbeddingNet(nn.Module):
    def __init__(self, batch_size, observation_shape, final_activation=False):
        super().__init__()
        assert final_activation == False

        self.observation_shape = observation_shape
        in_channels, out_channels = observation_shape[2], 128
        self.in_channels, self.out_channels = in_channels, out_channels

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        N = batch_size
        self.conv = nn.Sequential(
            init_(nn.Conv2d(in_channels=N*in_channels, out_channels=N*32, groups=N, kernel_size=(3, 3), stride=2, padding=1)),
            nn.ELU(),
            init_(nn.Conv2d(in_channels=N*32, out_channels=N*32, groups=N, kernel_size=(3, 3), stride=2, padding=1)),
            nn.ELU(),
            init_(nn.Conv2d(in_channels=N*32, out_channels=N*out_channels, groups=N, kernel_size=(3, 3), stride=2, padding=1)),
        )

    def forward(self, partial_obs):
        T, N, H, W, C = partial_obs.shape

        x = partial_obs.float() #/ 255.0
        x = x.permute(0, 1, 4, 2, 3) # T, N, C, H, W
        return self.conv(x.reshape(T, N*C, H, W)).reshape(T, N, self.out_channels).contiguous()


class MinigridStateEmbeddingNet(nn.Module):
    def __init__(self, observation_shape, final_activation=True):
        super(MinigridStateEmbeddingNet, self).__init__()
        self.observation_shape = observation_shape

        self.feat_extract = make_feat_extract(self.observation_shape[2], 128, final_activation=final_activation)

    def forward(self, inputs):

        # -- [unroll_length x batch_size x height x width x channels]
        x = inputs
        T, B, *_ = x.shape

        # -- [unroll_length*batch_size x height x width x channels]
        x = torch.flatten(x, 0, 1)  # Merge time and batch.

        x = x.float() #/ 255.0

        # -- [unroll_length*batch_size x channels x width x height]
        x = x.transpose(1, 3)
        x = self.feat_extract(x)

        state_embedding = x.view(T, B, -1)

        return state_embedding


class MinigridStateSequenceNet(nn.Module):
    def __init__(
        self,
        observation_shape,
        history=16, # how many frames to use as context. if <= 0, use all frames
        autoregressive=None,
        hidden_size=128,
        supervise_everything=False,
        supervise_early=False,
    ):
        super().__init__()
        self.autoregressive = autoregressive
        self.history = history
        self.supervise_everything = supervise_everything
        self.supervise_early = supervise_early

        self.embed = MinigridStateEmbeddingNet(observation_shape, final_activation=True)
        self.hidden_size = hidden_size

        if self.autoregressive is None or self.autoregressive == 'no':
            self.readin = nn.Linear(128, self.hidden_size, bias=True)
        elif self.autoregressive in ['forward-target', 'forward-target-difference']:
            self.readin = nn.Linear(128 + 128, self.hidden_size, bias=True)
        else:
            raise ValueError(f'Unknown autoregressive mode: {self.autoregressive}')

        self.core = nn.LSTMCell(self.hidden_size, self.hidden_size, 1)
        self.readout = nn.Linear(self.hidden_size, 128, bias=True)
        # self.readout = nn.Sequential(
        #     nn.Linear(self.hidden_size, self.hidden_size, bias=True),
        #     nn.ELU(),
        #     nn.Linear(self.hidden_size, 128, bias=True),
        # )

    def initial_state(self, batch_size):
        device = next(self.parameters()).device
        # LSTMCell
        return tuple(torch.zeros(batch_size, self.core.hidden_size, device=device)
                        for _ in range(2))

    def pad_unfold(
        self,
        x # [unroll_length x batch_size x d]
    ): # -> [history x unroll_length*batch_size x d]

        # pad unroll_length on the left with self.history-1 zeros
        T, B, C = x.shape
        history = max(1, self.history)

        x = F.pad(x, (0, 0, 0, 0, history-1, 0))

        x_windows = x.unfold(0, history, 1)
        # -- [unroll_length x batch_size x d x history]

        x_contexts = x_windows.permute(3, 0, 1, 2).contiguous().view(history, T*B, C)
        # -- [history x unroll_length x batch_size x d]
        # -- [history x unroll_length*batch_size x d]

        return x_contexts

    def run_sequence(
        self,
        x, # [history x unroll_length*batch_size x d]
        done, # [history x unroll_length*batch_size x 1]
    ):
        forward_target_difference = self.autoregressive == 'forward-target-difference'
        forward_target = self.autoregressive == 'forward-target'
        if forward_target_difference or forward_target:
            assert self.history == -1 or x.shape[0] == max(1, self.history)

        zero_hidden, zero_cell = self.initial_state(x.shape[1])

        outputs = []

        if forward_target_difference:
            input = self.readin(x[0, ...])
            hidden, cell = self.core(input, (zero_hidden, zero_cell))
            outputs = [self.readout(hidden)]
        else:
            input = x[0, ...]
            hidden, cell = self.core(input, (zero_hidden, zero_cell))
            outputs = [hidden]

        for i in range(1, x.shape[0]):
            hidden = torch.where(done[i], zero_hidden, hidden)
            cell = torch.where(done[i], zero_cell, cell)

            if forward_target_difference:
                # pad output with d//2 zeros where state should be (at the "top")
                # even if the episode is reset we're still adding error from the past observation
                ar_output = F.pad(outputs[-1], (x.shape[-1]//2, 0))

                # set the next input to be the difference
                # between the current input and the previous output
                input = x[i, ...] - ar_output
                input = self.readin(input)
                hidden, cell = self.core(input, (hidden, cell))
                outputs.append(self.readout(hidden))
            else:
                input = x[i, ...]
                hidden, cell = self.core(input, (hidden, cell))
                outputs.append(hidden)

        outputs = torch.stack(outputs)
        if not forward_target_difference:
            outputs = self.readout(outputs)
        return outputs


    def forward(self, inputs, *, done, targets=None): # targets must be present when auto-regressive
        rnd_loss = 0.
        if targets is not None:
            x = self.embed(inputs)

            if self.supervise_early:
                rnd_loss = (torch.norm(x - targets, dim=2, p=2)).mean(dim=1).sum()

            # shift random embedding by 1 step to the right
            shifted_targets = F.pad(targets, (0, 0, 0, 0, 1, 0))[:-1]

            # reset targets to zero when done: it's a respawn point
            shifted_targets[done] = 0

            if self.history == 0:
                # no history
                shifted_targets = torch.zeros_like(shifted_targets)

            x = torch.cat([x, shifted_targets], dim=-1)

            if self.autoregressive != 'forward-target-difference':
                # this can be done once up front
                x = self.readin(x)

            T, B, D = targets.shape
        else:
            x = self.embed(inputs)
            T, B, D = x.shape
            x = self.readin(x)
        # -- [unroll_length x batch_size x hidden_size]

        done = done.unsqueeze(-1) # [unroll_length x batch_size x 1]

        if self.history >= 0:
            # pad unroll_length on the left with self.history-1 zeros
            x = self.pad_unfold(x)
            done = self.pad_unfold(done)
            if targets is not None:
                targets = self.pad_unfold(targets)

        x = self.run_sequence(x, done)

        if targets is not None:
            if self.history >= 0:
                x = x.view(max(1, self.history), T, B, D)
                targets = targets.view(max(1, self.history), T, B, D)

                # norm over hidden, mean over batch, sum over time (0)
                rnd_loss1 = (torch.norm(x - targets, dim=-1, p=2)).mean(dim=-1).sum(dim=1)

                if self.supervise_everything:
                    # supervise all intermediate targets
                    rnd_loss += rnd_loss1.mean()
                else:
                    rnd_loss += rnd_loss1[-1]

                x = x[-1, ...]
            else:
                # norm over hidden, mean over batch (1), sum over time (0)
                rnd_loss += (torch.norm(x - targets, dim=2, p=2)).mean(dim=-1).sum()

            return x, rnd_loss
        else:
            if self.history >= 0:
                x = x.view(max(1, self.history), T, B, D)[-1, ...]
            return x


class MinigridTrajectoryEmbeddingNet(nn.Module):
    def __init__(self, observation_shape):
        super(MinigridTrajectoryEmbeddingNet, self).__init__()
        self.observation_shape = observation_shape

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        self.feat_extract = nn.Sequential(
            init_(nn.Conv2d(in_channels=self.observation_shape[2]*4, out_channels=32, kernel_size=(3, 3), stride=2,
                            padding=1)),
            nn.ELU(),
            init_(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2, padding=1)),
            nn.ELU(),
            init_(nn.Conv2d(in_channels=32, out_channels=128, kernel_size=(3, 3), stride=2, padding=1)),
            nn.ELU(),
        )

    def forward(self, inputs):
        # -- [unroll_length x batch_size x height x width x channels]
        # -- [unroll_length x batch_size x height x width x channels*trajectory_length]
        x = inputs
        T, B, *_ = x.shape

        # -- [unroll_length*batch_size x height x width x channels]
        x = torch.flatten(x, 0, 1)  # Merge time and batch.

        x = x.float() / 255.0

        # -- [unroll_length*batch_size x channels x width x height]
        x = x.transpose(1, 3)
        x = self.feat_extract(x)

        state_embedding = x.view(T, B, -1)
        return state_embedding

class MinigridInverseDynamicsNet(nn.Module):
    def __init__(self, num_actions):
        super(MinigridInverseDynamicsNet, self).__init__()
        self.num_actions = num_actions 
        
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                            constant_(x, 0), nn.init.calculate_gain('relu'))
        self.inverse_dynamics = nn.Sequential(
            init_(nn.Linear(2 * 128, 256)), 
            nn.ReLU(),  
        )

        init_ = lambda m: init(m, nn.init.orthogonal_, 
            lambda x: nn.init.constant_(x, 0))
        self.id_out = init_(nn.Linear(256, self.num_actions))

        
    def forward(self, state_embedding, next_state_embedding):
        inputs = torch.cat((state_embedding, next_state_embedding), dim=2)
        action_logits = self.id_out(self.inverse_dynamics(inputs))
        return action_logits
    

class MinigridForwardDynamicsNet(nn.Module):
    def __init__(self, num_actions):
        super(MinigridForwardDynamicsNet, self).__init__()
        self.num_actions = num_actions 

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                            constant_(x, 0), nn.init.calculate_gain('relu'))
    
        self.forward_dynamics = nn.Sequential(
            init_(nn.Linear(128 + self.num_actions, 256)), 
            nn.ReLU(), 
        )

        init_ = lambda m: init(m, nn.init.orthogonal_, 
            lambda x: nn.init.constant_(x, 0))

        self.fd_out = init_(nn.Linear(256, 128))

    def forward(self, state_embedding, action):
        action_one_hot = F.one_hot(action, num_classes=self.num_actions).float()
        inputs = torch.cat((state_embedding, action_one_hot), dim=2)
        next_state_emb = self.fd_out(self.forward_dynamics(inputs))
        return next_state_emb


class MarioDoomPolicyNet(nn.Module):
    def __init__(self, observation_shape, num_actions):
        super(MarioDoomPolicyNet, self).__init__()
        self.observation_shape = observation_shape
        self.num_actions = num_actions 

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                                constant_(x, 0), nn.init.calculate_gain('relu'))

        self.feat_extract = nn.Sequential(
            init_(nn.Conv2d(in_channels=self.observation_shape[0], out_channels=32, kernel_size=(3, 3), stride=2, padding=1)),
            nn.ELU(),
            init_(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2, padding=1)),
            nn.ELU(),
            init_(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2, padding=1)),
            nn.ELU(),
            init_(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2, padding=1)),
            nn.ELU(),
        )

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                                constant_(x, 0))

        self.core = nn.LSTM(288, 256, 2)

        self.policy = init_(nn.Linear(256, self.num_actions))
        self.baseline = init_(nn.Linear(256, 1))


    def initial_state(self, batch_size):
        return tuple(torch.zeros(self.core.num_layers, batch_size, 
                                self.core.hidden_size) for _ in range(2))

    def forward(self, inputs, core_state=()):
        # -- [unroll_length x batch_size x height x width x channels]
        x = inputs['frame']
        T, B, C, W, H = x.shape
        x = x.reshape(T, B, W, H, C)

        # -- [unroll_length*batch_size x height x width x channels]
        x = torch.flatten(x, 0, 1)  # Merge time and batch.
        x = x.float() #/ 255.0
        
        # -- [unroll_length*batch_size x channels x width x height]
        x = x.transpose(1, 3)
        x = self.feat_extract(x)

        core_input = x.view(T * B, -1)
 
        core_input = core_input.view(T, B, -1)
        core_output_list = []
        notdone = (~inputs['done'].type(torch.ByteTensor)).float()
        if core_input.is_cuda:
            notdone = notdone.cuda()
        for input, nd in zip(core_input.unbind(), notdone.unbind()):
            nd = nd.view(1, -1, 1)
            core_state = tuple(nd * s for s in core_state)
            output, core_state = self.core(input.unsqueeze(0), core_state)
            core_output_list.append(output)
        core_output = torch.flatten(torch.cat(core_output_list), 0, 1)
        
        policy_logits = self.policy(core_output)
        baseline = self.baseline(core_output)

        if self.training:
            action = torch.multinomial(
                F.softmax(policy_logits, dim=1), num_samples=1)
        else:
            action = torch.argmax(policy_logits, dim=1)

        policy_logits = policy_logits.view(T, B, self.num_actions)
        baseline = baseline.view(T, B)
        action = action.view(T, B)

        return dict(policy_logits=policy_logits, baseline=baseline, 
                    action=action), core_state


class MarioDoomStateEmbeddingNet(nn.Module):
    def __init__(self, observation_shape):
        super(MarioDoomStateEmbeddingNet, self).__init__()
        self.observation_shape = observation_shape

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                            constant_(x, 0), nn.init.calculate_gain('relu'))

        self.feat_extract = nn.Sequential(
            init_(nn.Conv2d(in_channels=self.observation_shape[0], out_channels=32, kernel_size=(3, 3), stride=2, padding=1)),
            nn.ELU(),
            init_(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2, padding=1)),
            nn.ELU(),
            init_(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2, padding=1)),
            nn.ELU(),
            init_(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2, padding=1)),
            nn.ELU(),
        )
    
    def forward(self, inputs):
        # -- [unroll_length x batch_size x height x width x channels]
        x = inputs
        T, B, C, W, H = x.shape
        x = x.reshape(T, B, W, H, C)

        # -- [unroll_length*batch_size x height x width x channels]
        x = torch.flatten(x, 0, 1)  # Merge time and batch.
        x = x.float() / 255.0

        # -- [unroll_length*batch_size x channels x width x height]
        x = x.transpose(1, 3)
        x = self.feat_extract(x)

        state_embedding = x.view(T, B, -1)
        
        return state_embedding


class MarioDoomForwardDynamicsNet(nn.Module):
    def __init__(self, num_actions):
        super(MarioDoomForwardDynamicsNet, self).__init__()
        self.num_actions = num_actions 
            
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                            constant_(x, 0), nn.init.calculate_gain('relu'))
    
        self.forward_dynamics = nn.Sequential(
            init_(nn.Linear(288 + self.num_actions, 256)), 
            nn.ReLU(), 
        )

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                            constant_(x, 0))

        self.fd_out = init_(nn.Linear(256, 288))

    def forward(self, state_embedding, action):
        action_one_hot = F.one_hot(action, num_classes=self.num_actions).float()
        inputs = torch.cat((state_embedding, action_one_hot), dim=2)
        next_state_emb = self.fd_out(self.forward_dynamics(inputs))
        return next_state_emb


class MarioDoomInverseDynamicsNet(nn.Module):
    def __init__(self, num_actions):
        super(MarioDoomInverseDynamicsNet, self).__init__()
        self.num_actions = num_actions 

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                            constant_(x, 0), nn.init.calculate_gain('relu'))
        self.inverse_dynamics = nn.Sequential(
            init_(nn.Linear(2 * 288, 256)), 
            nn.ReLU(), 
        )

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                            constant_(x, 0))

        self.id_out = init_(nn.Linear(256, self.num_actions))

        
    def forward(self, state_embedding, next_state_embedding):
        inputs = torch.cat((state_embedding, next_state_embedding), dim=2)
        action_logits = self.id_out(self.inverse_dynamics(inputs))
        return action_logits
    

