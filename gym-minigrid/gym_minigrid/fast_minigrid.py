from gym_minigrid.envs.multiroom import MultiRoomEnv
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def matshow(data, figsize=(15,5), axs=None):
    if axs is None:
        fig, axs = plt.subplots(1, 3, figsize=figsize)

    for chan, ax in enumerate(axs):
        ax.matshow(data[:,:,chan], cmap='tab20b')
        if chan == 0:
            ax.set_title('OBJECT_IDX')
        elif chan == 1:
            ax.set_title('COLOR_IDX')
        elif chan == 2:
            ax.set_title('STATE')

        # Add text labels inside each cell
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(j, i, str(data[i, j, chan].item()), va='center', ha='center', color='black')
                
# # Facing right
# if self.agent_dir == 0:
# # Facing down
# elif self.agent_dir == 1:
# # Facing left
# elif self.agent_dir == 2:
# # Facing up
# elif self.agent_dir == 3:

def get_view_top(agent_pos, agent_view_size):
    """
    Get the extents of the square set of tiles visible to the agent
    Note: the bottom extent indices are not included in the set
    """

    topXYOffset = torch.tensor([
        # Facing right
        [0, - (agent_view_size // 2)],
        # Facing down
        [- (agent_view_size // 2), 0],
        # Facing left
        [- agent_view_size + 1, - (agent_view_size // 2)],
        # Facing up
        [- (agent_view_size // 2), - agent_view_size + 1],        
    ])

    return topXYOffset + agent_pos

def see_thru(obj, state):
    if state == 1: # closed
        return False
    if obj == 2: # wall
        return False
    return True


def og_crop(self):
    topX, topY, botX, botY = self.get_view_exts()
    print('slice', topX, topY, self.agent_view_size, self.agent_view_size)
    agent_grid = self.grid.slice(topX, topY, self.agent_view_size, self.agent_view_size)
    return torch.tensor(agent_grid.encode())


def og_crop_rot(self):
    topX, topY, botX, botY = self.get_view_exts()
    agent_grid = self.grid.slice(topX, topY, self.agent_view_size, self.agent_view_size)

    for i in range(self.agent_dir + 1):
        agent_grid = agent_grid.rotate_left()

    return torch.tensor(agent_grid.encode())

def crop(self, pad=5):    
    # look for gen_obs_grid
    complete = torch.tensor(self.grid.encode())

    if pad:
        pad_value = 2
        complete = torch.nn.functional.pad(complete, (0,0,pad,pad,pad,pad), mode='constant', value=pad_value)
        
    exts = get_view_top(self.agent_pos, self.agent_view_size)
    print('view top', exts, 'exts', self.get_view_exts()[:2], 'dir', self.agent_dir, 'pos', self.agent_pos)
    topX, topY = exts[self.agent_dir] + pad

    print('sliceP', topX, topY, self.agent_view_size, self.agent_view_size)
    return complete[topX:topX+self.agent_view_size, topY:topY+self.agent_view_size, :]


def crop_rot(self):
    code = crop(self)
    code = code.rot90(self.agent_dir+1, dims=(1,0))
    return code    


def get_agent_grid_encoding(self, mask=True):
    code = crop_rot(self)

    agent_pos = torch.tensor([(self.agent_view_size // 2 , self.agent_view_size - 1)])

    print('pos and shape', agent_pos, code.shape)
    assert code.numel()

    if mask:
        return mask_agent_grid_encoding(code, agent_pos, self.agent_view_size)
    else:
        return code

def make(seed):
    env = MultiRoomEnv(7,7,4,coloredWalls=False)
    env.seed(seed)
    env.reset()
    return env

def mask_agent_grid_encoding(code, agent_pos, agent_view_size):
    neighbors = torch.tensor([(0,1), (1,0), (0,-1), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)])

    mask = torch.zeros_like(code[:,:,0], dtype=int)
    # 0 unk
    # 1 false
    # 2 true
    mask[agent_pos[0,0], agent_pos[0,1]] = 2

    visited = torch.zeros_like(code[:,:,0], dtype=torch.bool)
    visited[agent_pos[0,0], agent_pos[0,1]] = True

    targets = list(agent_pos + neighbors)
    sources = list([agent_pos[0]]*len(neighbors))

    # print(mask)
    # print('sources', sources)

    fuel = 100

    while targets:
        nei, src = targets.pop(0), sources.pop(0)
        if nei[0] < 0 or nei[1] < 0 or nei[0] >= agent_view_size or nei[1] >= agent_view_size:
            continue

        can_see_thru = see_thru(code[nei[0], nei[1], 0], code[nei[0], nei[1], 2])
        if mask[src[0], src[1]] == 2:
            if can_see_thru:
                mask[nei[0], nei[1]] = 2
            else:
                mask[nei[0], nei[1]] = 1
        elif mask[src[0], src[1]] == 1:
            mask[nei[0], nei[1]] = max(0, mask[nei[0], nei[1]])
        elif mask[src[0], src[1]] == 0:
            #print('wow', nei, src)
            pass

        visited[nei[0], nei[1]] = True
        for x in neighbors:
            source = nei
            target = nei+x
            if target[0] < 0 or target[1] < 0 or target[0] >= agent_view_size or target[1] >= agent_view_size:
                continue
            if not visited[nei[0]+x[0], nei[1]+x[1]]:
                targets.append(target)
                sources.append(source)

        fuel -= 1
        if not fuel:
            break

    return mask[:,:,None].to(bool)*code

def batch_mask(
    code, # N, 7, 7, 3
    agent_view_size=7
):
    agent_pos = torch.tensor([(agent_view_size // 2 , agent_view_size - 1)])

    neighbors = torch.tensor([(0,1), (1,0), (0,-1), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)])

    mask = torch.zeros_like(code[...,0], dtype=int)
    # 0 unk
    # 1 false
    # 2 true
    mask[:, agent_pos[0,0,0], agent_pos[0,0,1]] = 2

    visited = torch.zeros_like(code[...,0], dtype=torch.bool)
    visited[:, agent_pos[0,0], agent_pos[0,1]] = True

    targets = list(agent_pos + neighbors)
    sources = list([agent_pos[0]]*len(neighbors))

    # print(mask)
    # print('sources', sources)

    fuel = 100

    while targets:
        nei, src = targets.pop(0), sources.pop(0)
        if nei[0] < 0 or nei[1] < 0 or nei[0] >= agent_view_size or nei[1] >= agent_view_size:
            continue

        can_see_thru = see_thru(code[nei[0], nei[1], 0], code[nei[0], nei[1], 2])
        if mask[src[0], src[1]] == 2:
            if can_see_thru:
                mask[nei[0], nei[1]] = 2
            else:
                mask[nei[0], nei[1]] = 1
        elif mask[src[0], src[1]] == 1:
            mask[nei[0], nei[1]] = max(0, mask[nei[0], nei[1]])
        elif mask[src[0], src[1]] == 0:
            #print('wow', nei, src)
            pass

        visited[nei[0], nei[1]] = True
        for x in neighbors:
            source = nei
            target = nei+x
            if target[0] < 0 or target[1] < 0 or target[0] >= agent_view_size or target[1] >= agent_view_size:
                continue
            if not visited[nei[0]+x[0], nei[1]+x[1]]:
                targets.append(target)
                sources.append(source)

        fuel -= 1
        if not fuel:
            break

    return mask[:,:,None].to(bool)*code


def og_get_agent_grid_encoding(env, mask=True):
    self = env
    topX, topY, botX, botY = self.get_view_exts()

    agent_grid = self.grid.slice(topX, topY, self.agent_view_size, self.agent_view_size)

    for i in range(self.agent_dir + 1):
        agent_grid = agent_grid.rotate_left()

    if mask:
        # prevent seeing through walls
        vis_mask = agent_grid.process_vis(agent_pos=(self.agent_view_size // 2 , self.agent_view_size - 1))
        agent_frame = agent_grid.encode(vis_mask)
    else:
        agent_frame = agent_grid.encode()
    return torch.from_numpy(agent_frame)


class Mask(nn.Module):
    def __init__(self):
        super().__init__()
        # convolutional kernel to connect with neighbors on the grid
        self.step = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.step.weight.data = 0.3 * torch.tensor([[1, 1, 1],
                                                    [1, 2, 1],
                                                    [1, 1, 1]]).view(1,1,3,3)
        self.steps = 4

    def forward(
        self,
        grid, # float zero grid with 1 where agent is
        closed # bool zero grid with 1 where obstacles are
    ):
        # propagate signal from starting cell
        for _ in range(self.steps):
            grid = self.step(grid)
            print(grid, 'pre', _)
            # activation: squash and restore obstacles
            grid = -0.01 * closed + grid * (1 - closed.float())
            print(grid, _)

        # explicitly discriminate reached from unreached cells
        # and restore obstacles
        grid = torch.where((grid>0)|closed, 1., -1.)
        #print(grid, 'signed and restored')

        # connect nearby obstacles
        grid = self.step(grid)
        #print(grid)

        return grid>0



if __name__ == '__main__':
    for seed in [5,3,7,9,11,13,15]:
        env = MultiRoomEnv(7,7,4,coloredWalls=False)
        env.seed(seed)
        env.reset()
        print(seed)
        if not torch.allclose(fg.og_get_agent_grid_encoding(env), fg.get_agent_grid_encoding(env)):
            print('mismatch', seed)
            matshow(crop(env, pad=4))
            matshow(og_crop(env))
            assert False
