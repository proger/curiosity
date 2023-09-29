from gym_minigrid.envs.multiroom import MultiRoomEnv
import matplotlib.pyplot as plt
import torch
import numpy as np

def matshow(data, figsize=(15,5), axs=None):
    if axs is None:
        fig, axs = plt.subplots(1, 3, figsize=figsize)

    for chan, ax in enumerate(axs):
        ax.matshow(data[:,:,chan], cmap='tab20b')
        match chan:
            case 0:
                ax.set_title('OBJECT_IDX')
            case 1:
                ax.set_title('COLOR_IDX')
            case 2:
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

    # mask = torch.zeros_like(code[:,:,0], dtype=torch.bool)
    # mask[agent_pos[0,0], agent_pos[0,1]] = False

    # while stack:
    #     nei = stack.pop()
    #     if nei[0] < 0 or nei[1] < 0 or nei[0] >= self.agent_view_size or nei[1] >= self.agent_view_size:
    #         continue
    #     if see_thru(code[nei[0], nei[1], 0], code[nei[0], nei[1], 2]):
    #         mask[nei[0], nei[1]] = True
    #         visited.add(str(nei))
    #         for x in neighbors:
    #             if str(nei+x) not in visited:
    #                 stack.append(nei + x)
    #     else:
    #         mask[nei[0], nei[1]] = True
    #         visited.add(str(nei))

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

def crop(self, pad=4):    
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


def crop_rot(self, pad=4):
    code = crop(self)
    code = code.rot90(self.agent_dir+1, dims=(1,0))
    return code    


def get_agent_grid_encoding(self, mask=True):
    code = crop_rot(self)

    agent_pos = torch.tensor([(self.agent_view_size // 2 , self.agent_view_size - 1)])

    print(agent_pos, code.shape)

    if mask:
        return mask_agent_grid_encoding(code, agent_pos, self.agent_view_size)
    else:
        return code

def mask_agent_grid_encoding(code, agent_pos, agent_view_size):
    neighbors = torch.tensor([(0,1), (1,0), (0,-1), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)])

    sched = torch.zeros_like(code[:,:,0], dtype=int)
    # 0 unk
    # 1 false
    # 2 true
    sched[agent_pos[0,0], agent_pos[0,1]] = 2

    print(sched)

    stack = list(agent_pos + neighbors)
    srcs = list([agent_pos[0]]*len(neighbors))
    print('srcs', srcs)

    visited = {str(agent_pos)}
    fuel = 100

    while stack:
        nei, src = stack.pop(0), srcs.pop(0)
        if nei[0] < 0 or nei[1] < 0 or nei[0] >= agent_view_size or nei[1] >= agent_view_size:
            continue

        if sched[src[0], src[1]] == 2:
            if see_thru(code[nei[0], nei[1], 0], code[nei[0], nei[1], 2]):
                sched[nei[0], nei[1]] = 2
            else:
                sched[nei[0], nei[1]] = 1
        elif sched[src[0], src[1]] == 1:
            if see_thru(code[nei[0], nei[1], 0], code[nei[0], nei[1], 2]):
                sched[nei[0], nei[1]] = max(0, sched[nei[0], nei[1]])
            else:
                sched[nei[0], nei[1]] = max(0, sched[nei[0], nei[1]])
        elif sched[src[0], src[1]] == 0:
            #print('wow', nei, src)
            pass

        visited.add(str(nei))
        for x in neighbors:
            if str(nei+x) not in visited:
                stack.append(nei+x)
                srcs.append(nei)

        fuel -= 1
        if not fuel:
            break

    return sched[:,:,None].to(bool)*code

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
