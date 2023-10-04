from gym_minigrid.envs.multiroom import MultiRoomEnv
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def matshow(data, figsize=(15,5), axs=None, title='', agent_pos=None):
    if axs is None:
        fig, axs = plt.subplots(1, 3, figsize=figsize)
        if title:
            fig.suptitle(title)

    for chan, ax in enumerate(axs):
        ax.matshow(data[:,:,chan], cmap='tab20b')
        if chan == 0:
            ax.set_title('OBJECT_IDX')
            if agent_pos is not None:
                ax.scatter(agent_pos[1], agent_pos[0], marker='*', s=200, c='black')
        elif chan == 1:
            ax.set_title('COLOR_IDX')
        elif chan == 2:
            ax.set_title('STATE')

        # Add text labels inside each cell
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(j, i, str(data[i, j, chan].item()), va='center', ha='center', color='black')
    return axs


class TimeCrop(nn.Module):
    "Crop a chunk of size T given starting points"
    def __init__(self, side=100):
        super().__init__()
        self.side = side
        self.steps = nn.Parameter(torch.arange(side), requires_grad=False)

    def forward(
        self,
        grid, # T+k, N, C
        top # N,
    ): # -> (N, side, side, C)
        N = top.shape[0]
        indices = self.steps + top[:, None]
        indices = torch.cat([
            torch.arange(N, device=grid.device).repeat_interleave(self.side)[:, None],
            indices.view(-1, 1)
        ], dim=-1)
        return grid[indices[:,0], indices[:,1], :].view(N, self.side, -1).permute(1, 0, 2)


class BatchCrop2d(nn.Module):
    "Crop a grid of size HxW into a grid of size side x side given the top left corner of the crop"
    def __init__(self, side=7):
        super().__init__()
        self.side = side
        self.square = nn.Parameter(torch.cartesian_prod(torch.arange(side), torch.arange(side)), requires_grad=False)

    def forward(
        self,
        grid, # N, H, W, C
        topXY # N, 2
    ): # -> (N, side, side, C)
        N, _ = topXY.shape
        indices = self.square + topXY[:, None, :]
        indices = torch.cat([
            torch.arange(N, device=grid.device).repeat_interleave(self.side*self.side)[:, None],
            indices.view(-1, 2)
        ], dim=-1)
        return grid[indices[:,0], indices[:,1], indices[:,2], :].view(N, self.side, self.side, -1)


def inverse_permutation(perm):
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(perm.size(0), device=perm.device)
    return inv


def rotate_grouped(grid, directions):
    # split grid into 4 groups for each direction:
    # direction 0 means rot90 with k=1
    # direction 1 means rot90 with k=2
    # direction 2 means rot90 with k=3
    # direction 3 means rot90 with k=4 (identity)

    directions_0 = torch.where(directions==0)[0]
    directions_1 = torch.where(directions==1)[0]
    directions_2 = torch.where(directions==2)[0]
    directions_3 = torch.where(directions==3)[0]

    rot1 = grid[directions_0].rot90(1, dims=(2,1))
    rot2 = grid[directions_1].rot90(2, dims=(2,1))
    rot3 = grid[directions_2].rot90(3, dims=(2,1))
    rot4 = grid[directions_3]

    # reassemble groups
    grid = torch.cat([rot1, rot2, rot3, rot4], dim=0)
    perm = torch.cat([directions_0, directions_1, directions_2, directions_3], dim=0)
    # permute back to original order
    return grid[inverse_permutation(perm)]


class BatchMinigrid(nn.Module):
    def __init__(self, seeds=[3,5,7]):
        super().__init__()

        def make(seed):
            env = MultiRoomEnv(7,7,4,coloredWalls=False)
            env.seed(seed)
            env.reset()
            return env

        self.envs = [make(seed) for seed in seeds]
        self.grids = nn.Parameter(torch.tensor(np.stack([env.grid.encode() for env in self.envs])), requires_grad=False) # N, H, W, C
        self.agent_pos = nn.Parameter(torch.tensor(np.array([env.agent_pos for env in self.envs])), requires_grad=False) # N, 2
        self.agent_dir = nn.Parameter(torch.tensor([env.agent_dir for env in self.envs]), requires_grad=False) # N,
        self.step_counts = nn.Parameter(torch.zeros(len(self.envs), dtype=torch.int32), requires_grad=False) # N,
        self.max_steps = 140
        self.agent_view_size = 7

        self.agent_pos_fpv = (self.agent_view_size // 2 , self.agent_view_size - 1)

        self.masker = Mask()
        self.crop = BatchCrop2d(side=self.agent_view_size)

    def merge(self, other):
        # merge two batches of environments
        assert self.agent_view_size == other.agent_view_size
        assert self.max_steps == other.max_steps
        self.envs.extend(other.envs)
        self.grids.data = torch.cat([self.grids, other.grids], dim=0)
        self.agent_pos.data = torch.cat([self.agent_pos, other.agent_pos], dim=0)
        self.agent_dir.data = torch.cat([self.agent_dir, other.agent_dir], dim=0)
        self.step_counts.data = torch.cat([self.step_counts, other.step_counts], dim=0)
        return self

    def reset(self, done):
        # do env.reset() and read new grids for all environments that are done
        # TODO: maintain a list of fresh environments in another thread

        if done.sum() == 0:
            return

        for i, env in enumerate(self.envs):
            if done[i]:
                env.reset()
                self.grids[i] = torch.from_numpy(env.grid.encode())
                self.agent_pos[i] = torch.tensor(env.agent_pos)
                self.agent_dir[i] = env.agent_dir
                self.step_counts[i] = 0

    @staticmethod
    def render_fpv_slow1(env, mask=True):
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

    def render_fpv_slow(self):
        "Render all environments using original code, for reference testing"
        return torch.stack([self.render_fpv_slow1(env) for env in self.envs])

    def render_fpv(self, pad=5):
        "Render all environments in batch"
        N, H, W, C = self.grids.shape
        agent_view_size = self.agent_view_size

        grid = self.grids
        if pad:
            pad_value = 2
            grid = torch.nn.functional.pad(grid, (0,0,pad,pad,pad,pad), mode='constant', value=pad_value)

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
        ], device=grid.device)

        top = (self.agent_pos + topXYOffset[:, None, :])[self.agent_dir, torch.arange(N), :] + pad

        grid = self.crop(grid, top)
        grid = rotate_grouped(grid, self.agent_dir)

        walls = grid[:,:,:,0]==2 # object wall
        closed = grid[:,:,:,2]==1 # state closed
        me = torch.zeros(N, 1, agent_view_size, agent_view_size, device=grid.device)
        me[:, :, self.agent_pos_fpv[0], self.agent_pos_fpv[1]] = 1

        #print(walls|closed, 'obstacles')
        mask = self.masker(me, closed=(walls|closed).unsqueeze(1))

        return mask.permute(0, 2, 3, 1)*grid

    def _rotate(
        self,
        actions # N, +1 means rotate right, -1 means rotate left
    ):
        return (self.agent_dir + actions) % 4

    def _toggle_(
        self,
        actions # (N,), 1 means toggle, 0 means don't toggle
    ):
        # for each grid if the agent is in front of a closed door, open it
        each = torch.arange(len(actions), device=actions.device)

        directions = self.agent_dir

        front_pos = (torch.tensor([
            [1, 0],
            [0, 1],
            [-0, 0],
            [0, -1],
        ], device=actions.device)[None, :, :] + self.agent_pos[:, None, :])[each, directions] # N, 2

        # am i facing a closed door?
        next_door = self.grids[each, front_pos[:, 0], front_pos[:, 1], 2] == 1

        # if didn't ask to toggle, don't toggle
        next_door &= actions.bool()

        # change the door state to 0 (open)
        self.grids[each, front_pos[:, 0], front_pos[:, 1], 2] += next_door * -1

    def _move_forward(
        self,
        actions, # (N,), 1 means move forward, 0 means don't move
    ):
        each = torch.arange(len(actions), device=actions.device)
        assert len(self.agent_pos) == len(actions)

        directions = self.agent_dir

        # # Map of agent direction indices to vectors
        # DIR_TO_VEC = [
        #     # Pointing right (positive X)
        #     np.array((1, 0)),
        #     # Down (positive Y)
        #     np.array((0, 1)),
        #     # Pointing left (negative X)
        #     np.array((-1, 0)),
        #     # Up (negative Y)
        #     np.array((0, -1)),
        # ]
        front_pos = (torch.tensor([
            [1, 0],
            [0, 1],
            [-0, 0],
            [0, -1],
        ], device=actions.device)[None, :, :] + self.agent_pos[:, None, :])[each, directions] # N, 2

        #print('front_pos', front_pos)

        # if facing wall or closed door, don't move
        # if facing goal, don't move, end episode
        next_wall = self.grids[each, front_pos[:, 0], front_pos[:, 1], 0] == 2
        next_door = self.grids[each, front_pos[:, 0], front_pos[:, 1], 2] == 1
        next_goal = self.grids[each, front_pos[:, 0], front_pos[:, 1], 0] == 8 # N,2

        can_move = ~(next_wall | next_door | next_goal) # N,

        #print('can_move', can_move, 'dir', directions.data)

        next_agent_pos = torch.where(can_move[:, None]*actions[:, None].bool(), front_pos, self.agent_pos)
        done = next_goal*actions.bool()

        return next_agent_pos, done


    def step(
        self,
        actions, # N,
        test_slow=False
    ):
        rotate_left = actions == 0
        rotate_right = actions == 1
        move = actions == 2
        toggle = actions == 5

        next_agent_dir = self._rotate(rotate_right.int() - rotate_left.int())
        self.agent_dir.data = next_agent_dir
        self._toggle_(toggle)
        next_agent_pos, has_goal = self._move_forward(move)
        self.agent_pos.data = next_agent_pos

        self.step_counts += 1
        done = has_goal | (self.step_counts >= 140) # 140 is the timeout, Nrooms*20
        self.reset(done)

        obs = self.render_fpv()
        reward = has_goal.float() * (1 - 0.9 * (self.step_counts / self.max_steps))
        info = {'fast': True}

        if test_slow:
            slow_obs, slow_reward, slow_done, slow_info = self.step_slow(actions)
            if not torch.allclose(obs, slow_obs):
                print('obs mismatch! maybe due to incorrect occlusion implementation in Mask')
            assert torch.allclose(reward.float(), slow_reward.float())
            assert torch.allclose(done, slow_done)
            print('slow_info', slow_info)

        return obs, reward, done, info


    def step_slow(self, actions):
        obs, reward, done, info = [], [], [], []
        for i, env in enumerate(self.envs):
            obs1, reward1, done1, info1 = env.step(actions[i])
            obs.append(obs1)
            reward.append(reward1)
            done.append(done1)
            info.append(info1)
        return torch.stack([torch.from_numpy(obs1['image']) for obs1 in obs]).to(device), torch.tensor(reward).to(device).float(), torch.tensor(done).to(device), info


# XXX: this needs to mask out some occluded corners in subsequent rules to be fully compatible
class Mask(nn.Module):
    def __init__(self):
        super().__init__()
        # convolutional kernel to connect with neighbors on the grid
        self.step = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.step.weight.data = 0.3 * torch.tensor([[1, 1, 1],
                                                    [1, 2, 1],
                                                    [1, 1, 1]]).view(1,1,3,3)
        self.steps = 5

        # self.connect = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        # self.connect.weight.data = 0.3 * torch.tensor([[0, 1, 0],
        #                                                [1, 2, 1],
        #                                                [0, 1, 0]]).view(1,1,3,3)

    def forward(
        self,
        grid, # float zero grid with 1 where agent is, N 1 H H
        closed # bool zero grid with 1 where obstacles are, N 1 H H
    ):
        print = lambda *args: None

        # propagate signal from starting cell
        for _ in range(self.steps):
            grid = self.step(grid)
            print(grid, 'pre', _)
            # activation: squash and restore obstacles
            grid = -0.01 * closed + grid.tanh() * (1 - closed.float())
            print(grid, _)

        # take only reached cells
        grid = (grid>0).float()
        print(grid, 'signed and restored')

        # connect nearby obstacles
        grid = self.step(grid.float())
        print(grid)

        return grid>0



if __name__ == '__main__':
    def testeach(xs, ys):
        return [torch.allclose(x, y) for x, y in zip(xs, ys)]

    b = BatchMinigrid(seeds=[3,5,16,7,10,9,14,15,16,17,18,19,20,21,22])

    for x in testeach(b.render_fpv(), b.render_fpv_slow()):
        print(x)
