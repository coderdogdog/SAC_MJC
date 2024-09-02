import numpy as np
import torch


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class RandomBuffer(object):
    def __init__(self, state_dim, action_dim, Env_with_dead=True, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.Env_with_dead = Env_with_dead

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.reward = np.zeros((max_size, 1))
        self.next_state = np.zeros((max_size, state_dim))
        self.dead = np.zeros((max_size, 1), dtype=np.uint8)

        self.device = device

    def add(self, state, action, reward, next_state, dead):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        # it is important to distinguish between dead and done!!!
        # See https://zhuanlan.zhihu.com/p/409553262 for better understanding.
        if self.Env_with_dead:
            self.dead[self.ptr] = dead
        else:
            self.dead[self.ptr] = False

        self.ptr = self.ptr + 1
        self.ptr = self.ptr % self.max_size
        self.size = self.ptr

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        with torch.no_grad():
            return (
                torch.FloatTensor(self.state[ind]).to(self.device),
                torch.FloatTensor(self.action[ind]).to(self.device),
                torch.FloatTensor(self.reward[ind]).to(self.device),
                torch.FloatTensor(self.next_state[ind]).to(self.device),
                torch.FloatTensor(self.dead[ind]).to(self.device)
            )
