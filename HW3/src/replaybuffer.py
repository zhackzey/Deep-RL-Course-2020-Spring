# author: Zhu Zeyu
# stuID: 1901111360
'''
    This script implements replay buffer.
'''
import numpy as np
import threading


class ReplayBuffer:
    def __init__(self, args):
        self.args = args
        self.n_actions = self.args.n_actions
        self.n_agents = self.args.n_agents
        self.state_shape = self.args.state_shape
        self.obs_shape = self.args.obs_shape
        self.size = self.args.buffer_size
        self.max_episode_length_limit = self.args.max_episode_length_limit
        # memory management
        self.current_idx = 0
        self.current_size = 0
        # create the buffer to store info
        self.buffers = {'o': np.empty([self.size, self.max_episode_length_limit, self.n_agents, self.obs_shape]),
                        'u': np.empty([self.size, self.max_episode_length_limit, self.n_agents, 1]),
                        's': np.empty([self.size, self.max_episode_length_limit, self.state_shape]),  # state是对于全局而言的，不分agent
                        'r': np.empty([self.size, self.max_episode_length_limit, 1]),
                        'o_next': np.empty([self.size, self.max_episode_length_limit, self.n_agents, self.obs_shape]),
                        's_next': np.empty([self.size, self.max_episode_length_limit, self.state_shape]),
                        'u_onehot': np.empty([self.size, self.max_episode_length_limit, self.n_agents, self.n_actions]),
                        'padded': np.empty([self.size, self.max_episode_length_limit, 1]),
                        'terminated': np.empty([self.size, self.max_episode_length_limit, 1])
                        }
        # thread lock
        self.lock = threading.Lock()

        # store the episode
    def store_episode(self, episode_batch):
        batch_size = episode_batch['o'].shape[0] 
        with self.lock:
            idxs = self._get_storage_idx(inc=batch_size)
            # store the informations
            #print("idxs:",idxs)
            #print("episode batch['o'] shape", episode_batch['o'].shape)
            self.buffers['o'][idxs] = episode_batch['o']
            self.buffers['u'][idxs] = episode_batch['u']
            self.buffers['s'][idxs] = episode_batch['s']
            self.buffers['r'][idxs] = episode_batch['r']
            self.buffers['o_next'][idxs] = episode_batch['o_next']
            self.buffers['s_next'][idxs] = episode_batch['s_next']
            self.buffers['u_onehot'][idxs] = episode_batch['u_onehot']
            self.buffers['padded'][idxs] = episode_batch['padded']
            self.buffers['terminated'][idxs] = episode_batch['terminated']

    def sample(self, batch_size):
        temp_buffer = {}
        idx = np.random.randint(0, self.current_size, batch_size)
        for key in self.buffers.keys():
            temp_buffer[key] = self.buffers[key][idx]
        return temp_buffer

    def _get_storage_idx(self, inc=None):
        inc = inc or 1
        if self.current_idx + inc <= self.size:
            idx = np.arange(self.current_idx, self.current_idx + inc)
            self.current_idx += inc
        elif self.current_idx < self.size:
            overflow = inc - (self.size - self.current_idx)
            idx_a = np.arange(self.current_idx, self.size)
            idx_b = np.arange(0, overflow)
            idx = np.concatenate([idx_a, idx_b])
            self.current_idx = overflow
        else:
            idx = np.arange(0, inc)
            self.current_idx = inc
        self.current_size = min(self.size, self.current_size + inc)
        if inc == 1:
            idx = idx[0]
        return idx
