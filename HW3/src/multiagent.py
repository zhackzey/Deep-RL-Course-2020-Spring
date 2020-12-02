# author: Zhu Zeyu
# stuID: 1901111360
'''
    This script implements wrapper for multi-agents.
'''
import numpy as np
import torch
import copy
from qmix import QMix
class MultiAgents:
    def __init__(self,args):
        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        self.policy = QMix(args)

    def choose_action(self, obs, last_action, agent_num, epsilon):
        inputs = copy.deepcopy(obs)
        agent_id = np.zeros(self.n_agents)
        agent_id[agent_num] = 1 # one-hot
        if self.args.use_last_action:
            inputs = np.hstack((inputs, last_action))
        if self.args.reuse_network:
            inputs = np.hstack((inputs, agent_id))
        hidden_state = self.policy.eval_hidden[:,agent_num,:]
        inputs = torch.tensor(inputs, dtype = torch.float32).unsqueeze(0)
        if self.args.cuda:
            inputs= inputs.cuda()
            hidden_state= hidden_state.cuda()
        # get q value and hidden state of #agent_num agent
        q_value, self.policy.eval_hidden[:,agent_num,:] = self.policy.eval_rnn.forward(inputs, hidden_state)
        
        # epsilon greedy
        if np.random.uniform() < epsilon:
            action = np.random.randint(0, self.n_actions)
        else:
            action = torch.argmax(q_value)
        
        return int(action)

    def get_max_episode_len(self, batch):
        terminated = batch['terminated']
        episode_num = terminated.shape[0]
        res = 0
        for i in range(episode_num):
            t_flag = 0
            for t in range(self.args.max_episode_length_limit):
                if terminated[i,t, 0] == 1:
                    t_flag = 1
                    if t + 1 >= res:
                        res = t + 1
                    break
            if t_flag == 0:
                res = self.args.max_episode_length_limit
        return res
    
    def train(self,batch, train_step):
        max_episode_len = self.get_max_episode_len(batch)
        for key in batch.keys():
            batch[key] = batch[key][:,:max_episode_len]
        return self.policy.train(batch, max_episode_len, train_step)