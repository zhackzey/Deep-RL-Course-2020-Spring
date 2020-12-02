# author: Zhu Zeyu
# stuID: 1901111360
'''
    This script implements the qmix algorithm class.
'''
import os
import copy
import torch
import numpy as np
import torch.nn as nn 
import torch.nn.functional as F

'''
    RNN module for predicting each agent's Q value.
'''
class RNN(nn.Module):
    def __init__(self, input_shape, args):
        super(RNN, self).__init__()
        self.input_shape = input_shape
        self.args = args

        self.fc1 = nn.Linear(self.input_shape, self.args.rnn_hidden_size)
        self.rnn = nn.GRUCell(self.args.rnn_hidden_size, self.args.rnn_hidden_size)
        self.fc2 = nn.Linear(self.args.rnn_hidden_size, self.args.n_actions)

    def forward(self, obs, hidden_state):
        x = F.relu(self.fc1(obs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_size)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h
'''
    QMixNet responsible for combining multi-agents' Q values into one global Q value.
'''

class QMixNet(nn.Module):
    def __init__(self, args):
        super(QMixNet, self).__init__()
        self.args = args
        # Layer 1  
        # Here we actually want to get a matrix W1. So use pytorch's Linear to output a tensor of size args.n_agents * args.qmix_hidden_size
        # then we can use view to turn this tensor into a matrix
        self.hyper_w1 = nn.Linear(self.args.state_shape, self.args.n_agents * self.args.qmix_hidden_size)
        self.hyper_b1 = nn.Linear(self.args.state_shape, self.args.qmix_hidden_size)

        # Layer 2
        self.hyper_w2 = nn.Linear(self.args.state_shape, self.args.qmix_hidden_size)
        
        self.hyper_b2 = nn.Sequential(nn.Linear(self.args.state_shape, self.args.qmix_hidden_size),
                                      nn.ReLU(),
                                      nn.Linear(self.args.qmix_hidden_size, 1))
        
    
    def forward(self, q_values, states):
        '''
            forward step of QMixNet.
            input_params q_values: tensor. shape (episode_num, max_episode_len, n_agents)
            input params states: numpy ndarray. shape (episode_num, max_episode_len, state_shape)
        '''
        episode_num = q_values.size(0)

        states = states.reshape(-1, self.args.state_shape) # shape (episode_num * max_episode_len, state_shape)
        w1 = torch.abs(self.hyper_w1(states))
        b1 = self.hyper_b1(states)

        # turn this tensor into a matrix
        w1 = w1.view(-1, self.args.n_agents, self.args.qmix_hidden_size) # shape (episode_num * max_episode_len, n_agents, qmix_hidden_size)
        b1 = b1.view(-1, 1, self.args.qmix_hidden_size) # shape (episode_num * max_episode_len, 1, qmix_hidden_size)

        q_values = q_values.view(-1, 1, self.args.n_agents) # shape (episode_num * max_episode_len, 1, n_agents)

        x = F.elu(torch.bmm(q_values, w1) + b1) # shape (episode_num * max_episode_len, 1, qmix_hidden_size)
        #x = F.elu(torch.bmm(q_values, w1))
        w2 = torch.abs(self.hyper_w2(states))   
        b2 = self.hyper_b2(states)

        w2 = w2.view(-1, self.args.qmix_hidden_size, 1) # shape (episode_num * max_episode_len, qmix_hidden_size, 1)
        b2 = b2.view(-1, 1, 1)  # shape (episode_num * max_episode_len, 1, 1)

        q_tot = torch.bmm(x, w2) + b2 # shape (episode_num * max_episode_len, 1, 1)
        #q_tot = torch.bmm(x, w2)
        q_tot = q_tot.view(episode_num, -1, 1) # shape (episode_num, max_episode_len, 1)
        return q_tot

'''
    QMix algorithm class.
'''
class QMix:
    def __init__(self, args):
        self.args = args
        self.state_shape = args.state_shape
        self.obs_shape = self.input_shape = args.obs_shape
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.model_dir = args.model_dir 
        self.eval_hidden = None
        self.target_hidden = None
        
        if args.use_last_action:
            self.input_shape += self.n_actions
        if args.reuse_network:
            self.input_shape += self.n_agents
        # TODO: check input_shape
        self.eval_rnn = RNN(self.input_shape,args)
        self.target_rnn = RNN(self.input_shape,args)
        self.eval_qmix = QMixNet(args)
        self.target_qmix = QMixNet(args)
        # send to cuda
        if self.args.cuda:
            self.eval_rnn.cuda()
            self.target_rnn.cuda()
            self.eval_qmix.cuda()
            self.target_qmix.cuda()
        
        if self.args.load_model:
            path_rnn = os.path.join(self.model_dir ,str(self.args.model_idx) + '_rnn_params.pkl')
            path_qmix = os.path.join(self.model_dir ,str(self.args.model_idx) + '_qmix_params.pkl')
            if os.path.exists(path_rnn) and os.path.exists(path_qmix):
                self.eval_rnn.load_state_dict(torch.load(path_rnn))
                self.eval_qmix_net.load_state_dict(torch.load(path_qmix))
                print('Successfully load the model: {} and {}'.format(path_rnn, path_qmix))
            else:
                raise Exception("No model!")
        
        self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
        self.target_qmix.load_state_dict(self.eval_qmix.state_dict())

        self.eval_params = list(self.eval_qmix.parameters()) + list(self.eval_rnn.parameters())
        self.optimizer = torch.optim.RMSprop(self.eval_params, lr = args.lr)

    def init_hidden(self, episode_num):
        self.eval_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_size))
        self.target_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_size))

    def get_inputs(self, batch, transition_idx):
        '''
            get episode_num episodes' inputs of timestamp transition_idx
            input_params batch: Dict. batch data
            input_params transition_idx: Int. timestamp
        '''
        obs, obs_next, u_onehot = batch['o'][:, transition_idx], batch['o_next'][:,transition_idx], batch['u_onehot'][:]

        # obs      shape [episode_num, n_agents, obs_shape]
        # obs_next shape [episode_num, n_agents, obs_shape]
        # u_onehot shape [episode_num, n_agents, n_agents]
        # optional agent index(onehot) shape [episode, n_agents, n_agents]
        # TODO: assert
        episode_num = obs.shape[0]
        inputs = []
        inputs.append(obs)
        inputs_next = []
        inputs_next.append(obs_next)

        if self.args.use_last_action:
            # append actions into inputs list
            if transition_idx == 0 :
                inputs.append(torch.zeros_like(u_onehot[:, transition_idx]))
            else:
                inputs.append(u_onehot[:, transition_idx - 1])
            inputs_next.append(u_onehot[:, transition_idx])
        
        if self.args.reuse_network:
            # append the idx of agent into inputs list
            inputs.append(torch.eye(self.n_agents).unsqueeze(0).expand(episode_num, -1, -1))
            inputs_next.append(torch.eye(self.n_agents).unsqueeze(0).expand(episode_num, -1, -1))
        
        # flatten obs, u_onehot, (agent index) into one vector 
        inputs = torch.cat([x.reshape(episode_num * self.n_agents, -1) for x in inputs],dim = 1)
        inputs_next = torch.cat([x.reshape(episode_num * self.n_agents, -1) for x in inputs_next],dim = 1)
        
        #assert inputs.shape == self.input_shape 

        return inputs, inputs_next
    
    def compute_q_values(self, batch, max_episode_len):
        '''
            Compute q_values of this batch data.
            input_params batch: Dict. batch data
            input_params max_episode_len: int, max length of an episode
        '''
        episode_num = batch['o'].shape[0]
        q_eval_list =[]
        q_target_list = []
        for t in range(max_episode_len):
            inputs, inputs_next = self.get_inputs(batch, t)
            if self.args.cuda:
                inputs = inputs.cuda()
                inputs_next = inputs_next.cuda()
                self.eval_hidden = self.eval_hidden.cuda()
                self.target_hidden = self.target_hidden.cuda()
            q_eval, self.eval_hidden = self.eval_rnn(inputs, self.eval_hidden)
            q_target, self.target_hidden = self.target_rnn(inputs_next, self.target_hidden)

            q_eval = q_eval.view(episode_num, self.n_agents, -1)
            q_target = q_target.view(episode_num, self.n_agents, -1)
            
            q_eval_list.append(q_eval)
            q_target_list.append(q_target)

        # shape (episode_num, max_episode_len, n_agents, n_actions)
        return torch.stack(q_eval_list, dim = 1), torch.stack(q_target_list, dim = 1)
    
    def train(self, batch, max_episode_len, train_step):
        '''
            One train step with respect to batch data.
            input_params batch: Dict. batch data
            input_params max_episode_len: maximum length of an episode
            input_params train_step: int. the idx of train steps. Used to control target network update
        '''
        episode_num = batch['o'].shape[0]
        self.init_hidden(episode_num)
        # convert numpy ndarray into torch's tensors
        for key in batch.keys():
            if key == 'u':
                batch[key] = torch.tensor(batch[key], dtype = torch.long)
            else:
                batch[key] = torch.tensor(batch[key], dtype = torch.float32)
        
        s, s_next, u, r, terminated = batch['s'], batch['s_next'], batch['u'], batch['r'], batch['terminated']
        # padding mask. Some entries in batch is used for padding. mask them
        mask = 1.0 - batch['padded'].float()
        if self.args.cuda:
            s = s.cuda()
            s_next = s_next.cuda()
            u = u.cuda()
            r = r.cuda()
            terminated = terminated.cuda()
            mask = mask.cuda()

        q_evals, q_targets = self.compute_q_values(batch, max_episode_len)

        # note that q_evals and q_targets are of shape (episode_num, max_episode_len , n_agents, n_actions)
        # i.e it contains all q_action values. However, we need only one to compute target error
        q_evals = torch.gather(q_evals, dim = 3, index = u)
        q_evals = q_evals.squeeze(3)
        # max q
        q_targets = q_targets.max(dim = 3)[0]

        q_tot_eval = self.eval_qmix(q_evals,s)
        q_tot_target = self.target_qmix(q_targets,s_next)

        td_targets = r + self.args.gamma * q_tot_target * (1.0 - terminated)

        td_error = (q_tot_eval - td_targets.detach())
        masked_td_error = mask * td_error
        #print(masked_td_error)
        loss = (masked_td_error **2).sum()/mask.sum()
        print("loss: {}".format(loss))
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.eval_params, self.args.grad_norm_clip)

        self.optimizer.step()
        if train_step and train_step % self.args.target_update_freq == 0:
            rnn_state_dict = copy.deepcopy(self.eval_rnn.state_dict())
            for param in rnn_state_dict.keys():
                rnn_state_dict[param] = 0.999 * self.target_rnn.state_dict()[param] + 0.001 * self.eval_rnn.state_dict()[param]

            qmix_state_dict = copy.deepcopy(self.eval_qmix.state_dict())
            for param in qmix_state_dict.keys():
                qmix_state_dict[param] = 0.999 * self.target_qmix.state_dict()[param] + 0.001 *self.eval_qmix.state_dict()[param]

            #self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
            #self.target_qmix.load_state_dict(self.eval_qmix.state_dict())
            self.target_rnn.load_state_dict(rnn_state_dict)
            self.target_qmix.load_state_dict(qmix_state_dict)

        if train_step and train_step % self.args.save_freq == 0:
            if not os.path.exists(self.model_dir):
                os.makedirs(self.model_dir)
            torch.save(self.eval_rnn.state_dict(),os.path.join( self.model_dir, str(train_step) + '_rnn_params.pkl'))
            torch.save(self.eval_qmix.state_dict(), os.path.join(self.model_dir,str(train_step) + '_qmix_params.pkl'))
        
        return loss