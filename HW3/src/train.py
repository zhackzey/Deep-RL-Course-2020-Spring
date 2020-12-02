# author: Zhu Zeyu
# stuID: 1901111360
'''
    This script implements wrapper for training QMix algo.
'''

import os
import torch
import time
import copy
import argparse

import gfootball.env as football_env
import numpy as np
import matplotlib.pyplot as plt

from multiagent import MultiAgents
from qmix import QMix
from replaybuffer import ReplayBuffer

class DataCollector:
    def __init__(self, env, agents, args):
        self.env = env
        self.agents = agents
        self.args = args
        self.epsilon = args.max_epsilon
    
    def collect_one_episode_data(self, episode_num = None, if_train = True, if_init_buffer = False):
        o, u, u_onehot, r, s, terminated, padded = [], [], [], [], [], [], []
        obs = self.env.reset() # shape (3, 115)
        #state= np.reshape(obs, (345,))
        terminate_flag = False
        step = 0
        episode_reward = 0
        # onehot encoding
        last_action = np.zeros((self.args.n_agents, self.args.n_actions))
        self.agents.policy.init_hidden(1)
        if if_train == False:
            # no exploration for evaluation
            epsilon = 0
        else:
            epsilon = self.epsilon
        
        if if_init_buffer == True:
            epsilon = 1.0

        while terminate_flag == False and step < self.args.max_episode_length_limit:
            state = np.reshape(obs, (345,))
            actions, actions_onehot = [], []
            for agent_id in range(self.args.n_agents):
                action = self.agents.choose_action(obs[agent_id], last_action[agent_id], agent_id, epsilon)
                action_onehot = np.zeros(self.args.n_actions)
                action_onehot[action] = 1
                actions.append(action)
                actions_onehot.append(action_onehot)
                last_action[agent_id] = action_onehot
            
            obs, reward, done, _ = self.env.step(actions)

            o.append(obs)
            s.append(state)
            u.append(np.reshape(actions, (self.args.n_agents,1)))
            u_onehot.append(actions_onehot)
            r.append([reward[0]])
            terminated.append([done])
            padded.append([0.])
            episode_reward += reward[0]
            step += 1

            if done:
                terminate_flag = True
        if not if_init_buffer:
            if epsilon > self.args.min_epsilon:
                epsilon = epsilon - self.args.epsilon_decay_per_episode
            else:
                pass

        o.append(obs)
        state = np.reshape(obs, (345,))
        s.append(state)
        o_next = copy.deepcopy(o[1:])
        s_next = copy.deepcopy(s[1:])
        o = o[:-1]
        s = s[:-1]

        for i in range(step, self.args.max_episode_length_limit):
            o.append(np.zeros((self.args.n_agents, self.args.obs_shape)))
            o_next.append(np.zeros((self.args.n_agents, self.args.obs_shape)))
            s.append(np.zeros( self.args.state_shape))
            s_next.append(np.zeros(self.args.state_shape))
            r.append([0.0])
            u.append(np.zeros((self.args.n_agents, 1)))
            u_onehot.append(np.zeros((self.args.n_agents, self.args.n_actions)))
            padded.append([1.0])
            terminated.append([1.0])
        
        episode_data = dict(o = o.copy(), o_next = o_next.copy(),
                            s = s.copy(), s_next = s_next.copy(),
                            u = u.copy(), u_onehot = u_onehot.copy(),
                            r = r.copy(), padded = padded.copy(), 
                            terminated = terminated.copy())
        
        for key in episode_data.keys():
            episode_data[key] = np.array(episode_data[key])
        if if_train == True:
            self.epsilon = epsilon
        
        return episode_data, episode_reward

class QMixTrainer:
    def __init__(self, env, args):
        self.env = env
        self.args = args
        self.agents = MultiAgents(args)
        self.train_datacollector = DataCollector(self.env, self.agents, args)
        self.replaybuffer = ReplayBuffer(args)

    def evaluate(self):
        mean_episode_reward = 0
        for epsd in range(self.args.eval_episodes):
            _, episode_reward = self.train_datacollector.collect_one_episode_data(if_train = False)
            mean_episode_reward += episode_reward
        
        return mean_episode_reward / self.args.eval_episodes

    def train(self):
        episode_rewards = []
        loss_history = []
        eval_episode_rewards = []
        train_steps = 0
        
        print("Initializing replay buffer...")
        episodes_data = []
        for epsd in range(10000):
            #print("simulating {} episode ...".format(epsd))
            episode_data, episode_reward = self.train_datacollector.collect_one_episode_data(epsd, if_train = False, if_init_buffer = True)
            if episode_reward == 1:
                print("goal !!!!!")
                episodes_data.append(episode_data)
        print("collected {} episodes".format(len(episodes_data)))
        l = len(episodes_data)
        batch_data = {}
        for key in episodes_data[0].keys():
            batch_data[key] = np.zeros((l,) + episodes_data[0][key].shape)
        
        #batch_data = episodes_data[0]
        #episodes_data.pop(0)
        for epsd in range(l):
            for key in batch_data.keys():
                #print("key {} shape {}".format(key,batch_data[key].shape))
                batch_data[key][epsd] = episodes_data[epsd][key]
        
        self.replaybuffer.store_episode(batch_data)        

        print("Start to train")
        plt.figure()
        for epoch in range(self.args.n_epoch):
            print ("Training Epoch {} epsilon: {}".format(epoch,self.train_datacollector.epsilon))

            episodes_data = []
            reward_sum = 0
            for epsd in range(self.args.n_episodes_per_epoch):
                episode_data, episode_reward = self.train_datacollector.collect_one_episode_data(epsd,if_train = True)
                #print("Episode {} reward is {}".format(epsd, episode_reward))
                episodes_data.append(episode_data)
                #reward_sum += episode_reward
                episode_rewards.append(episode_reward)
            #episode_rewards.append(reward_sum / self.args.n_episodes_per_epoch)
        
            batch_data = {}
            for key in episodes_data[0].keys():
                batch_data[key] = np.zeros((self.args.n_episodes_per_epoch,) + episodes_data[0][key].shape)
            
            #batch_data = episodes_data[0]
            #episodes_data.pop(0)
            for epsd in range(self.args.n_episodes_per_epoch):
                for key in batch_data.keys():
                    #print("key {} shape {}".format(key,batch_data[key].shape))
                    batch_data[key][epsd] = episodes_data[epsd][key]
            
            self.replaybuffer.store_episode(batch_data)
            for t_stps in range(self.args.n_train_steps_per_epoch):
                mini_batch = self.replaybuffer.sample(min(self.replaybuffer.current_size,self.args.batch_size))
                loss = self.agents.train(mini_batch, train_steps)
                loss_history.append(loss)
                train_steps = train_steps + 1

            if epoch % self.args.evaluate_freq == 0 :
                mean_episode_reward = self.evaluate()
                eval_episode_rewards.append(mean_episode_reward)

                print("Evaluation Result (Mean Episode Reward) of Epoch {} is : {}".format(epoch, mean_episode_reward))
            
                plt.cla()
                plt.plot(range(len(episode_rewards)), episode_rewards)
                plt.xlabel('episode')
                plt.ylabel('episode reward')
                plt.savefig(os.path.join(self.args.resource_dir,"episode_reward_epoch_{}.png".format(epoch)))
                
                '''
                plt.figure()
                plt.plot(range(len(eval_episode_rewards)), eval_episode_rewards)
                plt.xlabel('episode')
                plt.ylabel('episode reward')
                plt.savefig(os.path.join(self.args.resource_dir,"eval_episode_reward_epoch_{}.png".format(epoch)))
                '''
                
                np.savetxt(os.path.join(self.args.resource_dir,"episode_rewards.txt"), episode_rewards,fmt = "%.4f")
                np.savetxt(os.path.join(self.args.resource_dir,"eval_episode_rewards.txt"),eval_episode_rewards,fmt = "%.4f")
                np.savetxt(os.path.join(self.args.resource_dir,"loss_history.txt"),loss_history)
        plt.cla()
        plt.plot(range(len(episode_rewards)), episode_rewards)
        plt.xlabel('episode')
        plt.ylabel('episode reward')
        plt.savefig(os.path.join(self.args.resource_dir,"episode_reward_epoch_{}.png".format(epoch)))
        
        '''
        plt.figure()
        plt.plot(range(len(eval_episode_rewards)), eval_episode_rewards)
        plt.xlabel('episode')
        plt.ylabel('episode reward')
        plt.savefig(os.path.join(self.args.resource_dir,"eval_episode_reward_epoch_{}.png".format(epoch)))
        '''
        
        np.savetxt(os.path.join(self.args.resource_dir,"episode_rewards.txt"), episode_rewards,fmt = "%.4f")
        np.savetxt(os.path.join(self.args.resource_dir,"eval_episode_rewards.txt"),eval_episode_rewards,fmt = "%.4f")
        np.savetxt(os.path.join(self.args.resource_dir,"loss_history.txt"),loss_history)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Training script for QMix in Google Football RL environment.")
    # training configuration
    parser.add_argument("--seed", type = int, default = 1234, help = "random seed")
    parser.add_argument("--model_dir", type = str, default = "./model", help = "model storage directory")
    parser.add_argument("--resource_dir", type = str, default = "./resource", help = "resources(pictures) storage directory")
    parser.add_argument("--cuda", type = bool, default = True, help = "whether use GPU")
    parser.add_argument("--load_model", type = bool, default = False, help = "whether use former trained model")
    parser.add_argument("--model_idx", type = int, default = 0, help = "number of the model to be loaded")
    parser.add_argument("--learn", type = bool, default = True, help= "whether training")
    parser.add_argument("--evaluate_freq", type = int, default = 100, help = "frequency of evaluation during training")
    parser.add_argument("--buffer_size", type =int, default = 3000, help = "replay buffer size")
    parser.add_argument("--batch_size", type = int, default = 32, help = "batch size")
    # mdp configuration
    parser.add_argument("--gamma", type = float, default = 0.99, help = "mdp discount factor")
    parser.add_argument("--max_epsilon", type = float, default = 1.0, help = "maximum epsilon rate")
    parser.add_argument("--min_epsilon", type = float, default = 0.01, help = "minimum epsilon rate")
    parser.add_argument("--epsilon_decay_episodes", type = int, default = 40000 , help = "linear decay episodes of epsilon")

    # network configuration
    parser.add_argument("--use_last_action", type = bool, default = False, help = "whether use last action in Q prediction")
    parser.add_argument("--reuse_network", type = bool, default = False, help = "whether share Q prediction network between agents")
    parser.add_argument("--rnn_hidden_size", type = int, default = 64, help = "rnn hidden state size")
    parser.add_argument("--qmix_hidden_size", type = int, default = 64, help = "qmix net hidden size")

    # training hyperparameters
    parser.add_argument("--lr", type = float, default = 5e-4, help = "learning rate")
    parser.add_argument("--n_epoch", type = int, default = 20000, help = "training epochs")
    parser.add_argument("--n_episodes_per_epoch", type = int, default = 16, help = "episodes collected in each epoch")
    parser.add_argument("--n_train_steps_per_epoch", type = int, default = 4, help = "peform how many training steps in each epoch")
    parser.add_argument("--eval_episodes", type = int, default = 1, help = "eval episodes number")
    parser.add_argument("--target_update_freq", type = int, default = 4, help ="target network update frequency")
    parser.add_argument("--save_freq", type = int, default = 1000, help = "model save frequency")
    parser.add_argument("--grad_norm_clip",type= float, default = 5, help = "gradient norm clipping value")
    parser.add_argument("--max_episode_length_limit", type = int, default = 150, help = "maximum length limit of each episode")

    args = parser.parse_args()

    env = football_env.create_environment(env_name = "academy_3_vs_1_with_keeper",
                                          stacked = False, representation = "simple115",
                                          number_of_left_players_agent_controls= 3,
                                           write_goal_dumps= False,
                                          write_full_episode_dumps= False, render = False)
    
    args.n_actions = 19
    args.n_agents = 3
    args.state_shape = 345
    args.obs_shape = 115
    args.epsilon = args.max_epsilon
    args.epsilon_decay_per_episode = (args.max_epsilon -args.min_epsilon) / args.epsilon_decay_episodes

    trainer = QMixTrainer(env,args)
    if args.learn:
        trainer.train()
    else:
        mean_episode_reward = trainer.evaluate()
        print("mean episode reward : {}".format(mean_episode_reward))
    
    env.close()

