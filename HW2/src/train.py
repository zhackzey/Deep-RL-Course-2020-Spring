# author: Zhu Zeyu
# stuID: 1901111360
'''
This script is for training PPO agent in Google football RL environment
'''
import os
import re
import shutil
import argparse

import numpy as np
import tensorflow as tf
import pandas as pd
import gfootball.env as football_env
import time

from ppo import ProximalPolicyOptimization
from utils import Dataset, InputNormalization

class Args(object):
    def __init__(self,params):
        self.seed = params['seed']
        self.num_episodes = params['num_episodes']
        self.batch_size = params['batch_size']
        self.max_step_per_round = params['max_step_per_round']
        self.gamma = params['gamma']
        self.lamda = params['lamda']
        self.log_num_episode = params['log_num_episode']
        self.num_epoch = params['num_epoch']
        self.minibatch_size = params['minibatch_size']
        self.clip = params['clip']
        self.loss_coeff_value = params['loss_coeff_value']
        self.loss_coeff_entropy = params['loss_coeff_entropy']
        self.lr = params['lr']
        # tricks
        self.schedule_adam = params['schedule_adam']
        self.schedule_clip = params['schedule_clip']
        self.layer_norm = params['layer_norm']
        self.state_norm = params['state_norm']
        self.advantage_norm = params['advantage_norm']
        self.EPS = params['EPS']
        self.model_name = params['model_name']
        self.save_interval = params['save_interval']
        self.eval_interval = params['eval_interval']

def train(args, model_name, save_interval = 50, eval_interval = 100):
    """
        Train function.
        param args: Args object.
        param model_name: string, model name
        param save_interval: int, save interval
        param eval_interval: int, eval interval
    """

    print("Creating football RL environment...")
    env = football_env.create_environment(env_name = "academy_empty_goal", 
                                    representation = "simple115",
                                    number_of_left_players_agent_controls = 1,
                                    stacked = False, logdir = "/tmp/football",
                                    write_goal_dumps = False,
                                    write_full_episode_dumps = False, render = False)

    assert env is not None, "Create environment failed"
    print("environment created.")

    input_shape = env.observation_space.shape
    num_actions = env.action_space.n

    env.seed(args.seed)
    
    print("Creating ppo model")
    model = ProximalPolicyOptimization(input_shape, num_actions,
                                       entropy_coef = args.loss_coeff_entropy,
                                       value_coef = args.loss_coeff_value,
                                       model_name = model_name
                                       )
    
    # input states normalizer
    Input_Normalizer = InputNormalization(input_shape, clip = 5.0)

    # records like average cumulative reward (one round) in every training episode
    episode_reward_record = []
    episode_loss_record = []

    # total transitions/samples 
    global_steps = 0

    lr_now = args.lr
    clip_now = args.clip

    print("Training loop...")
    for i_episode in range(args.num_episodes):
        # step1: collect trajectories with current policy
        # episode dataset, used to store collected trajectories
        dataset = Dataset()
        # number of samples already collected in this training episode
        num_steps = 0
        # trajectory reward of each round in this training episode
        reward_list = []
        # trajectory length of each round in this training episode
        len_list = []
        print("Start to collect trajectories...")
        while num_steps < args.batch_size:
            # if not enough samples for one batch
            # continue collecting new trajectories in one new round
            state = env.reset()

            # if perform normalization on input states
            if args.state_norm:
                state = Input_Normalizer(state)
            
            # total reward in this round
            round_reward = 0
            # simulate up to max_step_per_round steps

            for t in range(args.max_step_per_round):
                # use current policy to choose action, predict value
                action, value, neglogp = model.predict(input_states = state[None], use_policy_old = False)
                action = action[0]
                value = value[0]
                neglogp = neglogp[0]

                #print("model predict time: {:.4f}".format(ed - st))
                # step in RL environment using model predicted action
                next_state, reward, done, _ = env.step(action)

                round_reward += reward
                if args.state_norm:
                    next_state = Input_Normalizer(next_state)
                
                if done:
                    mask = 0
                else:
                    mask = 1
                
                dataset.push(state, value, action, - neglogp, mask, next_state, reward)

                if done:
                    break
                
                # update
                state = next_state
            
            # now we have finished one round
            # time to update some statistics
            print("One trajectory of length {} collected. Already collected {} samples in this batch".format(t + 1, num_steps))
            num_steps += (t + 1)
            global_steps += (t + 1)
            reward_list.append(round_reward)
            len_list.append(t + 1)
        
        # now we have finished batch data collection
        # note that the data size may not equal to args.batch_size

        # update statistics
        episode_reward_record.append({
            'episode':i_episode,                   # training episode number
            'steps': global_steps,                 # total steps so far
            'episode_mean_reward':np.mean(reward_list),     # episode mean reward
            'episode_mean_length':np.mean(len_list)         # episode mean length
        })

        batch_data = dataset.data()
        batch_size = dataset.size()

        # step 2. compute returns, advantages, ...
        rewards = np.asarray(batch_data.reward)
        values = np.asarray(batch_data.value)
        masks = np.asarray(batch_data.mask)
        actions = np.asarray(batch_data.action)
        states = np.asarray(batch_data.state)
        
        # to be computed
        returns = np.zeros(batch_size)
        deltas = np.zeros(batch_size)
        advantages = np.zeros(batch_size)

        prev_return = 0.0
        prev_value = 0.0
        prev_advantage = 0.0
        # compute in a reverse order
        for i in reversed(range(batch_size)):
            returns[i] = rewards[i] + args.gamma * prev_return * masks[i]
            deltas[i] = rewards[i] + args.gamma * prev_value * masks[i] - values[i]
            advantages[i] = deltas[i] + args.gamma * args.lamda * prev_advantage * masks[i]
            
            prev_return = returns[i]
            prev_value = values[i]
            prev_advantage = advantages[i]

        # if perform normalization on advantages
        if args.advantage_norm:
            advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + args.EPS)
        
        # step 3. train for some number of epochs
        # note that we use policy to collect trajectories
        # and now we are optimizing on paramters of policy
        # so we need to update policy_old to policy first
        model.update_old_policy() 

        total_loss_list = []
        policy_loss_list = []
        value_loss_list = []
        entropy_loss_list = []

        for i_epoch in range(int(args.num_epoch * batch_size / args.minibatch_size)):
            # sample mini-batch data from current batch data
            minibatch_indices = np.random.choice(batch_size, args.minibatch_size, replace = False) 
            minibatch_states = states[minibatch_indices]
            minibatch_actions = actions[minibatch_indices]
            minibatch_advantages = advantages[minibatch_indices]
            minibatch_returns = returns[minibatch_indices]           
            
            # determine learning rate and cliprange parameters
            if args.schedule_adam == "linear":
                ep_ratio = 1 - (i_episode / args.num_episodes)
                lr_now = args.lr * ep_ratio
            
            if args.schedule_clip == "linear":
                ep_ratio = 1 - (i_episode / args.num_episodes)
                clip_now = args.clip * ep_ratio

            # Optimize model
            total_loss, policy_loss, value_loss, entropy_loss =model.train( minibatch_states, 
                                                                            minibatch_actions, 
                                                                            minibatch_returns, 
                                                                            minibatch_advantages,
                                                                            clip_now,
                                                                            lr_now)

            total_loss_list.append(total_loss)
            policy_loss_list.append(policy_loss)
            value_loss_list.append(value_loss)
            entropy_loss_list.append(entropy_loss)


        episode_loss_record.append({
            'episode': i_episode,
            'episode_mean_total_loss' : np.mean(total_loss_list),
            'episode_mean_policy_loss' : np.mean(policy_loss_list),
            'episode_mean_value_loss' : np.mean(value_loss_list),
            'episode_mean_entropy_loss' : np.mean(entropy_loss_list)
        })   
            
        # Save model
        if i_episode % save_interval == 0:
            model.save()
            
            # save loss to txt
            t_loss =[ episode_loss_record[k]['episode_mean_total_loss'] for k in range(len(episode_loss_record))]
            p_loss = [ episode_loss_record[k]['episode_mean_policy_loss'] for k in range(len(episode_loss_record))]
            v_loss = [ episode_loss_record[k]['episode_mean_value_loss'] for k in range(len(episode_loss_record))]
            e_loss = [ episode_loss_record[k]['episode_mean_entropy_loss'] for k in range(len(episode_loss_record))]

            np.savetxt(os.path.join(model.log_dir, "episode_mean_total_loss.txt"), t_loss, fmt = "%.6f")
            np.savetxt(os.path.join(model.log_dir, "episode_mean_policy_loss.txt"), p_loss, fmt = "%.6f")
            np.savetxt(os.path.join(model.log_dir, "episode_mean_value_loss.txt"), v_loss, fmt = "%.6f")
            np.savetxt(os.path.join(model.log_dir, "episode_mean_entropy_loss.txt"), e_loss, fmt = "%.6f")

        if i_episode % args.log_num_episode == 0:
            print("============================= Episode: {} ===============================".format(i_episode))
            print(" Mean_Reward: {:.4f}  Mean_Length: {:.0f} ".format(episode_reward_record[-1]['episode_mean_reward'], episode_reward_record[-1]['episode_mean_length']))
            print(" Mean_Total_Loss: {:.4f}  Mean_Policy_Loss: {:.4f}  Mean_Value_Loss: {:.4f} Mean_Entropy_Loss: {:.4f}" \
                .format(episode_loss_record[-1]['episode_mean_total_loss'], 
                        episode_loss_record[-1]['episode_mean_policy_loss'],
                        episode_loss_record[-1]['episode_mean_value_loss'],
                        episode_loss_record[-1]['episode_mean_entropy_loss']))

            model.write_to_summary('episode_mean_reward', episode_reward_record[-1]['episode_mean_reward'], i_episode)
            model.write_to_summary('episode_mean_length',episode_reward_record[-1]['episode_mean_length'],i_episode)
            model.write_to_summary('episode_mean_total_loss',episode_loss_record[-1]['episode_mean_total_loss'], i_episode)
            model.write_to_summary('episode_mean_policy_loss',episode_loss_record[-1]['episode_mean_policy_loss'], i_episode)
            model.write_to_summary('episode_mean_value_loss',episode_loss_record[-1]['episode_mean_value_loss'], i_episode)
            model.write_to_summary('episode_mean_entropy_loss',episode_loss_record[-1]['episode_mean_entropy_loss'], i_episode)
  
        # Evaluate model
        if i_episode % eval_interval == 0:
            # TODO: maybe call render version of football to evaluate
            pass

    print("Training total {} episodes done.".format(args.num_episodes))

    return episode_reward_record, episode_loss_record

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = "Train PPO agent in football RL environment.")

    parser.add_argument("--seed", type = int, default = 1234, help = "Random seed")
    parser.add_argument("--num_episodes", type = int, default = 2000, help = "# of total training episodes")
    parser.add_argument("--batch_size", type = int, default = 2048, help = "# of samples in one batch")
    parser.add_argument("--max_step_per_round", type = int, default = 200, help = "time horizon for each simulation")
    parser.add_argument("--gamma", type = float, default = 0.995, help = "mdp discount factor")
    parser.add_argument("--lamda", type = float, default = 0.97, help ="gae lamda")
    parser.add_argument("--log_num_episode", type = int, default = 1, help= "log every # episodes ")
    parser.add_argument("--num_epoch", type = int, default = 10, help = "# of epochs for each batch")
    parser.add_argument("--minibatch_size", type = int, default = 256, help = "# of samples in one mini batch")
    parser.add_argument("--clip", type = float, default = 0.2, help = "initial ppo clipping parameter")
    parser.add_argument("--loss_coeff_value", type = float, default = 0.5, help = "weight of value loss")
    parser.add_argument("--loss_coeff_entropy", type = float, default = 0.01, help = "weight of entropy loss")
    parser.add_argument("--lr", type = float, default = 3e-4, help = "initial learning rate")
    
    parser.add_argument("--schedule_adam", type = str, default = "linear", help = "adam lr scheduler")
    parser.add_argument("--schedule_clip", type = str, default = "linear", help = "clip scheduler")
    parser.add_argument("--layer_norm", type = bool, default = True, help = "if perform layer normalization")
    parser.add_argument("--state_norm", type = bool, default = True, help = "if perform state normalization")
    parser.add_argument("--advantage_norm", type = bool, default = True, help = "if perform advantages normalization")
    parser.add_argument("--EPS", type = float, default = 1e-10, help = "small constant")

    parser.add_argument("--model_name", type=str, default = "ppo", help = "model name for storage")
    parser.add_argument("--save_interval", type=int, default=50, help = "save model for every # episodes")
    parser.add_argument("--eval_interval", type=int, default=100, help = "evaluate model for every # episodes")

    args = Args(vars(parser.parse_args()))

    train(args, args.model_name, args.save_interval, args.eval_interval)