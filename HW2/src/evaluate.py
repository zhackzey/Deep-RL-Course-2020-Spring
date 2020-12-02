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

def evaluate(args, model_name, save_interval = 50, eval_interval = 100):
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
                                    write_full_episode_dumps = False, render = True)

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

    print("Evaluating loop...")

    reward_list = []
    # trajectory length of each round in this training episode
    len_list = []

    for i_episode in range(args.num_episodes):

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
            #env.render()

            round_reward += reward
            if args.state_norm:
                next_state = Input_Normalizer(next_state)
            
            if done:
                break
            
            # update
            state = next_state
            
        reward_list.append(round_reward)
        len_list.append(t + 1)
    
    # now we have finished batch data collection
    # note that the data size may not equal to args.batch_size
    print("Evaluate mean reward :", np.mean(reward_list))
    print("Evaluate mean length :", np.mean(len_list))



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = "Train PPO agent in football RL environment.")

    parser.add_argument("--seed", type = int, default = 1234, help = "Random seed")
    parser.add_argument("--num_episodes", type = int, default = 50, help = "# of total evaluating episodes")
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

    evaluate(args, args.model_name, args.save_interval, args.eval_interval)