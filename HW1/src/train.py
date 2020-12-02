# author: Zhu Zeyu
# stuID: 1901111360
'''
Train script
'''
import os
import sys
import gfootball.env as football_env
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from dqn import DQN

import _thread
import re

ckpt_path = "../ckpt/model7"
# use large number :)
train_episodes = 20000
load_ckpt = False
#load_ckpt = True
ckpt_save_freq = 200

env = football_env.create_environment(env_name = "academy_empty_goal", 
                                    representation = "simple115",
                                    number_of_left_players_agent_controls = 1,
                                    stacked = False, logdir = "/tmp/football",
                                    write_goal_dumps = False,
                                    write_full_episode_dumps = False, render = False)


dqn_agent = DQN(
    env = env,
    obs_size = (115,),
    num_frame_stack = 1,
    batch_size = 128,
    mdp_gamma = 0.99,
    initial_epsilon = 1.0,
    min_epsilon = 0.1,
    epsilon_decay_steps = int(1e7),
    replay_capacity = int(1e5),
    min_replay_size = int(1024),
    train_freq = 4,
    network_update_freq = 4
)

dqn_agent.setup_graph(if_soft=True)
sess = dqn_agent.session = tf.Session()

saver = tf.train.Saver(max_to_keep = 100)

if load_ckpt:
    print("load newest ckpt from %s " % ckpt_path)
    ckpt = tf.train.get_checkpoint_state(ckpt_path)
    assert ckpt, "newest checkpoint not found in %s" % ckpt_path
    global_counter = int(re.findall("-(\d+)$", ckpt.model_checkpoint_path)[0])
    saver.restore(sess, ckpt.model_checkpoint_path)
    dqn_agent.global_counter = global_counter
else:
    if ckpt_path is not None:
        assert not os.path.exists(ckpt_path), "path exists already, but load_ckpt is False"
    
    dqn_agent.session.run(tf.global_variables_initializer())

def save_ckpt():
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    
    name = os.path.join(ckpt_path,"model.ckpt")
    saver.save(dqn_agent.session, name, dqn_agent.global_counter)
    print("saved to %s - %d" % (name, dqn_agent.global_counter))

def input_thread(list):
    key = input("=================== Press Enter to stop training =================")
    list.append("sss")

def train_loop():
    reward_history = []
    steps_history = []
    episode_loss_history = []

    if load_ckpt:
        reward_history = np.loadtxt(os.path.join(ckpt_path,"reward_history.txt")).tolist()
        #print(reward_history.shape)
        steps_history = np.loadtxt(os.path.join(ckpt_path,"steps_history.txt")).tolist()
        dqn_agent.loss_history = np.loadtxt(os.path.join(ckpt_path,"loss_history.txt")).tolist()
        episode_loss_history = np.loadtxt(os.path.join(ckpt_path,"episode_loss_history.txt")).tolist()

    print("Start training DQN.")
    print("=============================================")
    sys.stdout.flush()
    l = []
    _thread.start_new_thread(input_thread,(l,))

    while True:
        if l:
            break
        if dqn_agent.is_training and dqn_agent.episode_counter > train_episodes:
            break
        
        sum_reward, steps_in_episode, episode_loss = dqn_agent.play_episode()
        epsilon = dqn_agent.get_epsilon()
        print("episode: %d, reward: %f, episode steps: %d, epsilon: %f, episode loss: %f, total steps: %d" % (dqn_agent.episode_counter, sum_reward, steps_in_episode,epsilon, episode_loss, dqn_agent.global_counter))        
        
        reward_history.append(sum_reward)
        steps_history.append(steps_in_episode)
        episode_loss_history.append(episode_loss)

        if dqn_agent.episode_counter % ckpt_save_freq == 0 and dqn_agent.is_training:
            save_ckpt()
            np.savetxt(os.path.join(ckpt_path,"reward_history.txt"),reward_history)
            np.savetxt(os.path.join(ckpt_path,"steps_history.txt"),steps_history)
            np.savetxt(os.path.join(ckpt_path,"loss_history.txt"),dqn_agent.loss_history)
            np.savetxt(os.path.join(ckpt_path,"episode_loss_history.txt"),episode_loss_history)

            reward_ = pd.Series(reward_history).rolling(window = 100)
            steps_ = pd.Series(steps_history).rolling(window = 100)
            episode_loss_ = pd.Series(episode_loss_history).rolling(window = 100)

            fig1 = plt.figure()

            plt.subplot(2,2,1)

            #plt.plot(range(len(reward_history)), reward_history)
            if len(reward_history) > 50:
                plt.plot(reward_.mean(),c="red")
            plt.fill_between(range(len(reward_.mean())) ,reward_.mean() +  0.25 * reward_.std(), reward_.mean() - 0.25 * reward_.std(), color = "lightcoral",alpha = 0.4)
            plt.grid()
            plt.title("Episode Reward (Smoothed)",c='r')
            plt.xlabel("Episode")
            plt.ylabel("Reward")

            #fig2= plt.figure()
            plt.subplot(2,2,2)
            #plt.plot(range(len(steps_history)), steps_history)
            if len(steps_history) > 50:
                plt.plot(steps_.mean(),c="green")
            plt.fill_between(range(len(steps_.mean())) ,steps_.mean() +  0.25 * steps_.std(), steps_.mean() - 0.25 * steps_.std(), color = "lightgreen",alpha = 0.4)
            plt.grid()
            plt.title("Episode Steps (Smoothed)",c = "green")
            plt.xlabel("Episode")
            plt.ylabel("Steps")
            
            #fig3 = plt.figure()
            plt.subplot(2,2,3)
            #plt.plot(range(len(dqn_agent.loss_history)), dqn_agent.loss_history)
            #plt.plot(range(len(episode_loss_history)), episode_loss_history)
            if len(episode_loss_history) > 50:
                plt.plot(episode_loss_.mean(),c="dodgerblue")
            plt.fill_between(range(len(episode_loss_.mean())) ,episode_loss_.mean() +  0.25 * episode_loss_.std(), episode_loss_.mean() - 0.25 * episode_loss_.std(), color = "deepskyblue",alpha = 0.4)

            plt.grid()
            plt.title("Episode Loss (Smoothed)",c="dodgerblue")
            plt.xlabel("Episode")
            plt.ylabel("Loss")

            #plt.show()

            plt.ion()
            plt.pause(5)
            plt.close(fig1)
            #plt.close(fig2)
            #plt.close(fig3)

    print("Training stopped by user.")
    
    reward_ = pd.Series(reward_history).rolling(window = 50)
    steps_ = pd.Series(steps_history).rolling(window = 50)
    episode_loss_ = pd.Series(episode_loss_history).rolling(window = 100)

    np.savetxt(os.path.join(ckpt_path,"reward_history.txt"),reward_history)
    np.savetxt(os.path.join(ckpt_path,"steps_history.txt"),steps_history)
    np.savetxt(os.path.join(ckpt_path,"loss_history.txt"),dqn_agent.loss_history)
    np.savetxt(os.path.join(ckpt_path,"episode_loss_history.txt"),episode_loss_history)
    
    plt.figure()

    #plt.plot(range(len(reward_history)), reward_history)
    if len(reward_history) > 50:
        plt.plot(reward_.mean())
    plt.xlabel("Episode")
    plt.ylabel("Reward")

    plt.figure()
    #plt.plot(range(len(steps_history)), steps_history)
    if len(steps_history) > 50:
        plt.plot(steps_.mean())
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    
    plt.figure()
    #plt.plot(range(len(dqn_agent.loss_history)), dqn_agent.loss_history)
    #plt.plot(range(len(episode_loss_history)), episode_loss_history)
    if len(episode_loss_history) > 50:
        plt.plot(episode_loss_.mean())
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.ion()



    #plt.pause(5)

if train_episodes > 0:
    print("=================Train==================")
    sys.stdout.flush()
    train_loop()
    save_ckpt()
    