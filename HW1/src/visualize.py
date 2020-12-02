import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
ckpt_path ="../ckpt/model7"

reward_history = np.loadtxt(os.path.join(ckpt_path,"reward_history.txt")).tolist()
#print(reward_history.shape)
steps_history = np.loadtxt(os.path.join(ckpt_path,"steps_history.txt")).tolist()
loss_history = np.loadtxt(os.path.join(ckpt_path,"loss_history.txt")).tolist()
episode_loss_history = np.loadtxt(os.path.join(ckpt_path,"episode_loss_history.txt")).tolist()

reward_ = pd.Series(reward_history).rolling(window = 100)
steps_ = pd.Series(steps_history).rolling(window = 100)
loss_ = pd.Series(loss_history).rolling(window = 100)
episode_loss_ = pd.Series(episode_loss_history).rolling(window = 100)
fig1 = plt.figure()

#plt.plot(range(len(reward_history)), reward_history)
if len(reward_history) > 100:
    plt.plot(reward_.mean(),c="red")
#plt.plot(reward_.std())
plt.fill_between(range(len(reward_.mean())) ,reward_.mean() +  0.25 * reward_.std(), reward_.mean() - 0.25 * reward_.std(), color = "lightcoral",alpha = 0.4)
plt.grid()
plt.title("Episode Reward (Smoothed)",c='r')
#plt.legend([p],loc="best")
plt.xlabel("Episode")
plt.ylabel("Reward")

fig2= plt.figure()
#plt.plot(range(len(steps_history)), steps_history)
if len(steps_history) > 100:
    plt.plot(steps_.mean(),c = "green")
plt.fill_between(range(len(steps_.mean())) ,steps_.mean() +  0.25 * steps_.std(), steps_.mean() - 0.25 * steps_.std(), color = "lightgreen",alpha = 0.4)
plt.grid()
plt.title("Episode Steps (Smoothed)",c = "green")
#plt.legend(loc="best")
plt.xlabel("Episode")
plt.ylabel("Steps")

fig3 = plt.figure()
#plt.plot(range(len(loss_history)), loss_history)
plt.plot(episode_loss_.mean(),c="dodgerblue")
plt.fill_between(range(len(episode_loss_.mean())) ,episode_loss_.mean() +  0.25 * episode_loss_.std(), episode_loss_.mean() - 0.25 * episode_loss_.std(), color = "deepskyblue",alpha = 0.4)

plt.grid()
plt.title("Episode Loss (Smoothed)",c="dodgerblue")
#plt.legend(loc="best")
plt.xlabel("Episode")
plt.ylabel("Loss")

plt.show()
