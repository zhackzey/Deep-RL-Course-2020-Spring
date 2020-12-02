# author: Zhu Zeyu
# stuID: 1901111360
'''
    This script implements plotting some figures.
'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
ckpt_path ="./resource/try9"

reward_history = np.loadtxt(os.path.join(ckpt_path,"episode_rewards.txt")).tolist()
#reward_history = np.loadtxt(os.path.join(ckpt_path,"reward_history.txt")).tolist()[:800]

reward_ = pd.Series(reward_history).rolling(window =100)

loss_history = np.loadtxt(os.path.join(ckpt_path,"loss_history.txt")).tolist()
loss_ = pd.Series(loss_history).rolling(window = 100)

fig1 = plt.figure()

#plt.plot(range(len(reward_history)), reward_history)
plt.plot(reward_.mean(),c="red")
#plt.plot(reward_.std())
plt.fill_between(range(len(reward_.mean())) ,reward_.mean() +  0.1 * reward_.std(), reward_.mean() - 0.1 * reward_.std(), color = "lightcoral",alpha = 0.4)
plt.grid()
plt.title("Episode Reward (Smoothed)",c='r')
#plt.legend([p],loc="best")
plt.xlabel("Episode")
plt.ylabel("Reward")
#plt.show()

fig3 = plt.figure()
#plt.plot(range(len(loss_history)), loss_history)
plt.plot(loss_.mean(),c="dodgerblue")
plt.fill_between(range(len(loss_.mean())) ,loss_.mean() +  0.25 * loss_.std(), loss_.mean() - 0.25 * loss_.std(), color = "deepskyblue",alpha = 0.4)

plt.grid()
plt.title("Batch Loss (Smoothed)",c="dodgerblue")
#plt.legend(loc="best")
plt.xlabel("Train steps")
plt.ylabel("Loss")
plt.show()