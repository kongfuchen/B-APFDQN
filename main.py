#!/usr/bin/env python
# coding=utf-8
import matplotlib.pyplot as plt
import torch
import numpy as np
from Env import Env
from agent import DQN
import random
import math


from matplotlib.ticker import MultipleLocator, FormatStrFormatter

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"



class Config:
    """
    Initializing parameters
    """
    def __init__(self):
        self.gamma = 0.9     # Discount Factor
        self.lr = 0.0001     # Learning rate of neural networks
        self.memory_capacity = 1000000     # Capacity of memory replay
        self.batch_size = 32
        self.train_eps = 100
        self.target_update = 2     # Frequency of target network updates
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hidden_dim = 256     # Number of neurons in the hidden layers
        self.n_layer=2     # Number of hidden layers
        self.best_x = []
        self.best_y = []
        self.epsilon_start = 1
        self.epsilon_end = 0.01
        self.epsilon_decay = 800     # eplison update delay

def show(x0,y0, env,visit,episode,iter):
    """
    Search process show
    """
    plt.ion()
    obs = [[[0, 0], [3, 3]], [[3, 3], [8, 8]], [[5, 5], [7, 7]], [[7, 7], [2, 2]], [[2, 2], [0, 0]], [[3, 3], [1, 1]],
           [[9, 9], [8, 8]], [[2, 2], [9, 9]], [[2, 2], [3, 3]], [[5, 5], [9, 9]], [[10, 10], [1, 1]], [[1, 1], [1, 1]],
           [[2, 2], [6, 6]], [[9, 9], [7, 7]], [[4, 4], [1, 1]], [[10, 10], [6, 6]], [[8, 8], [9, 9]], [[2, 2], [8, 8]]]

    map = np.ones((11, 11))
    map_grid = np.full((11, 11), int(10), dtype=np.int8)
    for i in range(11):
        for j in range(11):
            V=0
            for v in visit:
                if v==[i,j]:
                    V=V+1
            map[j][i]=10-10/1000*V
            for k in range(len(obs)):
                if obs[k][0][0] <= i <= obs[k][0][1] \
                        and obs[k][1][0] <= j <= obs[k][1][1]:
                    map[j][i]=0
    for i in range(11):
        for j in range(11):
            map_grid[j,i]=map[j][i]
    plt.plot([x0, env.x], [y0, env.y], c='b')

    if (env.x == 0 and env.y == 0) or (env.x == 10 and env.y == 10):
        plt.clf()
    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax.xaxis.set_minor_locator(MultipleLocator(1 / 2))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax.yaxis.set_minor_locator(MultipleLocator(1 / 2))
    ax.xaxis.grid(True, which='minor')
    ax.yaxis.grid(True, which='minor')
    plt.title('episode:{}/{}, iters:{}'.format(episode + 1, 100, iter))
    plt.imshow(map_grid, cmap=plt.cm.hot, interpolation='nearest', vmin=0, vmax=10)
    plt.pause(0.0001)
    plt.ioff()

def train(cfg, env, agent):
    """
    Training process
    """
    rewards = []
    BestIters=math.inf
    count = 0
    ITER=[]
    visit = []
    for i_episode in range(cfg.train_eps):
        state= env.reset()
        done = False
        ep_reward = 0

        iter=1

        while not done:
            x0=env.x
            y0=env.y
            env.x_g.append(env.x)
            env.y_g.append(env.y)
            visit.append([env.x,env.y])
            action = agent.action_selection(state, env)
            state1, reward, done, label,next_label = env.step(action)
            show(x0,y0,env,visit,i_episode,iter)


            if iter>4000:
                return [],[],[]

            count = count+1
            reward = reward
            ep_reward += reward
            agent.memory.push(state, action, reward,state1, done,label,next_label)
            state = state1
            agent.update_net(env)
            iter=iter+1
        visit.append([env.target_x,env.target_y])

        if iter<BestIters:
            BestIters=iter
            env.best_x=env.x_g
            env.best_y=env.y_g
        if i_episode % cfg.target_update == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())

        print('episode:{}/{}, Reward:{}, iters:{}'.format(i_episode + 1, cfg.train_eps, ep_reward, iter))
        rewards.append([ep_reward,iter])
    print('Complete training！')
    np.save("T.npy",visit)
    return rewards




if __name__ == "__main__":
    cfg = Config()
    env = Env()

    state_dim = 2
    action_dim = 8

    random.seed(1)
    agent = DQN(state_dim, action_dim, cfg)
    """对照实验"""
    reward= train(cfg, env, agent)



