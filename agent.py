#!/usr/bin/env python
# coding=utf-8

import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
import numpy as np
from memory import ReplayBuffer
from model import MLP
import torchvision.transforms as T
from PIL import Image
torch.manual_seed(0)

resize = T.Compose([T.ToPILImage(),
                    T.Resize(3, interpolation=Image.CUBIC),
                    T.ToTensor()])


class DQN:
    def __init__(self, state_dim,action_dim, cfg):
        self.action_dim = action_dim
        self.device = cfg.device
        self.gamma = cfg.gamma
        self.batch_size=cfg.batch_size
        self.policy_net = MLP(state_dim,action_dim,
                              hidden_dim=cfg.hidden_dim).to(self.device)
        self.target_net = MLP(state_dim,action_dim,
                              hidden_dim=cfg.hidden_dim).to(self.device)

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.lr)
        self.memory = ReplayBuffer(cfg.memory_capacity)
        self.frame_idx=0

        self.alpha=0.9  # criterion
        self.beta=0.5  # mseLoss

        self.frame_idx = 0
        self.epsilon = lambda frame_idx: cfg.epsilon_end + \
            (cfg.epsilon_start - cfg.epsilon_end) * \
            math.exp(-1. * frame_idx / cfg.epsilon_decay)     # SA-eplison-greedy


    def action_selection(self, state_linear, env):
        """
        Select an action
        """
        self.frame_idx += 1
        if random.random() > self.epsilon(self.frame_idx):
            with torch.no_grad():
                a1 = env.compute_label(state_linear)     # Select action according to APF
                state_linear = torch.tensor(
                    [state_linear], device=self.device, dtype=torch.float32)
                q_value, label = self.policy_net(state_linear)
                a0 = int(torch.argmax(label))     # Action according to Q-values
                cosangel=env.compute_angel(a0,a1)
                if math.cos(1*math.pi/4) <= cosangel <= math.cos(math.pi*0):     # Whether it is within the allowable threshold of angular error
                    action=a0
                else:
                    action=a1

        else:
            a=[0,1,2,3,4,5,6,7]
            action = random.choice(a)
        return action



    def update_net(self, env):
        if len(self.memory) < self.batch_size:
            return
        state_linear_batch,action_batch, reward_batch, next_state_linear_batch, done_batch, label_batch,next_label_batch = self.memory.sample(
            self.batch_size)
        state_linear_batch = torch.tensor(
            state_linear_batch, device=self.device, dtype=torch.float)
        action_batch = torch.tensor(action_batch, device=self.device).unsqueeze(
            1)
        reward_batch = torch.tensor(
            reward_batch, device=self.device, dtype=torch.float)
        next_state_linear_batch = torch.tensor(
            next_state_linear_batch, device=self.device, dtype=torch.float)
        done_batch = torch.tensor(np.float32(
            done_batch), device=self.device)
        label_batch = torch.tensor(
            label_batch, device=self.device)
        next_label_batch = torch.tensor(
            next_label_batch, device=self.device)
        q_value, label_qvalue = self.policy_net(state_linear_batch)
        q_values = q_value.gather(dim=1, index=action_batch)
        next_q_value, label_next_q_value = self.target_net(next_state_linear_batch)

        next_a1=[]
        next_a0 = (next_label_batch).cpu().numpy()
        for i in range(len(label_next_q_value)):
            next_a1.append(int(torch.argmax(label_next_q_value[i])))

        next=next_q_value.detach().cpu().numpy()

        next_q_values=[]

        for i in range(len(next)):
            if 0<=env.compute_angel(next_a0[i],next_a1[i])<=1:
                next_q_values.append(next[i][next_a1[i]])
            else:
                next_q_values.append(next[i][next_a0[i]])

        next_q_values = torch.tensor(
            next_q_values, device=self.device, dtype=torch.float)

        index=[]
        a1 = []
        a0 = (label_batch).cpu().numpy()

        for i in range(len(label_qvalue)):
            a1.append(int(torch.argmax(label_qvalue[i])))

        for i in range(len(label_qvalue)):
            if 0 <= env.compute_angel(a0[i], a1[i]) <= 1:
                continue
            else:
                index.append(i)
        expected_q_values = reward_batch + \
                            self.gamma * next_q_values * (1 - done_batch)

        mseLoss = nn.MSELoss()
        criterion = nn.CrossEntropyLoss()

        if len(index)>2:
            label_batch0=[]
            label_qvalue0=torch.cat((label_qvalue[index[0]].unsqueeze(0),label_qvalue[index[1]].unsqueeze(0)),0)
            label_batch0.append(a0[index[0]])
            label_batch0.append(a0[index[1]])
            for i in range(2,len(index)):
                label_qvalue0 = torch.cat((label_qvalue0,label_qvalue[index[i]].unsqueeze(0)),0)
                label_batch0.append(a0[index[i]])
            label_batch0=torch.tensor(label_batch0, device=self.device)
            self.loss = self.alpha*criterion(label_qvalue0, label_batch0) + self.beta*mseLoss(q_values, expected_q_values.unsqueeze(1))
        elif len(index)==2:
            label_batch0=[]
            label_qvalue0 = torch.cat((label_qvalue[index[0]].unsqueeze(0), label_qvalue[index[1]].unsqueeze(0)), 0)
            label_batch0.append(a0[index[0]])
            label_batch0.append(a0[index[1]])
            label_batch0=torch.tensor(label_batch0, device=self.device)
            self.loss = self.alpha*criterion(label_qvalue0, label_batch0) + self.beta*mseLoss(q_values, expected_q_values.unsqueeze(1))
        elif len(index)==1:
            label_batch0=[]
            label_qvalue0=label_qvalue[index[0]].unsqueeze(0)
            label_batch0.append(a0[index[0]])
            label_batch0=torch.tensor(label_batch0, device=self.device)
            self.loss = self.alpha*criterion(label_qvalue0, label_batch0) + self.beta*mseLoss(q_values, expected_q_values.unsqueeze(1))
        else:
            self.loss = self.beta*mseLoss(q_values, expected_q_values.unsqueeze(1))



        self.optimizer.zero_grad()

        self.loss.backward()

        self.optimizer.step()

