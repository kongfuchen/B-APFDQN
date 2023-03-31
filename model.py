#!/usr/bin/env python
# coding=utf-8

import torch.nn as nn
# import torch.nn.functional as F
import torch
torch.manual_seed(0)
class MLP(nn.Module):
    def __init__(self,input_dim,output_dim,hidden_dim):
        """ 初始化q网络，为全连接网络
            input_dim: 输入的feature即环境的state数目
            output_dim: 输出的action总个数
        """
        super(MLP, self).__init__()
        self.linear = nn.Linear(input_dim,hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(hidden_dim,hidden_dim) # 隐藏层
        self.fc1_relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # 隐藏层
        self.fc2_relu = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(hidden_dim, output_dim)  # 隐藏层
        self.softmax = nn.Softmax(dim=1)

    def forward(self, y):
        # 各层对应的激活函数
        y = self.relu(self.linear(y))
        y = self.fc1_relu(self.fc1(y))
        y = self.fc2_relu(self.fc2(y))
        y = self.fc3(y)
        return y,self.softmax(y)



class Dueling(nn.Module):
    def __init__(self,input_dim,output_dim,hidden_dim):
        """ 初始化q网络，为全连接网络
            input_dim: 输入的feature即环境的state数目
            output_dim: 输出的action总个数
        """
        super(Dueling, self).__init__()
        self.linear = nn.Linear(input_dim,hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(hidden_dim,hidden_dim) # 隐藏层
        self.fc1_relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # 隐藏层
        self.fc2_relu = nn.ReLU(inplace=True)
        self.A = nn.Linear(hidden_dim, output_dim)  # 隐藏层
        self.V=nn.Linear(hidden_dim,1)

    def forward(self, y):
        # 各层对应的激活函数
        y = self.relu(self.linear(y))
        y = self.fc1_relu(self.fc1(y))
        y = self.fc2_relu(self.fc2(y))
        A = self.A(y)
        v = self.V(y)

        A1=A.mean(1).unsqueeze(1).expand(y.size(0), 8)
        V=torch.cat((v,v,v,v,v,v,v,v),1)
        Q=V+(A-A1)
        return Q



