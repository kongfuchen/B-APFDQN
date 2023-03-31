import numpy as np
import math
import random
# import matplotlib.pyplot as plt

# a = 1
# w = 1
# b = 0.001
# c = 0
# pi = 3.14
# m = 1/4
# n = 6000
# x = np.arange(0, 11, 1)
# y = np.arange(0, 11, 1)
# x, y = np.meshgrid(x, y)
# shape_ = len(x)
"""
1、去掉路径中的环
2、确定起点终点与势场
"""
random.seed(1)
ita=20
zita=5

obs=[[[0, 0], [3, 3]], [[3, 3], [8, 8]], [[5, 5], [7, 7]], [[7, 7], [2, 2]], [[2, 2], [0, 0]], [[3, 3], [1, 1]], [[9, 9], [8, 8]], [[2, 2], [9, 9]], [[2, 2], [3, 3]], [[5, 5], [9, 9]], [[10, 10], [1, 1]], [[1, 1], [1, 1]], [[2, 2], [6, 6]], [[9, 9], [7, 7]], [[4, 4], [1, 1]], [[10, 10], [6, 6]], [[8, 8], [9, 9]], [[2, 2], [8, 8]]]

class Env(object):

    def __init__(self):
        self.x = 0
        self.y = 0
        self.actions = ['left','left_forward', 'forward','right_forward', 'right', 'right_back', 'back', 'left_back']  # 索引[0,1]
        self.towards = [[0,1],  [1,1],          [1,0],    [1,-1],          [0,-1],  [-1,-1],     [-1,0],  [-1,1]]
        self.x_g = []
        self.y_g = []
        self.best_x = []
        self.best_y = []
        self.target_x=10
        self.target_y=10
        self.obs_x=[]
        self.obs_y=[]
        self.alpha=1
        self.beta=0
        self.gamma=3
        self.rate=0.4
        self.f_Total=self.F()

    def F(self):
        Uatt = np.empty((self.target_x+1, self.target_y+1))
        for i in range(0, self.target_x+1):
            for j in range(0, self.target_y+1):
                Uatt[i][j] = zita * math.sqrt((self.target_x+1 - i) ** 2 + (self.target_y+1 - j) ** 2)
                if self.if_in_obs(i, j):
                    Uatt[i][j]=Uatt[i][j]+math.inf
        return Uatt
    def compute_angel(self,a0,a1):
        towards0 = self.towards[a0][:]
        towards1 = self.towards[a1][:]
        cosangel = (towards0[0] * towards1[0] + towards0[1] * towards1[1]) / (
                    math.sqrt(towards0[0] ** 2 + towards0[1] ** 2) * math.sqrt(towards1[0] ** 2 + towards1[1] ** 2))
        return cosangel


    def distance(self,_x, _y, x_, y_):
        return math.sqrt((_x - x_) ** 2 + (_y - y_) ** 2)

    def if_in_obs(self, x, y):
        flag = []
        for i in range(len(obs)):
            if obs[i][0][0] <= x <= obs[i][0][1] and obs[i][1][0] <= y <= obs[i][1][1] or x<0 or x>self.target_x or y<0 or y>self.target_y:
                flag.append(True)
            else:
                flag.append(False)
        if True in flag:
            return True
        else:
            return False


    def compute_label(self,a):
        x=a[0]
        y=a[1]
        state = np.empty((3, 3), dtype=float)
        if x-1<0 or y+1>self.target_y:
            state[0][0]=math.inf
        else:
            state[0][0] = self.f_Total[x - 1][y + 1]
        if y + 1>self.target_y:
            state[0][1]=math.inf
        else:
            state[0][1] = self.f_Total[x][y + 1]
        if x + 1>self.target_x or y + 1>self.target_y:
            state[0][2]=math.inf
        else:
            state[0][2] = self.f_Total[x + 1][y + 1]
        if x - 1<0:
            state[1][0]=math.inf
        else:
            state[1][0] = self.f_Total[x - 1][y]
        state[1][1] = self.f_Total[x][y]
        if x+1>self.target_x:
            state[1][2] =math.inf
        else:
            state[1][2] = self.f_Total[x + 1][y]
        if x - 1<0 or y - 1<0:
            state[2][0] =math.inf
        else:
            state[2][0] = self.f_Total[x - 1][y - 1]
        if y - 1<0:
            state[2][1] =math.inf
        else:
            state[2][1] = self.f_Total[x][y - 1]
        if x + 1>self.target_x or y - 1<0:
            state[2][2]=math.inf
        else:
            state[2][2]=self.f_Total[x + 1][y - 1]
        index = np.argmin(state)
        label = -1
        if index == 0:
            label = self.actions.index('left_back')
        elif index == 1:
            label = self.actions.index('left')
        elif index == 2:
            label = self.actions.index('left_forward')
        elif index == 3:
            label = self.actions.index('back')
        elif index == 4:
            label = self.actions.index('forward')
        elif index == 5:
            label = self.actions.index('forward')
        elif index == 6:
            label = self.actions.index('right_back')
        elif index == 7:
            label = self.actions.index('right')
        elif index == 8:
            label = self.actions.index('right_forward')
        return label



    def _step(self, action, x, y):
        if self.actions[action] == 'left':
            self.y = y+1
        elif self.actions[action] == 'right' :
            self.y = y-1
        elif self.actions[action] == 'forward':
            self.x = x+1
        elif self.actions[action] == 'back':
            self.x = x-1
        elif self.actions[action] == 'left_forward':
            self.y = y + 1
            self.x = x + 1
        elif self.actions[action] == 'left_back':
            self.y = y + 1
            self.x = x - 1
        elif self.actions[action] == 'right_forward':
            self.y = y - 1
            self.x = x + 1
        elif self.actions[action] == 'right_back':
            self.y = y - 1
            self.x = x - 1

    def reset(self):
        """"
        重置环境，返回状态0
        """
        self.x=0
        self.y=0
        self.x_g=[]
        self.y_g=[]
        return [self.x,self.y]




    def step(self, action):
        is_win =False
        x0=self.x
        y0=self.y
        label = self.compute_label([self.x, self.y])
        self._step(action, self.x, self.y)
        if self.if_in_obs(self.x, self.y):
            self.reset()
            reward = -10
            next_label = self.compute_label([self.x, self.y])
            state = [self.x,self.y]
        else:
            next_label = self.compute_label([self.x, self.y])
            state = [self.x, self.y]
            ds_1 = self.distance(x0, y0, self.target_x, self.target_y)
            ds = self.distance(self.x, self.y, self.target_x, self.target_y)

            reward = self.alpha * (ds_1 - ds)-1
        if [self.x, self.y] == [self.target_x,self.target_y]:
            is_win=True
        return state,reward,is_win,label,next_label






