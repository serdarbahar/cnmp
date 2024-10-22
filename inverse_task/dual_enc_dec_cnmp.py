from cProfile import label
from re import X
from turtle import color
from sympy import li
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import math


def predict_inverse(model, idx, time_len, condition_points, d_x, d_y1, d_y2, data):

    X1, X2, Y1, Y2 = data
    
    num_conditions = len(condition_points)
    obs = torch.zeros((num_conditions, d_x + d_y1))
    for i in range(num_conditions):
        x_obs = torch.from_numpy(X1[0, condition_points[i]:condition_points[i]+1])
        y_obs = torch.from_numpy(Y1[idx, condition_points[i]:condition_points[i]+1])
        obs[i] = torch.cat((x_obs, y_obs), dim=-1)
    
    obs = obs.double()

    means = torch.zeros(0)
    stds = torch.zeros(0)

    with torch.no_grad():
        for x_tar in torch.linspace(0, 1, time_len):
            x_tar = x_tar.view(1, 1)
            l = model.encoder2(obs) # (2, 128)
            a1 = torch.mean(l, dim=0, keepdim=True)  # (1, 128)
            concat = torch.cat((a1, x_tar), dim=-1)  # (1, d_x + 128)
            output = model.decoder2(concat)  # (2*d_y1,)
            mean, std = output.chunk(2, dim=-1)
            #std = np.log(1+np.exp(std))
            std = F.softplus(std) + 1e-6
            means = torch.cat((means, mean), dim=-1)
            stds = torch.cat((stds, std), dim=-1)
       
    return means, stds

class DualEncoderDecoder(nn.Module):
    def __init__(self, d_x, d_y1, d_y2):
        super(DualEncoderDecoder, self).__init__()

        self.d_x = d_x
        self.d_y1 = d_y1
        self.d_y2 = d_y2
        
        self.encoder1 = nn.Sequential(
            nn.Linear(d_x + d_y1, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, 128)
        )

        self.encoder2 = nn.Sequential(
            nn.Linear(d_x + d_y2, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, 128)
        )

        self.decoder1 = nn.Sequential(
            nn.Linear(d_x + 128, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 32), nn.ReLU(),
            nn.Linear(32, 2*d_y1)
        )

        self.decoder2 = nn.Sequential(
            nn.Linear(d_x + 128, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 32), nn.ReLU(),
            nn.Linear(32, 2*d_y2)
        )

    def forward(self, obs, x_tar, p=0):

        obs1, obs2 = obs[:, :self.d_x + self.d_y1], obs[:, self.d_x + self.d_y1:]

        n = obs1.shape[0]

        r1 = self.encoder1(obs1)  # (n,128)
        r2 = self.encoder2(obs2)  # (n,128)

        a1 = torch.mean(r1, dim=0, keepdim=True)  # (1,128)
        a2 = torch.mean(r2, dim=0, keepdim=True)
        
        if p==0:
            p1 = torch.rand(1).double()  # (1,)
            p2 = torch.rand(1).double()
            p1 = p1 / (p1 + p2)
            
            latent = a1 * p + a2 * (1-p)  # (1,128)  

        concat = torch.cat((latent, x_tar), dim=-1) # (n, d_x + 128)

        output1 = self.decoder1(concat)  # (2*d_y1,)
        output2 = self.decoder2(concat)  # (2*d_y2,)

        return torch.cat((output1, output2), dim=-1) # (2*d_y1 + 2*d_y2,)
    
def log_prob_loss(output, targets, d_y1):
    output1, output2 = output[:, :2*d_y1], output[:, 2*d_y1:]
    mean1, std1 = output1.chunk(2, dim=-1)
    mean2, std2 = output2.chunk(2, dim=-1)
    std1 = F.softplus(std1) + 1e-6
    std2 = F.softplus(std2)
    dist1 = D.Independent(D.Normal(loc=mean1, scale=std1), 1)
    dist2 = D.Independent(D.Normal(loc=mean2, scale=std2), 1)
    return -torch.mean((dist1.log_prob(targets[0]))+(dist2.log_prob(targets[1])))

def get_training_sample(X1, Y1, X2, Y2, OBS_MAX, d_N, d_x, d_y1, d_y2, time_len):
    n = np.random.randint(0, OBS_MAX) + 1  # random number of obs. points
    d = np.random.randint(0, d_N) # random trajectory

    observations = np.zeros((n, 2 * d_x + d_y1 + d_y2))
    target_X = np.zeros((1, d_x))
    target_Y1 = np.zeros((1, d_y1))
    target_Y2 = np.zeros((1, d_y2))

    perm = np.random.permutation(time_len)
    observations[:, :d_x] = X1[:1, perm[:n]]
    observations[:, d_x:d_x+d_y1] = Y1[d, perm[:n]]
    observations[:,d_x+d_y1:2*d_x+d_y1] = X2[:1, perm[n:2*n]]
    observations[:,2*d_x+d_y1:] = Y2[d, perm[n:2*n]]

    perm = np.random.permutation(time_len)
    target_X = X1[:1, perm[0]]
    target_Y1 = Y1[d, perm[0]].reshape(1,-1)
    target_Y2 = Y2[d, perm[0]].reshape(1,-1)

    return torch.from_numpy(observations), torch.from_numpy(target_X), [torch.from_numpy(target_Y1), torch.from_numpy(target_Y2)]