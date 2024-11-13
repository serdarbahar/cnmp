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


def predict_inverse(model, idx, time_len, condition_points, d_x, d_y1, data):

    X1, X2, Y1, Y2 = data
    
    num_conditions = len(condition_points)
    obs = torch.zeros((num_conditions, d_x + d_y1))
    for condition in condition_points:
        x_obs = torch.tensor(condition[0]).view(1, 1)
        y_obs = torch.tensor(condition[1]).view(1, 1)
        obs[condition_points.index(condition)] = torch.cat((x_obs, y_obs), dim=-1)
    
    obs = obs.double()

    means = torch.zeros(0)
    stds = torch.zeros(0)

    with torch.no_grad():
        T = torch.linspace(0,1,time_len).reshape(-1,1)
        output = model(obs, T)
        mean1, std1 = output[:,:d_y1], output[:,d_y1:]
        std1 = np.log(1+np.exp(std1))
        means = mean1
        stds = std1
    
    return means[:,0], stds[:,0]

class DualEncoderDecoder(nn.Module):
    def __init__(self, d_x, d_y1):
        super(DualEncoderDecoder, self).__init__()

        self.d_x = d_x
        self.d_y1 = d_y1
        
        self.encoder1 = nn.Sequential(
            nn.Linear(d_x + d_y1, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 128), 
            #nn.Linear(d_x + d_y1, 32), nn.ReLU(),
            #nn.Linear(32, 64), nn.ReLU(),
            #nn.Linear(64, 64), nn.ReLU(),
            #nn.Linear(64, 128), nn.ReLU(),
            #nn.Linear(128, 128),
        )

        self.decoder1 = nn.Sequential(
            nn.Linear(d_x + 128, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 2*d_y1)
            #nn.Linear(d_x + 128, 128), nn.ReLU(),
            #nn.Linear(128, 64), nn.ReLU(),
            #nn.Linear(64, 32), nn.ReLU(),
            #nn.Linear(32, 2*d_y1)
        )


    def forward(self, obs, x_tar):

        obs1 = obs # (n,d_x+d_y1)

        r1 = self.encoder1(obs1)  # (n,128)

        a1 = torch.mean(r1, dim=0)
        a1 = a1.repeat(x_tar.shape[0], 1)
        
        latent = a1

        concat = torch.cat((latent, x_tar), dim=-1) # (1, d_x + 128)
        output1 = self.decoder1(concat)  # (2*d_y1,)

        return output1
    
def log_prob_loss(output, targets, d_y1):
    mean1, std1 = output.chunk(2, dim=-1)
    std1 = F.softplus(std1)
    dist1 = D.Independent(D.Normal(loc=mean1, scale=std1), 1)
    #return -torch.mean((dist1.log_prob(targets[0]))) - torch.mean((dist2.log_prob(targets[1])))
    return -torch.mean((dist1.log_prob(targets[0])))

def get_training_sample(validation_indices, X1, Y1, X2, OBS_MAX, d_N, d_x, d_y1, time_len):
    n = np.random.randint(0, OBS_MAX) + 1  # random number of obs. points
    
    d = np.random.randint(0, d_N) # random trajectory
    while d in validation_indices:
        d = np.random.randint(0, d_N)
    
    observations = np.zeros((n, d_x + d_y1))
    target_X = np.zeros((1, d_x))
    target_Y2 = np.zeros((1, d_y1))

    perm = np.random.permutation(time_len)
    observations[:, :d_x] = X1[:1, perm[:n]]
    observations[:, d_x:d_x+d_y1] = Y1[d, perm[:n]]

    perm = np.random.permutation(time_len)
    target_X = X1[:1, perm[0]]
    target_Y2 = Y1[d, perm[0]].reshape(1,-1)
    return torch.from_numpy(observations), torch.from_numpy(target_X), [torch.from_numpy(target_Y2)]