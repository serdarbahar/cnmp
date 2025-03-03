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
from sklearn.decomposition import PCA


def predict_inverse(model, idx, time_len, condition_points, d_x, d_y1, d_y2, data):

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
        obs = torch.cat((obs,obs), dim=-1) # (n, 2*d_x + 2*d_y1)
        
        #adding context
        if idx % 2 == 0:
            obs = torch.cat((obs, torch.zeros(num_conditions, 1)), dim=-1)
        else:
            obs = torch.cat((obs, torch.ones(num_conditions, 1)), dim=-1) # (n, 2*d_x + 2*d_y1 + 1)


        output = model(obs, T, p=1)
        mean1, std1, mean2, std2 = output.chunk(4, dim=-1)
        std2 = np.log(1+np.exp(std2))
        means = mean2
        stds = std2
    
    return means[:,0], stds[:,0]

class DualEncoderDecoder(nn.Module):
    def __init__(self, d_x, d_y1, d_y2):
        super(DualEncoderDecoder, self).__init__()

        self.d_x = d_x
        self.d_y1 = d_y1
        self.d_y2 = d_y2
        
        self.encoder1 = nn.Sequential(
            nn.Linear(d_x + d_y1 + 1, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 128), 
        )

        self.encoder2 = nn.Sequential(
            nn.Linear(d_x + d_y2 + 1, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 128),     
        )

        self.decoder1 = nn.Sequential(
            nn.Linear(d_x + 128, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 2*d_y1)
        )

        self.decoder2 = nn.Sequential(
            nn.Linear(d_x + 128, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 2*d_y2)
        )

    def forward(self, obs, x_tar, p=0):

        obs1, obs2 = obs[:, :self.d_x + self.d_y1], obs[:, self.d_x + self.d_y1:2*self.d_x + self.d_y1 + self.d_y2]
        context = obs[:, -1].reshape(-1,1)
        obs1 = torch.cat((obs1, context), dim=-1)
        obs2 = torch.cat((obs2, context), dim=-1)

        r1 = self.encoder1(obs1)  # (n,128)
        r2 = self.encoder2(obs2)  # (n,128)

        a1 = torch.mean(r1, dim=0)
        a1 = a1.repeat(x_tar.shape[0], 1)
        a2 = torch.mean(r2, dim=0)
        a2 = a2.repeat(x_tar.shape[0], 1)
        
        latent = torch.zeros(0)
        if p==0:
            p1 = torch.rand(1).double()  # (1,)
            p2 = torch.rand(1).double()
            p1 = p1 / (p1 + p2)
            latent = a1 * p1 + a2 * (1-p1)  # (1,128)  
        else:
            latent = a1

        concat = torch.cat((latent, x_tar), dim=-1) # (1, d_x + 128)
        output1 = self.decoder1(concat)  # (2*d_y1,)
        output2 = self.decoder2(concat)  # (2*d_y2,)

        return torch.cat((output1, output2), dim=-1) # (2*d_y1 + 2*d_y2,)
    
def log_prob_loss(output, targets, d_y1):
    output1, output2 = output[:, :2*d_y1], output[:, 2*d_y1:]
    mean1, std1 = output1.chunk(2, dim=-1)
    mean2, std2 = output2.chunk(2, dim=-1)
    std1 = F.softplus(std1)
    std2 = F.softplus(std2)
    dist1 = D.Independent(D.Normal(loc=mean1, scale=std1), 1)
    dist2 = D.Independent(D.Normal(loc=mean2, scale=std2), 1)
    #return -torch.mean((dist1.log_prob(targets[0]))) - torch.mean((dist2.log_prob(targets[1])))
    return -torch.mean((dist2.log_prob(targets[1])))

def get_training_sample(validation_indices, X1, Y1, X2, Y2, OBS_MAX, d_N, d_x, d_y1, d_y2, time_len):
    n = np.random.randint(0, OBS_MAX) + 1  # random number of obs. points
    
    d = np.random.randint(0, d_N) # random trajectory
    while d in validation_indices:
        d = np.random.randint(0, d_N)
    
    observations = np.zeros((n, 2 * d_x + d_y1 + d_y2))
    target_X = np.zeros((1, d_x))
    target_Y1 = np.zeros((1, d_y1))
    target_Y2 = np.zeros((1, d_y2))

    perm = np.random.permutation(time_len)
    observations[:, :d_x] = X1[:1, perm[:n]]
    observations[:, d_x:d_x+d_y1] = Y1[d, perm[:n]]
    observations[:,d_x+d_y1:2*d_x+d_y1] = X2[:1, perm[n:2*n]]
    observations[:,2*d_x+d_y1:] = Y2[d, perm[n:2*n]]

    if d % 2 == 0:
        zeros = np.zeros((n,1))
        observations = np.concat((observations,zeros), axis=-1)
    else:
        ones = np.ones((n,1))
        observations = np.concat((observations, ones), axis=-1)

    # observations: (n, (x1,y1,x2,y2,context))
        
    perm = np.random.permutation(time_len)
    target_X = X1[:1, perm[0]]
    target_Y1 = Y1[d, perm[0]].reshape(1,-1)
    target_Y2 = Y2[d, perm[0]].reshape(1,-1)

    return torch.from_numpy(observations), torch.from_numpy(target_X), [torch.from_numpy(target_Y1), torch.from_numpy(target_Y2)]

def plot_latent_space(model, observations_f, observations_i):
    with torch.no_grad():
        l_f = model.encoder1(observations_f) # condition points is (n, d_x + d_y), l is (n, 128)
        l_i = model.encoder2(observations_i)
    l_f = np.array(l_f)
    l_i = np.array(l_i)
    l = np.concat((l_f,l_i), axis=0)

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(l)
    print(pca_result.shape)
    return pca_result



        