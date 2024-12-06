from cProfile import label
from re import X
from sre_constants import IN
from turtle import color
from matplotlib.pylab import norm
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

from sklearn.metrics.pairwise import cosine_similarity
import torch.masked

class DualEncoderDecoder(nn.Module):
    def __init__(self, d_x, d_y1, d_y2):
        super(DualEncoderDecoder, self).__init__()

        self.d_x = d_x
        self.d_y1 = d_y1
        self.d_y2 = d_y2
        
        self.encoder1 = nn.Sequential(
            nn.Linear(d_x + d_y1, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 128), 
        )

        self.encoder2 = nn.Sequential(
            nn.Linear(d_x + d_y2, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
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

    def forward(self, obs, mask, x_tar, p=0):
        # obs: (num_traj, max_obs_num, 2*d_x + d_y1 + d_y2) 
        # mask: (num_traj, max_obs_num, 1)
        # x_tar: (num_traj, num_tar, d_x)

        mask_forward, mask_inverse = mask[0], mask[1] # (num_traj, max_obs_num, max_obs_num)

        obs_f = obs[:, :, :self.d_x + self.d_y1]  # (num_traj, max_obs_num, d_x + d_y1)
        obs_i = obs[:, :, self.d_x + self.d_y1:]  # (num_traj, max_obs_num, d_x + d_y2)


        r1 = self.encoder1(obs_f)  # (num_traj, max_obs_num, 128)
        r2 = self.encoder2(obs_i)  # (num_traj, max_obs_num, 128)
        
        masked_r1 = torch.bmm(mask_forward, r1) # (num_traj, max_obs_num, 128)
        masked_r2 = torch.bmm(mask_inverse, r2)

        sum_masked_r1 = torch.sum(masked_r1, dim=1) # (num_traj, 128)
        sum_masked_r2 = torch.sum(masked_r2, dim=1) # (num_traj, 128)

        L_F = sum_masked_r1 / torch.sum(mask_forward, dim=[1,2]).reshape(-1,1)
        L_I = sum_masked_r2 / torch.sum(mask_inverse, dim=[1,2]).reshape(-1,1)

        L_F = L_F.unsqueeze(1).expand(-1, x_tar.shape[1], -1) # (num_traj, num_tar, 128)
        L_I = L_I.unsqueeze(1).expand(-1, x_tar.shape[1], -1) # (num_traj, num_tar, 128)

        latent = torch.zeros(0)
        if p==0:
            p1 = torch.rand(1)
            p2 = torch.rand(1)
            p1 = p1 / (p1 + p2)
            latent = L_F * p1 + L_I * (1-p1)  # (num_traj, num_tar, 128)
        else:
            latent = L_F # (1, num_tar, 128) , used for validation pass

        concat = torch.cat((latent, x_tar), dim=-1)  # (num_traj, num_tar, 128 + d_x) 

        output1 = self.decoder1(concat)  # (num_traj, num_tar, 2*d_y1)
        output2 = self.decoder2(concat)  # (num_traj, num_tar, 2*d_y2)

        # (num_traj, num_tar, 2*d_y1 + 2*d_y2)
        return torch.cat((output1, output2), dim=-1), L_F, L_I
    

def get_training_sample(validation_indices, X1, Y1, X2, Y2, OBS_MAX, d_N, d_x, d_y1, d_y2, time_len):

    num_traj = torch.randint(5, 10, (1,)) # random number of trajectories

    # random traj indices, indices should not be in validation_indices
    traj_indices = []

    for i in range(num_traj):
        d = torch.randint(d_N, (1,))
        while d in validation_indices in traj_indices:
            d = torch.randint(d_N, (1,))
        traj_indices.append(d)

    torch.tensor(traj_indices)

    obs_num_list = torch.randint(0, OBS_MAX, (2*num_traj,)) + 1  # random number of obs. points
    
    max_obs_num = torch.max(obs_num_list)

    observations = torch.zeros((num_traj, max_obs_num, 2*d_x + d_y1 + d_y2))

    mask_forward = torch.zeros((num_traj, max_obs_num, max_obs_num))
    mask_inverse = torch.zeros((num_traj, max_obs_num, max_obs_num))

    target_X = torch.zeros((num_traj, 1, d_x))
    target_Y1 = torch.zeros((num_traj, 1, d_y1))
    target_Y2 = torch.zeros((num_traj, 1, d_y2))

    T = torch.linspace(0,1,time_len).reshape(-1)

    for i in range(num_traj):
        traj_index = int(traj_indices[i])
        obs_num_f = int(obs_num_list[i])
        obs_num_i = int(obs_num_list[num_traj + i])

        obs_indices_f = torch.multinomial(T, obs_num_f, replacement=False)
        obs_indices_i = torch.multinomial(T, obs_num_i, replacement=False)

        for j in range(obs_num_f):
            observations[i][j][0] = X1[0][obs_indices_f[j]]
            observations[i][j][d_x] = Y1[traj_index][obs_indices_f[j]]
            mask_forward[i][j][j] = 1

        for j in range(obs_num_i):
            observations[i][j][d_x + d_y1] = X2[0][obs_indices_i[j]]
            observations[i][j][2*d_x + d_y1] = Y2[traj_index][obs_indices_i[j]]
            mask_inverse[i][j][j] = 1
        

        target_index = torch.multinomial(T, 1)
        target_X[i] = X1[0][target_index]
        target_Y1[i] = Y1[traj_index][target_index]
        target_Y2[i] = Y2[traj_index][target_index]

    return observations, [mask_forward, mask_inverse], target_X, target_Y1, target_Y2
    
def loss(output, target_f, target_i, d_y1, d_y2, L_F, L_I):

    # L_F/I (num_traj, 128)

    mse_of_pairs = compute_mse_of_pairs(L_F, L_I) # scalar
    distance_trajwise = compute_distance_trajwise(L_F, L_I) #scalar
    log_prob = log_prob_loss(output, target_f, target_i, d_y1, d_y2) # scalar
    norm = compute_norm(L_F, L_I) # scalar

    lambda1, lambda2, lambda3, lambda4 = 1, 1, 10, 1

    return lambda1 * log_prob + lambda2 * mse_of_pairs + lambda3 * -1 * distance_trajwise + lambda4 * norm

def predict_inverse(model, idx, time_len, condition_points, d_x, d_y1, d_y2, data):

    X1, X2, Y1, Y2 = data
    
    num_conditions = len(condition_points)
    obs = torch.zeros((1, num_conditions, d_x + d_y1))
    mask = torch.eye(num_conditions).repeat(1,1,1)
    mask = [mask, mask]
    for condition in condition_points:
        x_obs = torch.tensor(condition[0]).reshape(1,1)
        y_obs = condition[1].reshape(1,1)
        obs[0][condition_points.index(condition)] = torch.cat((x_obs, y_obs), dim=-1) 

    means = torch.zeros(0)
    stds = torch.zeros(0)

    with torch.no_grad():
        T = torch.linspace(0,1,time_len).reshape(1, time_len, -1)
        obs = torch.cat((obs,obs), dim=-1)
        output, _, __ = model(obs, mask, T, p=1)
        _, __, mean2, std2 = output.chunk(4, dim=-1)
        std2 = np.log(1+np.exp(std2))
        means = mean2
        stds = std2
    
    return means[0,:,0], stds[0,:,0]


############################################################################################################

def log_prob_loss(output, targets_f, targets_i, d_y1, d_y2): # output (num_traj, num_tar, 2*d_y1 + 2*d_y2), targets (num_traj, num_tar, d_y1 + d_y2)
    means_f, stds_f, means_i, stds_i = output[:, :, :d_y1], output[:, :, d_y1:2*d_y1], output[:, :, 2*d_y1:2*d_y1+d_y2], output[:, :, 2*d_y1+d_y2:]

    stds_f = F.softplus(stds_f)
    stds_i = F.softplus(stds_i) # (num_traj, num_tar, d_y1)

    normal_f = D.MultivariateNormal(means_f, torch.diag_embed(stds_f))
    normal_i = D.MultivariateNormal(means_i, torch.diag_embed(stds_i))
    
    log_prob_f = normal_f.log_prob(targets_f) # (num_traj, num_tar)
    log_prob_i = normal_i.log_prob(targets_i) # (num_traj, num_tar)

    total_loss = torch.mean(log_prob_f + log_prob_i, dim=1) # (num_traj)

    return - total_loss.mean() # scalar  

def compute_mse_of_pairs(L_F, L_I): #L_F/I (num_traj, 128)

    mse = 0
    for i in range(len(L_F)):
        mse += F.pairwise_distance(L_F[i], L_I[i], p=2)
    return mse

def compute_distance_trajwise(L_F, L_I):

    trajwise_dist = 0
    # forward
    trajwise_dist += F.pairwise_distance(L_F[0], L_F[1], p=2)
    
    trajwise_dist += F.pairwise_distance(L_I[0], L_I[1], p=2)

    return trajwise_dist / 2

def compute_norm(L_F, L_I):
    norm = 0
    for i in range(len(L_F)):
        norm += torch.norm(L_F[i]) + torch.norm(L_I[i])
    return norm

def plot_latent_space(model, observations_f, observations_i):
    with torch.no_grad():
        l_f = model.encoder1(observations_f) # condition points is (n, d_x + d_y), l is (n, 128)
        l_i = model.encoder2(observations_i)
    l_f = np.array(l_f)
    l_i = np.array(l_i)
    l = np.concat((l_f,l_i), axis=0)

    print(l.shape)

    l = l.squeeze(1)

    print(l.shape)

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(l)
    return pca_result