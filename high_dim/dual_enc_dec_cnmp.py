from ast import Is
from cProfile import label
from re import I, X
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
from sklearn.manifold import TSNE

from sklearn.metrics.pairwise import cosine_similarity
import torch.masked

class DualEncoderDecoder(nn.Module):
    def __init__(self, d_x, d_y1, d_y2):
        super(DualEncoderDecoder, self).__init__()

        self.d_x = d_x
        self.d_y1 = d_y1
        self.d_y2 = d_y2
        self.context_size = 4
        
        self.encoder1 = nn.Sequential(
            nn.Linear(d_x + d_y1 - self.context_size + 2, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 256)
        )

        self.encoder2 = nn.Sequential(
            nn.Linear(d_x + d_y2 - self.context_size + 2, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 256)
        )

        self.encoder3 = nn.Sequential(
            nn.Linear(4, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 2),
        )

        self.decoder1 = nn.Sequential(
            nn.Linear(d_x + 256, 256), nn.ReLU(),
            nn.Linear(256,512), nn.ReLU(),
            nn.Linear(512,512), nn.ReLU(),
            nn.Linear(512,512), nn.ReLU(),
            nn.Linear(512,512), nn.ReLU(),
            nn.Linear(512,256), nn.ReLU(),
            nn.Linear(256,2*7)
        )

        self.decoder2 = nn.Sequential(
            nn.Linear(d_x + 256, 256), nn.ReLU(),
            nn.Linear(256,512), nn.ReLU(),
            nn.Linear(512,512), nn.ReLU(),
            nn.Linear(512,512), nn.ReLU(),
            nn.Linear(512,512), nn.ReLU(),
            nn.Linear(512,256), nn.ReLU(),
            nn.Linear(256,2*7)
        )

    def forward(self, obs, mask, x_tar, p=0):
        # obs: (num_traj, max_obs_num, 2*d_x + d_y1 + d_y2) 
        # mask: (num_traj, max_obs_num, 1)
        # x_tar: (num_traj, num_tar, d_x)

        mask_forward, mask_inverse = mask[0], mask[1] # (num_traj, max_obs_num, max_obs_num)

        obs_f = obs[:, :, :self.d_x + self.d_y1]  # (num_traj, max_obs_num, d_x + d_y1)
        obs_i = obs[:, :, self.d_x + self.d_y1:]  # (num_traj, max_obs_num, d_x + d_y2)

        context = obs_f[:, :, self.d_x + 7: self.d_x + 11] # (num_traj, max_obs_num, 4)
        context = self.encoder3(context) # (num_traj, max_obs_num, 128)
        obs_f = torch.cat((obs_f[:,:,:self.d_x + self.d_y1 - self.context_size], context), dim=-1) # (num_traj, max_obs_num, 128 + d_x + d_y1)
        obs_i = torch.cat((obs_i[:,:,:self.d_x + self.d_y2 - self.context_size], context), dim=-1) # (num_traj, max_obs_num, 128 + d_x + d_y2)
        
        r1 = self.encoder1(obs_f)  # (num_traj, max_obs_num, 128)
        r2 = self.encoder2(obs_i)  # (num_traj, max_obs_num, 128)
        
        masked_r1 = torch.bmm(mask_forward, r1) # (num_traj, max_obs_num, 128)
        masked_r2 = torch.bmm(mask_inverse, r2)

        sum_masked_r1 = torch.sum(masked_r1, dim=1) # (num_traj, 128)
        sum_masked_r2 = torch.sum(masked_r2, dim=1) # (num_traj, 128)

        L_F = sum_masked_r1 / (torch.sum(mask_forward, dim=[1,2]).reshape(-1,1) + 1e-10)
        L_I = sum_masked_r2 / (torch.sum(mask_inverse, dim=[1,2]).reshape(-1,1) + 1e-10)

        L_F = L_F.unsqueeze(1).expand(-1, x_tar.shape[1], -1) # (num_traj, num_tar, 128)
        L_I = L_I.unsqueeze(1).expand(-1, x_tar.shape[1], -1) # (num_traj, num_tar, 128)

        latent = torch.zeros(0)
        if p==0:
            p1 = torch.rand(1)
            p2 = torch.rand(1)
            p1 = p1 / (p1 + p2)
            latent = L_F * p1 + L_I * (1-p1)  # (num_traj, num_tar, 128)
        elif p == 1:
            latent = L_F # (1, num_tar, 128) , used for validation pass
        elif p == 2:
            latent = L_I

        concat = torch.cat((latent, x_tar), dim=-1)  # (num_traj, num_tar, 128 + d_x) 

        output1 = self.decoder1(concat)  # (num_traj, num_tar, 2*d_y1)
        output2 = self.decoder2(concat)  # (num_traj, num_tar, 2*d_y2)

        # (num_traj, num_tar, 2*d_y1 + 2*d_y2)
        return torch.cat((output1, output2), dim=-1), L_F, L_I
    
def get_training_sample(validation_indices, valid_inverses, unpaired_traj, X1, Y1, X2, Y2, OBS_MAX, d_N, d_x, d_y1, d_y2, time_len):

    num_traj = torch.randint(40, 50, (1,)) # random number of trajectories

    traj_multinom = torch.ones(d_N) # multinomial distribution for trajectories
    for i in validation_indices:
        traj_multinom[i] = 0
    
    if not unpaired_traj:
        for i in range(len(valid_inverses)):
            if not valid_inverses[i]:
                traj_multinom[i] = 0
    
    traj_indices = torch.multinomial(traj_multinom, num_traj.item(), replacement=False) # random indices of trajectories

    obs_num_list = torch.randint(0, OBS_MAX, (2*num_traj,)) + 1  # random number of obs. points
    
    max_obs_num = torch.max(obs_num_list)

    observations = torch.zeros((num_traj, max_obs_num, 2*d_x + d_y1 + d_y2))

    mask_forward = torch.zeros((num_traj, max_obs_num, max_obs_num))
    mask_inverse = torch.zeros((num_traj, max_obs_num, max_obs_num))

    target_X = torch.zeros((num_traj, 1, d_x))
    target_Y1 = torch.zeros((num_traj, 1, d_y1))
    target_Y2 = torch.zeros((num_traj, 1, d_y2))

    T = torch.ones(time_len)

    for i in range(num_traj):
        traj_index = int(traj_indices[i])
        obs_num_f = int(obs_num_list[i])
        obs_num_i = int(obs_num_list[num_traj + i])

        obs_indices_f = torch.multinomial(T, obs_num_f, replacement=False)
        obs_indices_i = torch.multinomial(T, obs_num_i, replacement=False)


        for j in range(obs_num_f):
            observations[i][j][:d_x] = X1[0][obs_indices_f[j]]
            observations[i][j][d_x:d_x+d_y1] = Y1[traj_index][obs_indices_f[j]]
            mask_forward[i][j][j] = 1

        for j in range(obs_num_i):
        
            if valid_inverses[traj_index]:
                observations[i][j][d_x + d_y1:2*d_x + d_y1] = X2[0][obs_indices_i[j]]
                observations[i][j][2*d_x + d_y1:] = Y2[traj_index][obs_indices_i[j]]
                mask_inverse[i][j][j] = 1
        
        target_index = torch.multinomial(T, 1)
        target_X[i] = X1[0][target_index]
        target_Y1[i] = Y1[traj_index][target_index]
        if valid_inverses[traj_index]:
            target_Y2[i] = Y2[traj_index][target_index]
        
    return observations, [mask_forward, mask_inverse], target_X, target_Y1, target_Y2, traj_indices
    
def loss(output, target_f, target_i, d_y1, d_y2, L_F, L_I, valid_inverses, traj_indices):

    L_F, L_I = rescale_latent_representations(L_F, L_I)

    #print(F.pairwise_distance(L_F[0], L_I[0], p=2))

    mse_of_pairs = compute_mse_of_pairs(L_F, L_I, valid_inverses, traj_indices) # scalar
    distance_trajwise = compute_distance_trajwise(L_F, L_I, valid_inverses, traj_indices) #scalar
    log_prob = log_prob_loss(output, target_f, target_i, d_y1, d_y2, valid_inverses, traj_indices) # scalar
    norm = compute_norm(L_F, L_I) # scalar

    lambda1, lambda2, lambda3, lambda4 = 1, 0, 0, 0

    return lambda1 * log_prob + lambda2 * mse_of_pairs + lambda3 * torch.clamp(1-distance_trajwise, min=0) + lambda4 * norm

def predict_inverse(model, idx, time_len, condition_points, d_x, d_y1, d_y2, data):

    X1, X2, Y1, Y2 = data
    
    num_conditions = len(condition_points)
    obs = torch.zeros((1, num_conditions, d_x + d_y1))
    mask = torch.eye(num_conditions).repeat(1,1,1)
    mask = [mask, mask]
    for condition in condition_points:
        x_obs = torch.tensor(condition[0]).reshape(1,d_x)
        y_obs = condition[1].reshape(1,d_y1)
        obs[0][condition_points.index(condition)] = torch.cat((x_obs, y_obs), dim=-1) 
    
    means = torch.zeros(0)
    stds = torch.zeros(0)

    with torch.no_grad():
        T = torch.linspace(0,1,time_len).reshape(1, time_len, -1)
        obs = torch.cat((obs,obs), dim=-1)
        output, _, __ = model(obs, mask, T, p=1)
        mean1, std1, mean2, std2 = output.chunk(4, dim=-1)
        std2 = np.log(1+np.exp(std2))
        std1 = np.log(1+np.exp(std1))
        means = mean2
        stds = std2
    
    return means[0], stds[0]

def predict_forward_forward(model, idx, time_len, condition_points, d_x, d_y1, d_y2, data):

    X1, X2, Y1, Y2 = data
    
    num_conditions = len(condition_points)
    obs = torch.zeros((1, num_conditions, d_x + d_y1))
    mask = torch.eye(num_conditions).repeat(1,1,1)
    mask = [mask, mask]
    for condition in condition_points:
        x_obs = torch.tensor(condition[0]).reshape(1,d_x)
        y_obs = condition[1].reshape(1,d_y1)
        obs[0][condition_points.index(condition)] = torch.cat((x_obs, y_obs), dim=-1) 
    
    means = torch.zeros(0)
    stds = torch.zeros(0)

    with torch.no_grad():
        T = torch.linspace(0,1,time_len).reshape(1, time_len, -1)
        obs = torch.cat((obs,obs), dim=-1)
        output, _, __ = model(obs, mask, T, p=1)
        mean1, std1, mean2, std2 = output.chunk(4, dim=-1)
        std2 = np.log(1+np.exp(std2))
        std1 = np.log(1+np.exp(std1))
        means = mean1
        stds = std1
    
    return means[0], stds[0]

def predict_forward(model, idx, time_len, condition_points, d_x, d_y1, d_y2, data):

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
        output, _, __ = model(obs, mask, T, p=2)
        mean1, std1, mean2, std2 = output.chunk(4, dim=-1)
        std2 = np.log(1+np.exp(std2))
        std1 = np.log(1+np.exp(std1))
        means = mean1
        stds = std1
    
    return means[0,:,0], stds[0,:,0]

def predict_inverse_inverse(model, idx, time_len, condition_points, d_x, d_y1, d_y2, data):

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
        output, _, __ = model(obs, mask, T, p=2)
        mean1, std1, mean2, std2 = output.chunk(4, dim=-1)
        std2 = np.log(1+np.exp(std2))
        std1 = np.log(1+np.exp(std1))
        means = mean2
        stds = std2
    
    return means[0,:,0], stds[0,:,0]


############################################################################################################

def log_prob_loss(output, targets_f, targets_i, d_y1, d_y2, valid_inverses, traj_indices): # output (num_traj, num_tar, 2*d_y1 + 2*d_y2), targets (num_traj, num_tar, d_y1 + d_y2)
    d_y1, d_y2 = 7, 7
    means_f, stds_f, means_i, stds_i = output[:, :, :d_y1], output[:, :, d_y1:2*d_y1], output[:, :, 2*d_y1:2*d_y1+d_y2], output[:, :, 2*d_y1+d_y2:]

    stds_f = F.softplus(stds_f)
    stds_i = F.softplus(stds_i) # (num_traj, num_tar, d_y1)

    normal_f = D.Normal(means_f[:,:,:7], stds_f[:,:,:7])
    normal_i = D.Normal(means_i[:,:,:7], stds_i[:,:,:7])
    
    log_prob_f = normal_f.log_prob(targets_f[:,:,:7]) # (num_traj, num_tar)
    log_prob_i = normal_i.log_prob(targets_i[:,:,:7]) # (num_traj, num_tar)

    for i in range(len(traj_indices)):
        if not valid_inverses[traj_indices[i]]:
            log_prob_i[i] = torch.zeros(1)

    total_loss = torch.mean(log_prob_f + log_prob_i, dim=1) # (num_traj)

    return - total_loss.mean()  # scalar  

def compute_mse_of_pairs(L_F, L_I, valid_inverses, traj_indices): #L_F/I (num_traj, 128)

    mse = 0

    i = 0
    
    while not valid_inverses[traj_indices[i]]:
        i += 1
        if i == len(L_F):
            return torch.zeros(1)

    mse += F.pairwise_distance(L_F[i], L_I[i], p=2)
    return mse

def compute_distance_trajwise(L_F, L_I, valid_inverses, traj_indices):

    trajwise_dist = 0

    i = 0

    while not valid_inverses[traj_indices[i]]:
        i += 1
        if i == len(L_F):
            return torch.zeros(1)

    j = i + 1

    if j == len(L_F):
            return torch.zeros(1)

    while not valid_inverses[traj_indices[j]]:
        j += 1
        if j == len(L_F):
            return torch.zeros(1)

    # forward
    trajwise_dist += F.pairwise_distance(L_F[j], L_F[i], p=2)
    
    trajwise_dist += F.pairwise_distance(L_I[j], L_I[i], p=2)

    return trajwise_dist / 2

def compute_norm(L_F, L_I):
    norm = 0
    for i in range(len(L_F)):
        norm += torch.norm(L_F[i]) + torch.norm(L_I[i])
    return norm

def rescale_latent_representations(L_F, L_I):
    max = torch.max(torch.max(L_F), torch.max(L_I))
    min = torch.min(torch.min(L_F), torch.min(L_I))
    # rescale between -1 and 1
    L_F = 2 * (L_F - min) / (max - min) - 1
    L_I = 2 * (L_I - min) / (max - min) - 1
    return L_F, L_I

def plot_latent_space(model, observations_f, observations_i):
    with torch.no_grad():
        l_f = model.encoder1(observations_f) # condition points is (n, d_x + d_y), l is (n, 128)
        l_i = model.encoder2(observations_i)
    l_f = np.array(l_f)
    l_i = np.array(l_i)
    l = np.concat((l_f,l_i), axis=0)

    
    for i in range(len(l_f)):
        plt.scatter(i,np.linalg.norm(l_f[i]-l_i[i]), color='blue')
    plt.show()

    print(l.shape)

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(l)
    return pca_result


def tsne_analysis(model, observations_f, observations_i):
    with torch.no_grad():
        l_f = model.encoder1(observations_f) # condition points is (n, d_x + d_y), l is (n, 128)
        l_i = model.encoder2(observations_i)
    l_f = np.array(l_f)
    l_i = np.array(l_i)
    l = np.concat((l_f,l_i), axis=0)

    l = l.squeeze(1)

    tsne = TSNE(n_components=2, random_state=41)
    tsne_result = tsne.fit_transform(l)
    return tsne_result

