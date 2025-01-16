from cProfile import label
from fileinput import filename
from re import A, X
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
import time
import dual_enc_dec_cnmp

def gaussian_with_offset(param, noise = 0):
    def dist(x, param, noise = 0):
        f = (math.exp(-x**2/(2.*param[0]**2))/(math.sqrt(2*math.pi)*param[0]))+param[1]
        return f+(noise*(np.random.rand()-0.5)/100.)
    return dist

def x_sinx(param, noise = 0):
    def dist(x, noise = 0):
        f =  x  + param[0] * math.sin(param[1] * x) #+ param[1]
        return f+(noise*(np.random.rand()-0.5)/100.)
    return dist

def sinx(frequency, amplitude, phase):
    def dist(x):
        return amplitude * math.sin(2 * torch.pi * frequency * x + phase) # + torch.randn(1) * 0.05
    return dist

# write a linear function
def linear(min, max):
    def dist(x):
        return max * x + min * (1-x)
    return dist

def generate_demonstrations(num_demo, time_len = 200, params = None, plot_title = None):

    x = np.linspace(0, 1, time_len)
    times = np.zeros((2*num_demo, time_len, 1))
    times[:] = x.reshape((1, time_len, 1))
    values = np.zeros((2*num_demo, time_len, 1))

    #frequencies = np.linspace(0.75,1.5,num_demo//2)  # Example frequencies
    frequencies = [1, 1.75, 2.5]  # Example frequencies
    #frequencies = np.linspace(1,2,num_demo)
    #amplitudes = [1.1, 1]  # Example amplitudes
    amplitudes = np.linspace(0.1,0.5,num_demo)
    
    phases = [0]  # Example phases
        
    for d in range(num_demo):

        dist1 = sinx(frequencies[0], amplitudes[d % len(amplitudes)], phases[d % len(phases)])
        dist2 = sinx(frequencies[1], amplitudes[d % len(amplitudes)], phases[d % len(phases)])
        dist3 = sinx(frequencies[2], amplitudes[d % len(amplitudes)], phases[d % len(phases)])

        for i in range(time_len):
            values[d, i] = dist1(x[i]) #+ dist2(x[i]) + dist3(x[i])
        # reverse array
        values[d+num_demo] = np.flip(values[d], axis=0)
        #values[d+num_demo] = -1 * values[d]
        # normalize between -1 , 1
        #values[d] = (values[d] - np.min(values[d])) / (np.max(values[d]) - np.min(values[d])) * 2 - 1
        #values[d+num_demo] = (values[d+num_demo] - np.min(values[d+num_demo])) / (np.max(values[d+num_demo]) - np.min(values[d+num_demo])) * 2 - 1
        plt.plot(times[d], values[d], color="black", alpha=0.5)
        plt.plot(times[d], values[d+num_demo], color="black", alpha=0.5)
    
    plt.title(plot_title + ' Demonstrations')
    plt.ylabel('Y')
    plt.xlabel('time (t)')
    plt.savefig("../figs/TrainingDemonstrations.png")
    plt.show()
    
    """
    data_path = 'y.pt'

    forward = torch.load(data_path, map_location='cpu').to('cpu').numpy().squeeze(-1)
    # take only odd indices, use np
    forward = forward[::2]
    # delete the middle 32 trajectory
    for i in range(32):
        curr_demo_num = forward.shape[0]
        forward = np.delete(forward, curr_demo_num//2, axis=0)
    

    inverse = np.flip(forward, axis=1)
    inverse = inverse.copy()

    for forward_traj, inverse_traj in zip(forward, inverse):
        plt.plot(forward_traj, color='black', alpha=0.05)
        plt.plot(inverse_traj, color='black', alpha=0.05)
        
    plt.title(plot_title + ' Demonstrations')
    plt.ylabel('Y')
    plt.xlabel('time (t)')
    plt.savefig("../figs/TrainingDemonstrations.png")
    plt.show()
    """
    
    forward = values[:num_demo]
    inverse = values[num_demo:]
    # num_demo is the half of the number of demonstrations
    Y1 = forward.reshape(num_demo, time_len, 1)
    Y2 = inverse.reshape(num_demo, time_len, 1)
    X1 = times[:1]
    X2 = times[:1]

    plt.title("Forward")
    for i in range(num_demo):
        plt.plot(times[i], forward[i], color = "black", alpha=0.05)
    #plt.ylim(-0.3, 0.3)
    plt.show()

    plt.title("Inverse")
    for i in range(num_demo):
        plt.plot(times[i], inverse[i], color = "black", alpha=0.05)
    #plt.ylim(-0.3, 0.3)
    plt.show()

    validation_forward = np.zeros((16, time_len, 1))
    validation_inverse = np.zeros((16, time_len, 1))
    for i in range(len(forward)):
        if i % 8 == 0:
            validation_forward[i//8] = Y1[i]
            validation_inverse[i//8] = Y2[i]

    return X1, X2, Y1, Y2, validation_forward, validation_inverse

def validate_model(means, stds, idx, demo_data, time_len, condition_points, epoch_count, plot=False):

    X1, X2, Y1, Y2 = demo_data

    target_demo = Y2[idx]
    target_demo = target_demo.reshape(time_len)
    error = torch.mean(torch.nn.functional.mse_loss(means, target_demo))

    if plot:
        plot_test(idx, Y1, Y2, means, stds, time_len, condition_points, epoch_count)
    
    return error

def plot_test(idx, Y1, Y2, means, stds, time_len, condition_points, epoch_count):
    d_N = Y1.shape[0]
    print(Y1.shape)
    T = np.linspace(0,1,time_len)
    for j in range(d_N):
        if j == idx:
            plt.plot(T, Y1[j], color='blue', label='Forward')
            plt.plot(T, Y2[j], color='red', label='Expected (Inverse)')
            continue
        plt.plot(T, Y1[j], color='black', alpha=0.02)
        if j<96:
            plt.plot(T, Y2[j], color='black', alpha=0.02)

    plt.plot(T, means.detach().numpy(), color='green', label='Prediction')
    plt.errorbar(T, means.detach().numpy(), yerr=stds.detach().numpy(), color='black', alpha=0.2)
    
    for i in range(len(condition_points)):
        cd_pt_x = condition_points[i][0]
        cd_pt_y = condition_points[i][1]
        if i == 0:
            pass
            plt.scatter(cd_pt_x, cd_pt_y, color='black', label='Observations')
            continue
        plt.scatter(cd_pt_x, cd_pt_y, color='black')

    plt.title(f'Prediction at epoch {epoch_count}')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig(f"../figs/Prediction_{epoch_count}.png")
    plt.show()

def plot_results(best_mean, best_std, Y1, Y2, idx, condition_points, errors, losses, time_len, d_N, plot_errors=True, test_dist=None):
    
    if plot_errors:
        plt.plot(errors)
        plt.title('Errors')
        plt.show()

        plt.figure(figsize=(20, 10))
        plt.plot(losses)
        plt.grid()
        plt.ylim(-10, 100)    
        plt.title('Loss')
        plt.show()

    T = np.linspace(0,1,time_len)

    #plot Y1 and Y2 in different subplots

    plt.figure(figsize=(20, 10))
    
    #access subplot
    sub0 = plt.subplot(1, 2, 1)
    sub1 = plt.subplot(1, 2, 2)

    for ax in (sub0, sub1):
        #ax.axhline(y=condition_points[0][-1], color='black', linewidth=0.5)
        pass

    for i in range(d_N):
        if i == 0:
            sub0.plot(T, Y1[i], color='black', alpha=0.06, label = "Forward Trajectories")
            continue
        sub0.plot(T, Y1[i], color='black', alpha=0.06)
    
    for i in range(len(condition_points)):
        cd_pt_x = condition_points[i][0]
        cd_pt_y = condition_points[i][1]
        if i == 0:
            sub1.scatter(cd_pt_x, cd_pt_y, color='black', label='Observations') 
            continue
        sub1.scatter(cd_pt_x, cd_pt_y, color='black')
    
    sub0.set_title('Forward Trajectories')
    sub0.legend(loc='center left', bbox_to_anchor=(1, 0.5))


    
    for i in range(d_N):
        if i == 0:
            sub1.plot(T, Y2[i], color='black', alpha=0.06, label = "Inverse Trajectories")
            continue
        if i<96:
            sub1.plot(T, Y2[i], color='black', alpha=0.06)
    
    ## add legend
    sub0.plot(T, best_mean.detach().numpy(), color='green', label='Prediction')
    sub0.errorbar(T, best_mean.detach().numpy(), yerr=best_std.detach().numpy(), color='black', alpha=0.2)

    # set y-ticks for every 0.1 from -2.5 to 2.5, for two axes
    sub1.set_yticks(np.arange(-0.7, 0.7, 0.1))
    sub0.set_yticks(np.arange(-0.7, 0.7, 0.1))
    sub0.grid(alpha=0.3)
    sub1.grid(alpha=0.3)

    if test_dist:
        dist, i, num_validate = test_dist[0], test_dist[1], test_dist[2]
        amplitudes = np.linspace(0.5,1.0, num_validate)
        Y = []
        for t in T:
            Y.append(dist(amplitudes[i],t))
        Y = np.flip(np.array(Y))
        sub1.plot(T, Y, color="blue", label="Expected")

    sub1.set_title('Inverse Trajectories and Best Prediction')
    sub1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    t = int(time.time())
    plt.savefig(f"../figs/Results.png")
    plt.show()


def plot_results_ii(best_mean, best_std, Y1, Y2, idx, condition_points, errors, losses, time_len, d_N, plot_errors=True, test_dist=None):
    
    T = np.linspace(0,1,time_len)

    #plot Y1 and Y2 in different subplots

    plt.figure(figsize=(10,5))

    """
    for i in range(d_N):
        if i == 0:
            plt.plot(T, Y1[i], color='black', alpha=0.06, label = "Forward Trajectories")
            continue
        plt.plot(T, Y1[i], color='black', alpha=0.06)
    """
    
    for i in range(len(condition_points)):
        cd_pt_x = condition_points[i][0]
        cd_pt_y = condition_points[i][1]
        if i == 0:
            plt.scatter(cd_pt_x, cd_pt_y, color='black', label='Observations') 
            continue
        plt.scatter(cd_pt_x, cd_pt_y, color='black')
    
    
    for i in range(d_N):
        if i == 0:
            plt.plot(T, Y2[i], color='black', alpha=0.06, label = "Inverse Trajectories")
            continue
        if i<96:
            plt.plot(T, Y2[i], color='black', alpha=0.06)
    
    ## add legend
    plt.plot(T, best_mean.detach().numpy(), color='green', label='Prediction')
    plt.errorbar(T, best_mean.detach().numpy(), yerr=best_std.detach().numpy(), color='black', alpha=0.2)

    # set y-ticks for every 0.1 from -2.5 to 2.5, for two axes
    plt.yticks(np.arange(-0.7, 0.7, 0.1))
    plt.grid(alpha=0.3)

    plt.title('Inverse Trajectories and Best Prediction')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    t = int(time.time())
    plt.savefig(f"../figs/Results.png")
    plt.show()


def plot_latent_tsne(model, Y1, Y2, epoch):
    
    time_len=200
    time_ = [0.25, 0.75]

    ## do the same for t = 0.25 and t = 0.75, obtain the pca result for all the points

    intervals_y_f = []
    intervals_y_i = []

    all_observations_f = []  
    all_observations_i = []

    NUM_STEP = 200

    for t in time_:
        min_y_f = torch.min(Y1[:, int(t * time_len)])
        max_y_f = torch.max(Y1[:, int(t * time_len)])
        interval_y_f = torch.linspace(min_y_f, max_y_f, NUM_STEP)
        if t > 0.5:
            interval_y_f = torch.flip(interval_y_f, dims=[0])
        intervals_y_f.append(interval_y_f)

        min_y_i = torch.min(Y2[:, int(t * time_len)])
        max_y_i = torch.max(Y2[:, int(t * time_len)])
        interval_y_i = torch.linspace(min_y_i, max_y_i, NUM_STEP)
        if t < 0.5:
            interval_y_i = torch.flip(interval_y_i, dims=[0])
        intervals_y_i.append(interval_y_i)

        for idx in interval_y_f:
            condition_points = [[t, idx]]
            all_observations_f.append(condition_points)
        for idx in interval_y_i:
            condition_points = [[t, idx]]
            all_observations_i.append(condition_points)

    all_observations_f = torch.tensor(all_observations_f)
    all_observations_i = torch.tensor(all_observations_i)

    tsne_result = dual_enc_dec_cnmp.tsne_analysis(model, all_observations_f, all_observations_i) # [ft0, ft1, ... , it0, it1, ...]
    plt.figure(figsize=(5, 5))
    for i, t in enumerate(time_):
        plt.scatter(tsne_result[i*NUM_STEP:(i+1)*NUM_STEP, 0], tsne_result[i*NUM_STEP:(i+1)*NUM_STEP, 1], label=f"t={t} forward", s=8)
    
    for i, t in enumerate(time_):
        plt.scatter(tsne_result[(i+len(time_))*NUM_STEP:(i+len(time_)+1)*NUM_STEP, 0], tsne_result[(i+len(time_))*NUM_STEP:(i+len(time_)+1)*NUM_STEP, 1], 
                    label=f"t={t} inverse", s=8)
    
    
    plt.legend()
    plt.title(f"t-SNE plot for latent space at epoch {epoch}")
    filename = time.time()
    plt.tight_layout()
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid()
    plt.savefig(f"../figs/tsne_{filename}.png", bbox_inches='tight')
    plt.show()