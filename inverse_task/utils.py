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

def gaussian_with_offset(param, noise = 0):
    def dist(x, param, noise = 0):
        f = (math.exp(-x**2/(2.*param[0]**2))/(math.sqrt(2*math.pi)*param[0]))+param[1]
        return f+(noise*(np.random.rand()-0.5)/100.)
    return dist

def x_sinx(param, noise = 0):
    def dist(x, param, noise = 0):
        f =  2*x  + math.sin(param[0] * x) #+ param[1]
        return f+(noise*(np.random.rand()-0.5)/100.)
    return dist

def generate_demonstrations(time_len = 200, params = None, plot_title = None):


    num_demo = 128
    x = np.linspace(0, 1, time_len)
    times = np.zeros((2*num_demo, time_len, 1))
    times[:] = x.reshape((1, time_len, 1))
    values = np.zeros((2*num_demo, time_len, 1))

    """
    for d in range(num_demo):
        #dist = gaussian_with_offset(params[d])
        dist = x_sinx(params[d])
        for i in range(time_len):
            values[d, i] = dist(x[i], params[d])
            # reverse array
        values[d+num_demo] = np.flip(values[d])
        # normalize between -1 , 1
        values[d] = (values[d] - np.min(values[d])) / (np.max(values[d]) - np.min(values[d])) * 2 - 1
        values[d+num_demo] = (values[d+num_demo] - np.min(values[d+num_demo])) / (np.max(values[d+num_demo]) - np.min(values[d+num_demo])) * 2 - 1
        if d == validation_idx:
            plt.plot(times[d], values[d], color= f"black") #, label = f"Forward Validation Trajectory")
            plt.plot(times[d], values[d+num_demo], color= f"black") #, label = f"Inverse Validation Trajectory")
        else:
            plt.plot(times[d], values[d], color= f"{1 / ((d+1))}")
            plt.plot(times[d], values[d+num_demo], color= f"{1 / ((d+1))}")
    """

    data_path = 'y.pt'

    forward = torch.load(data_path, map_location='cpu').to('cpu').numpy().squeeze(-1)
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

def test_model(best_m, best_s, means, stds, idx, demo_data, errors, time_len, condition_points, epoch_count, plot=False):

    X1, X2, Y1, Y2 = demo_data

    target_demo = Y2[idx]
    target_demo = torch.from_numpy(target_demo).double()
    target_demo = target_demo.reshape(1,-1)
    error = torch.mean(torch.nn.functional.mse_loss(means, target_demo))
    target_demo = target_demo.reshape(-1,1)
    errors.append(error.item())

    best_mean = best_m
    best_std = best_s
    if error <= min(errors):
        best_mean = means
        best_std = stds

    if plot:
        plot_test(idx, Y1, Y2, means, stds, time_len, condition_points, epoch_count)

    return errors, best_mean, best_std     

def plot_test(idx, Y1, Y2, means, stds, time_len, condition_points, epoch_count):
    d_N = Y1.shape[0]
    print(Y1.shape)
    T = np.linspace(0,1,time_len)
    for j in range(d_N):
        if j == idx:
                ## make bolder
            plt.plot(T, Y1[j], color='black', alpha=0.02)
            plt.plot(T, Y2[j], color='red', label='Expected (Inverse)')
            continue
        plt.plot(T, Y1[j], color='black', alpha=0.02)
        plt.plot(T, Y2[j], color='black', alpha=0.02)

    plt.plot(T, means.detach()[0].numpy(), color='green', label='Prediction')
    plt.errorbar(T, means.detach()[0].numpy(), yerr=stds.detach()[0].numpy(), color='black', alpha=0.2)
    
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

def plot_results(best_mean, best_std, Y1, Y2, idx, condition_points, errors, losses, time_len, d_N):
    
    plt.plot(errors)
    plt.title('Errors')
    plt.show()

    plt.plot(losses)
    plt.title('Loss')
    plt.show()

    T = np.linspace(0,1,time_len)

    for i in range(d_N):
        if i == idx:
            plt.plot(T, Y1[i], color='black', alpha=0.02)
            plt.plot(T, Y2[i], color='red', label = 'Expected (Inverse)')
            continue
        plt.plot(T, Y1[i], color='black', alpha=0.02)
        plt.plot(T, Y2[i], color='black', alpha=0.02)

    ## add legend
    plt.plot(T, best_mean.detach()[0].numpy(), color='green', label='Best Prediction')
    plt.errorbar(T, best_mean.detach()[0].numpy(), yerr=best_std.detach()[0].numpy(), color='black', alpha=0.2)

    for i in range(len(condition_points)):
        cd_pt_x = condition_points[i][0]
        cd_pt_y = condition_points[i][1]
        if i == 0:
            plt.scatter(cd_pt_x, cd_pt_y, color='black', label='Observations')
            continue
        plt.scatter(cd_pt_x, cd_pt_y, color='black')

    plt.title('Best Prediction')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig("../figs/BestPrediction.png")
    plt.show()

