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

def upper_lower_arcs(x, param, noise = 0):
    def dist(x, param, noise = 0):
        f = (math.exp(-x**2/(2.*param[0]**2))/(math.sqrt(2*math.pi)*param[0]))+param[1]
        return f+(noise*(np.random.rand()-0.5)/100.)
    return dist

def generate_demonstrations(time_len = 100, params = None, plot_title = None):

    params = np.array([[0.6,-0.1],[0.5,-0.23],[0.4,-0.43],[-0.6,0.1],[-0.5,0.23],[-0.4,0.43]])
    

    num_demo = params.shape[0]
    x = np.linspace(-0.5, 0.5, time_len)
    times = np.zeros((num_demo, time_len, 1))
    times[:] = x.reshape((1, time_len, 1)) + 0.5
    values = np.zeros((num_demo, time_len, 1))

    for d in range(num_demo):
        dist = upper_lower_arcs(time_len, params[d], plot_title)
        for i in range(time_len):
            values[d, i] = dist(x[i], params[d])
        plt.plot(times[d], values[d], color='black')
    
    plt.title(plot_title + ' Demonstrations')
    plt.ylabel('Y')
    plt.xlabel('time (t)')
    plt.savefig("../figs/TrainingDemonjkj strations.png")
    plt.show()
    
    Y1 = values[:num_demo//2]
    Y2 = values[num_demo//2:]
    X1 = times[:1]
    X2 = times[:1]
    return X1, X2, Y1, Y2

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
    for j in range(d_N):
        if j == idx:
                ## make bolder
            plt.plot(np.linspace(0, 100, time_len), Y1[j], color='blue', linewidth=4, label='Forward')
            plt.plot(np.linspace(0, 100, time_len), Y2[j], color='red', linewidth=4, label='Inverse')
            continue
        plt.plot(np.linspace(0, 100, time_len), Y1[j], color='black', alpha=0.4)
        plt.plot(np.linspace(0, 100, time_len), Y2[j], color='black', alpha=0.4)

    plt.plot(np.linspace(0, 100, time_len), means.detach()[0].numpy(), color='green', linewidth=3, label='Prediction')
    plt.errorbar(np.linspace(0, 100, time_len), means.detach()[0].numpy(), yerr=stds.detach()[0].numpy(), color='black', alpha=0.2)
    
    for i in range(len(condition_points)):
        cd_pt = condition_points[i]
        if i == 0:
            plt.scatter(cd_pt, Y1[idx, cd_pt], color='black', label='Observations')
            continue
        plt.scatter(cd_pt, Y1[idx, cd_pt], color='black')

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

    for i in range(d_N):
        if i == idx:
            plt.plot(np.linspace(0, 100, time_len), Y1[i], color='blue', linewidth=4, label = 'Forward')
            plt.plot(np.linspace(0, 100, time_len), Y2[i], color='red', linewidth=4, label = 'Inverse')
            continue
        plt.plot(np.linspace(0, 100, time_len), Y1[i], color='black', alpha=0.4)
        plt.plot(np.linspace(0, 100, time_len), Y2[i], color='black', alpha=0.4)

    ## add legend
    plt.plot(np.linspace(0, 100, time_len), best_mean.detach()[0].numpy(), color='green', linewidth=3, label='Best Prediction')
    plt.errorbar(np.linspace(0, 100, time_len), best_mean.detach()[0].numpy(), yerr=best_std.detach()[0].numpy(), color='black', alpha=0.2)

    for i in range(len(condition_points)):
        cd_pt = condition_points[i]
        if i == 0:
            plt.scatter(cd_pt, Y1[idx, cd_pt], color='black', label='Observations')
            continue
        plt.scatter(cd_pt, Y1[idx, cd_pt], color='black')

    plt.title('Best Prediction')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig("../figs/BestPrediction.png")
    plt.show()

