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

    frequencies = np.linspace(0.75,1.5,num_demo//2)  # Example frequencies
    #frequencies = np.linspace(1,2,num_demo)
    #amplitudes = [1.1, 1]  # Example amplitudes
    amplitudes = np.linspace(0,1.0,num_demo//3)
    
    phases = [0]  # Example phases
        
    for d in range(num_demo):
        #dist = gaussian_with_offset(params[d])
        #dist = x_sinx(params[d])

        """
        if d < num_demo//3:
            dist = sinx(frequencies[d % len(frequencies)], amplitudes[d % len(amplitudes)], phases[d % len(phases)])
        elif d > num_demo//3 and d < 2*num_demo//3:
            dist = linear(amplitudes[d % len(amplitudes)], -1 * amplitudes[d % len(amplitudes)])
        else:
            dist = x_sinx([0.7*amplitudes[d % len(amplitudes)], 20*frequencies[d % len(frequencies)]])
        """
        
        if d < num_demo//2:
            dist = sinx(frequencies[d % len(frequencies)], amplitudes[d % len(amplitudes)], phases[d % len(phases)])
        else:
            dist = linear(amplitudes[(d-num_demo//2) % len(amplitudes)], -1 * amplitudes[(d-num_demo//2) % len(amplitudes)])
        
        # dist = sinx(frequencies[d % len(frequencies)], amplitudes[d % len(amplitudes)], phases[d % len(phases)])

        for i in range(time_len):
            values[d, i] = dist(x[i])
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
    target_demo = torch.from_numpy(target_demo).double()
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
        plt.ylim(-6, 100)    
        plt.title('Loss')
        plt.show()

    T = np.linspace(0,1,time_len)

    #plot Y1 and Y2 in different subplots

    plt.figure(figsize=(20, 10))
    
    #access subplot
    sub0 = plt.subplot(1, 2, 1)
    sub1 = plt.subplot(1, 2, 2)

    for ax in (sub0, sub1):
        ax.axhline(y=condition_points[0][-1], color='black', linewidth=0.5)

    
    for i in range(d_N):
        if i == 0:
            sub0.plot(T, Y1[i], color='black', alpha=0.06, label = "Forward Trajectories")
            continue
        sub0.plot(T, Y1[i], color='black', alpha=0.06)
    
    for i in range(len(condition_points)):
        cd_pt_x = condition_points[i][0]
        cd_pt_y = condition_points[i][1]
        if i == 0:
            sub0.scatter(cd_pt_x, cd_pt_y, color='black', label='Observations') 
            continue
        sub0.scatter(cd_pt_x, cd_pt_y, color='black')
    
    sub0.set_title('Forward Trajectories')
    sub0.legend(loc='center left', bbox_to_anchor=(1, 0.5))


    for i in range(d_N):
        if i == 0:
            sub1.plot(T, Y2[i], color='black', alpha=0.06, label = "Inverse Trajectories")
            continue
        sub1.plot(T, Y2[i], color='black', alpha=0.06)
    
    ## add legend
    sub1.plot(T, best_mean.detach().numpy(), color='green', label='Prediction')
    sub1.errorbar(T, best_mean.detach().numpy(), yerr=best_std.detach().numpy(), color='black', alpha=0.2)

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
    plt.savefig("../figs/Results.png")
    plt.show()

def forward_inverse_weights(model):
    weights = model.state_dict()
    ## store encoder1.0,2,4 weights and biases
    encoder1_0_weights = weights['encoder1.0.weight']
    encoder1_0_bias = weights['encoder1.0.bias']
    encoder1_2_weights = weights['encoder1.2.weight']
    encoder1_2_bias = weights['encoder1.2.bias']
    encoder1_4_weights = weights['encoder1.4.weight']
    encoder1_4_bias = weights['encoder1.4.bias']
    E1 = np.concat([encoder1_0_weights.flatten(), encoder1_2_weights.flatten(), encoder1_4_weights.flatten(), encoder1_0_bias.flatten(), encoder1_2_bias.flatten(), encoder1_4_bias.flatten()], axis=0)


    ## store encoder2.0,2,4 weights and biases
    encoder2_0_weights = weights['encoder2.0.weight']
    encoder2_0_bias = weights['encoder2.0.bias']
    encoder2_2_weights = weights['encoder2.2.weight']
    encoder2_2_bias = weights['encoder2.2.bias']
    encoder2_4_weights = weights['encoder2.4.weight']
    encoder2_4_bias = weights['encoder2.4.bias']
    E2 = np.concat([encoder2_0_weights.flatten(), encoder2_2_weights.flatten(), encoder2_4_weights.flatten(), encoder2_0_bias.flatten(), encoder2_2_bias.flatten(), encoder2_4_bias.flatten()], axis=0)

    ## store decoder1.0,2,4 weights and biases
    decoder1_0_weights = weights['decoder1.0.weight']
    decoder1_0_bias = weights['decoder1.0.bias']
    decoder1_2_weights = weights['decoder1.2.weight']
    decoder1_2_bias = weights['decoder1.2.bias']
    decoder1_4_weights = weights['decoder1.4.weight']
    decoder1_4_bias = weights['decoder1.4.bias']
    D1 = np.concat([decoder1_0_weights.flatten(), decoder1_2_weights.flatten(), decoder1_4_weights.flatten(), decoder1_0_bias.flatten(), decoder1_2_bias.flatten(), decoder1_4_bias.flatten()], axis=0)

    ## store decoder2.0,2,4 weights and biases
    decoder2_0_weights = weights['decoder2.0.weight']
    decoder2_0_bias = weights['decoder2.0.bias']
    decoder2_2_weights = weights['decoder2.2.weight']
    decoder2_2_bias = weights['decoder2.2.bias']
    decoder2_4_weights = weights['decoder2.4.weight']
    decoder2_4_bias = weights['decoder2.4.bias']
    D2 = np.concat([decoder2_0_weights.flatten(), decoder2_2_weights.flatten(), decoder2_4_weights.flatten(), decoder2_0_bias.flatten(), decoder2_2_bias.flatten(), decoder2_4_bias.flatten()], axis=0)

    F = np.concat([E1,D2], axis=0)
    I = np.concat([E2,D1], axis=0)

    return F, I