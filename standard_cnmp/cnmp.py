import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import math

time_len = 200

def generate_demonstrations(time_len = 200, params = None, title = None):
    def dist_generator(d, x, param, noise = 0):
        f = (math.exp(-x**2/(2.*param[0]**2))/(math.sqrt(2*math.pi)*param[0]))+param[1]
        return f+(noise*(np.random.rand()-0.5)/100.)
    def sinx(x, frequency, amplitude, phase):
        return amplitude * math.sin(2 * math.pi * frequency * x + phase)
    
    num_demo = 32
    frequencies = [1]
    amplitudes = np.linspace(0.5,1,num_demo)
    phases = [0] 

    #fig = plt.figure(figsize=(5,5))
    x = np.linspace(-0.5,0.5,time_len)
    times = np.zeros((num_demo,time_len,1))
    times[:] = x.reshape((1,time_len,1))+0.5
    values = np.zeros((num_demo,time_len,1))
    for d in range(num_demo):
        for i in range(time_len):
            values[d,i] = sinx(times[0][i][0], frequencies[d % len(frequencies)], amplitudes[d % len(amplitudes)], phases[d % len(phases)])
        plt.plot(times[d], values[d], color="black", alpha=0.05)
    #plt.title(title+' Demonstrations')
    #plt.ylabel('Y')
    #plt.xlabel('time (t)')
    plt.show()
    return times, values

# gets random number of random obs. points from a random trajectory. Also gets a 
# random target (x,y) from the same trajectory
def get_training_sample():
    
    n = np.random.randint(0,OBS_MAX)+1
    d = np.random.randint(0, d_N)
    
    observations = np.zeros((n,d_x+d_y)) 
    target_X = np.zeros((1,d_x))
    target_Y = np.zeros((1,d_y))
    
    perm = np.random.permutation(time_len)
    observations[:,:d_x] = X[d,perm[:n]]
    observations[:,d_x:d_x+d_y] = Y[d,perm[:n]]
    target_X[0] = X[d,perm[n]]
    target_Y[0] = Y[d,perm[n]]
    return torch.from_numpy(observations), torch.from_numpy(target_X), torch.from_numpy(target_Y)

def log_prob_loss(output, y_target): 
    mean, std = output.chunk(2, dim=-1)
    std = F.softplus(std)
    dist = D.Independent(D.Normal(loc=mean, scale=std), 1)  # (d_y distributions)
    return -torch.mean(dist.log_prob(y_target)) 

def predict_model(observations, target_X, plot = True):
    d_N = X.shape[0]
    predicted_Y = np.zeros((time_len,d_y))
    predicted_std = np.zeros((time_len,d_y))
    with torch.no_grad():
        prediction = model(torch.from_numpy(observations),torch.from_numpy(target_X)).numpy()
    predicted_Y = prediction[:,:d_y]
    predicted_std = np.log(1+np.exp(prediction[:,d_y:]))
    if plot: # We highly recommend that you customize your own plot function, but you can use this function as default
        for i in range(d_y): #for every feature in Y vector we are plotting training data and its prediction
            fig = plt.figure(figsize=(5,5))
            for j in range(d_N):
                plt.plot(X[j,:,0],Y[j,:,i]) # assuming X[j,:,0] is time
            plt.plot(X[j,:,0],predicted_Y[:,i],color='black')
            plt.errorbar(X[j,:,0],predicted_Y[:,i],yerr=predicted_std[:,i],color = 'black',alpha=0.4)
            plt.scatter(observations[:,0],observations[:,d_x+i],marker="X",color='black')
            plt.show()  
    return predicted_Y, predicted_std

class CNMP(nn.Module):
    def __init__(self, d_x, d_y):
        super(CNMP, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(d_x + d_y, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 128)
        )

        self.decoder = nn.Sequential(
            nn.Linear(d_x + 128, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 2*d_y)
        )
    def forward(self, obs, x_tar): # obs is (n, d_x + d_y)

        r = self.encoder(obs) # (n,128)
        r_avg = torch.mean(r, 0) # (1,128)
        r_avg = r_avg.repeat(x_tar.shape[0],1) # Duplicating general representation for every target_t

        concat = torch.cat((r_avg, x_tar), dim=-1)
        output = self.decoder(concat) # (2*d_y,)
        return output

############################################################################################################

X, Y = generate_demonstrations(time_len=200, params=np.array([[0.6,-0.1],[0.5,-0.23],[0.4,-0.43],[-0.6,0.1],[-0.5,0.23],[-0.4,0.43]]), title='Training')


print('training X ', X.shape)
print('training Y ',Y.shape)

OBS_MAX = 5
d_x = X.shape[-1]
d_y = Y.shape[-1]
d_N = X.shape[0]

model = CNMP(d_x, d_y).double()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
#print(model.eval())

for i in range(100000):

    obs, x_tar, y_tar = get_training_sample()

    optimizer.zero_grad()

    output = model(obs, x_tar)
    loss = log_prob_loss(output, y_tar)
    
    loss.backward()
    optimizer.step()
    #print('loss ', loss.item())

    if i % 1000 == 0:
        #print('iteration ', i)
        print('loss ', loss.item())
        


predicted_Y,predicted_std = predict_model((np.concat((X[0,:1],Y[0,:1]), axis=-1)),X[0])

# Testing







