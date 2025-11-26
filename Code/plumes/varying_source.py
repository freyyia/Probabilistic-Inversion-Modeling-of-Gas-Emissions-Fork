# DEfine model y(t,x)=A(x)s(t)+beta+epsilon
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import source_varying_functions as svf
import numpy as np
import matplotlib.pyplot as plt
from math import cos, sin
from collections.abc import Iterable
# s(t)=sum_k a_k cost(2pi kt) + sum_k b_k sin(2pi kt) + a_0



#Test s_function
a0 =4
ak = np.array([1])
bk = np.array([1])
t = 1

print(s_function(t,ak,bk,a0))
#Plots source function over [0,2pi]
t = np.linspace(0,0.5,100)
plt.plot(t,list(map(lambda t: s_function(t,ak,bk,a0),t)))
plt.show()


#Defines the coupling matrix that  maps source (x_s,y_s) to concentration at (x,y)
# 

Asaved = np.load('A_matrix.npy')


beta = 1
sigma_epsilon = 0.01
# Define model class y(t,x)=A(x)s(t)+beta+epsilon
class Model:
    def __init__(self,x_1s,x_2s,beta,sigma_epsilon,s_function, A_matrix):
        self.x_1s = x_1s
        self.x_2s = x_2s
        self.beta = beta
        self.sigma_epsilon = sigma_epsilon
        self.s_function = s_function
        self.A_matrix = A_matrix
    def y(self,x_1,x_2,t):
        return self.A_matrix(self.x_1s,self.x_2s,x_1,x_2)*self.s_function(t,ak,bk,a0)+self.beta+np.random.normal(0,self.sigma_epsilon)

#Tests the model
x_1s =50
x_2s =50
model = Model(x_1s,x_2s,beta,sigma_epsilon,s_function,A_matrix)
t=0.5
x_1 = 1
x_2 = 1
y = model.y(x_1,x_2,t)
print(y)
#Plots over spatial grid for fixed time t
x_1 = np.linspace(-1,1,100)
x_2 = np.linspace(-1,1,100)
X_1,X_2 = np.meshgrid(x_1,x_2)
Y = np.array([model.y(x_1,x_2,t) for x_1,x_2 in zip(X_1.flatten(),X_2.flatten())]).reshape(X_1.shape)
plt.contourf(X_1,X_2,Y)
plt.colorbar()
plt.show()
# Now varies time in plot too
import matplotlib.animation as animation

# Set up the figure and axis
fig, ax = plt.subplots()
x_1 = np.linspace(-1, 1, 100)
x_2 = np.linspace(-1, 1, 100)
X_1, X_2 = np.meshgrid(x_1, x_2)

# Initial plot
t_start = 0
Y = np.array([model.y(x_1, x_2, t_start) for x_1, x_2 in zip(X_1.flatten(), X_2.flatten())]).reshape(X_1.shape)
contour = ax.contourf(X_1, X_2, Y, levels=20, cmap='viridis')
cbar = fig.colorbar(contour)
ax.set_title(f'Observation Model at t={t_start:.2f}')

# Create animation
ani = animation.FuncAnimation(fig, update, frames=50, interval=200)

# Save animation
try:
    ani.save('observation_model_animation.gif', writer='pillow')
    print("Animation saved as observation_model_animation.gif")
except Exception as e:
    print(f"Could not save animation: {e}")
    plt.show()

# Generate data
def gen_data(model,T):
    x_1 = np.linspace(-1, 1, 100)
    x_2 = np.linspace(-1, 1, 100)
    X_1, X_2 = np.meshgrid(x_1, x_2)
    Y = np.array([])
    for t in np.linspace(0,T,100):
        Yt = np.array([model.y(x_1, x_2, t) for x_1, x_2 in zip(X_1.flatten(), X_2.flatten())]).reshape(X_1.shape)
        Y = np.append(Y,Yt)
    return Y


# Function for RWMHS given starting point, variance, time steps and posterior


#%%
# Generate data  on grid
data = gen_data(model,10)
#Define log prior
def log_prior_coefficients(coeff):
    a0 = coeff[0]
    ak = np.atleast_1d(coeff[1])
    bk = np.atleast_1d(coeff[2])
    n_coeff = len(ak)
    variance_k = [1/(1+(k+1)**2) for k in range(n_coeff)]
    return -1/2 * np.sum((ak)**2/variance_k) -1/2 * np.sum((bk)**2/variance_k) -1/2 * (a0)**2

coefficients = [a0,ak,bk]
print(log_prior_coefficients(coefficients))
#y=As+beta+epsilon
# %%

# Test log_likelihood_y
ll = log_likelihood_y(coefficients, data, x_1s, x_2s, beta, sigma_epsilon, A_matrix)
print(f"Log likelihood: {ll}")



#Run rwmh
initial_point = [0,0,0]
chain,acceptance_rate = rwmh(initial_point,1,10000,log_posterior)
print(chain)
print(acceptance_rate)
