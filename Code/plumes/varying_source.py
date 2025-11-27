import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from source_varying_functions import s_function, A_matrix, rwmh, log_likelihood_y
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

# Constants
physical_constants = {
    'RHO_CH4': 0.656, # kg/m^3, density of methane at 25 deg C and 1 atm
    'U': 5.0,         # m/s, wind speed
    'wind_vector': np.array([1,0]),
    'SIGMA_H': 10.0,  # m, horizontal dispersion coefficient
    'SIGMA_V': 10.0,  # m, vertical dispersion coefficient
    'N_REFL': 5,      # number of reflections
    'P': 1000.0,      # m, PBL height
    'XS': 50.0,       # m, source x-coordinate
    'YS': 50.0,       # m, source y-coordinate
    'ZS': 50.0,       # m, source height
    'Z': 2.0         # m, sensor height
}

beta = 1
sigma_epsilon = 0.01
# Define model class y(t,x)=A(x)s(t)+beta+epsilon


#Tests the model
x_1s =50
x_2s =50
model = Model(x_1s,x_2s,beta,sigma_epsilon,s_function,physical_constants)
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

def update(frame):
    ax.clear()
    t = frame * 0.01  # Time step
    Y = np.array([model.y(x_1, x_2, t) for x_1, x_2 in zip(X_1.flatten(), X_2.flatten())]).reshape(X_1.shape)
    contour = ax.contourf(X_1, X_2, Y, levels=20, cmap='viridis')
    ax.set_title(f'Observation Model at t={t:.2f}')
    return contour,

# Create animation
ani = animation.FuncAnimation(fig, update, frames=50, interval=200)

# Save animation
try:
    ani.save('observation_model_animation.gif', writer='pillow')
    print("Animation saved as observation_model_animation.gif")
except Exception as e:
    print(f"Could not save animation: {e}")
    plt.show()




# Function for RWMHS given starting point, variance, time steps and posterior


#%%
# Generate data  on grid
data = model.gen_data(10,100,100)
#Test log_prior
coefficients = [a0,ak,bk]
print(log_prior_coefficients(coefficients))
#y=As+beta+epsilon
# %%

# Test log_likelihood_y
ll = log_likelihood_y(coefficients, data, x_1s, x_2s, beta, sigma_epsilon, A_matrix)
print(f"Log likelihood: {ll}")



def log_posterior(coeff):
    return log_prior_coefficients(coeff) + log_likelihood_y(coeff, data, x_1s, x_2s, beta, sigma_epsilon, A_matrix)
    
#Run rwmh
initial_point = [0,0,0]
chain,acceptance_rate = rwmh(initial_point,1,10000,log_posterior)
print(chain)
print(acceptance_rate)
