import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from source_varying_functions import s_function, A_matrix, rwmh, Model, log_prior_coefficients
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
T=10
Nt=100
Nx=100
Lx=1
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
    'ZS': 2.0,       # m, source height
    'Z': 2.0         # m, sensor height
}

beta = 1
sigma_epsilon = 0.01
# Define model class y(t,x)=A(x)s(t)+beta+epsilon
model = Model(0, 0, beta, sigma_epsilon, s_function, physical_constants)
t_start = 0


#Plots over spatial grid for fixed time t
data = model.gen_data(T,Nt,Nx,Lx,ak,bk,a0)
plt.contourf(data['X1'],data['X2'],data['Y'][0])
plt.colorbar()
plt.show()
# Now varies time in plot too
import matplotlib.animation as animation

# Set up the figure and axis
fig, ax = plt.subplots()

# Initial plot
data = model.gen_data(T,Nt,Nx,Lx,ak,bk,a0)
contour = ax.contourf(data['X1'], data['X2'], data['Y'][0], levels=20, cmap='viridis')
cbar = fig.colorbar(contour)
ax.set_title(f'Observation Model at t={t_start:.2f}')

def update(frame):
    ax.clear()
    t = frame * T / (Nt - 1)
    # data is already generated outside
    contour = ax.contourf(data['X1'], data['X2'], data['Y'][frame], levels=20, cmap='viridis')
    ax.set_title(f'Observation Model at t={t:.2f}')
    return contour,

# Create animation
ani = animation.FuncAnimation(fig, update, frames=Nt, interval=200)

# Save animation
try:
    ani.save('observation_Model_animation.gif', writer='pillow')
    print("Animation saved as observation_Model_animation.gif")
except Exception as e:
    print(f"Could not save animation: {e}")
    plt.show()




# Function for RWMHS given starting point, variance, time steps and posterior


#%%
# Generate data  on grid
data = model.gen_data(10,100,100,1,ak,bk,a0)
#Test log_prior
coefficients = [a0,ak,bk]
print(log_prior_coefficients(coefficients))
#y=As+beta+epsilon
# %%

# Test log_likelihood_y
ll = model.log_likelihood_y(coefficients,T,Nt,Nx,data)
print(f"Log likelihood: {ll}")



def log_posterior(coeff):
    return log_prior_coefficients(coeff) + model.log_likelihood_y(coeff,T,Nt,Nx,data)
    
#Run rwmh
initial_point = [0,0,0]
chain,acceptance_rate = rwmh(initial_point,1,10000,log_posterior)
print(chain)
print(acceptance_rate)
