# %%
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
a0 = np.array([4.0, 2.0])
ak = np.array([[1.0], [0.5]])
bk = np.array([[1.0], [-0.5]])
t = 1

print(s_function(t,ak,bk,a0))
#Plots source function over [0,2pi]
t = np.linspace(0,0.5,100)
plt.plot(t,list(map(lambda t: s_function(t,ak,bk,a0),t)))
# plt.show() replaced with savefig to avoid webview error
plt.savefig('source_function_plot.png')
plt.close()
T=10
Nt=10
Nx=50
Lx=5
# Constants, now implmenting wind as Fourier sum
physical_constants = {
    'RHO_CH4': 0.656, # kg/m^3, density of methane at 25 deg C and 1 atm
    'U': 5.0,         # m/s, wind speed
    'wind_vector': np.array([0,1]),
    'SIGMA_H': 10.0,  # m, horizontal dispersion coefficient
    'SIGMA_V': 10.0,  # m, vertical dispersion coefficient
    'N_REFL': 5,      # number of reflections
    'P': 1000.0,      # m, PBL height
    'XS': [0.0, 2.0],       # m, source x-coordinate (multiple sources)
    'YS': [0.0, -2.0],       # m, source y-coordinate (multiple sources)
    'ZS': 0,       # m, source height
    'Z': 0,         # m, sensor height
    'a_H': 1,
    'b_H': 1,
    'w': 1,
    'a_V': 1,
    'b_V': 1,
    'h': 1,
    'gamma_H': 1,
    'gamma_V': 1,
    'wa0': np.array([0.0, 5.0]), # Mean wind vector
    'wa': [np.array([1.0, 0.0])], # Fourier coefficients for wind (j=1)
    'wb': [np.array([0.0, 1.0])]
}
#%%
beta = 1
sigma_epsilon = 0.01
# Define model class y(t,x)=A(x)s(t)+beta+epsilon
model = Model(beta, sigma_epsilon, s_function, physical_constants)
t_start = 0

#%%
#Plots over spatial grid for fixed time t
data = model.gen_data(T,Nt,Nx,Lx,ak,bk,a0)
plt.contourf(data['X1'],data['X2'],data['Y'][0])
plt.colorbar()
# plt.show() replaced with savefig to avoid webview error
plt.savefig('source_function_plot.png')
plt.close()
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
    # plt.show() replaced with savefig to avoid webview error
plt.savefig('source_function_plot.png')
plt.close()




# Function for RWMHS given starting point, variance, time steps and posterior


#%%
# Generate data  on grid
# Generate data  on grid
data = model.gen_data(10,Nt,Nx,5,ak,bk,a0)
#Test log_prior
coefficients = [a0,ak,bk]
print(log_prior_coefficients(coefficients))
#y=As+beta+epsilon
# %%

# Test log_likelihood_y
ll = model.log_likelihood_y(coefficients,T,Nt,Nx,data)
print(f"Log likelihood: {ll}")



#Run rwmh
# 2 sources, 1 coeff each -> 2*1 + 2*1 + 2 = 6 parameters
initial_point = np.zeros(6) 
# Need to reshape inside log_posterior? 
# No, log_likelihood_y expects [a0, ak, bk].
# But rwmh passes a flat array.
# We need a wrapper to reshape the flat array into [a0, ak, bk] structure expected by model.
# Wait, model.log_likelihood_y expects `coeff` list: [a0, ak, bk].
# If we pass flat array to rwmh, log_posterior receives flat array.
# We need to restructure it.

def log_posterior(flat_coeff):
    # Reshape flat_coeff to [a0, ak, bk]
    # Structure: a0 (2,), ak (2,1), bk (2,1)
    # Total 6 params.
    # Order in flat array: a0_0, a0_1, ak_0, ak_1, bk_0, bk_1?
    # Or a0 (2), ak (2), bk (2).
    n_sources = 2
    n_coeff = 1
    
    a0_extracted = flat_coeff[:n_sources]
    ak_extracted = flat_coeff[n_sources:n_sources + n_sources*n_coeff].reshape(n_sources, n_coeff)
    bk_extracted = flat_coeff[n_sources + n_sources*n_coeff:].reshape(n_sources, n_coeff)
    
    coeff_list = [a0_extracted, ak_extracted, bk_extracted]
    
    return log_prior_coefficients(coeff_list) + model.log_likelihood_y(coeff_list,T,Nt,Nx,data)

chain,acceptance_rate = rwmh(initial_point,0.01,10000,log_posterior)
print(chain)
print(acceptance_rate)

# Plot chains
plt.figure(figsize=(12, 12))
labels = ['a0', 'ak', 'bk']
# True values flattened in same order
true_values_flat = np.concatenate([a0.flatten(), ak.flatten(), bk.flatten()])

# Plot all 6 parameters
for i in range(6):
    plt.subplot(6, 1, i+1)
    plt.plot(chain[:, i], label='Chain')
    plt.axhline(true_values_flat[i], color='r', linestyle='--', label='True Value')
    plt.ylabel(f'Param {i}')
    plt.legend()

plt.xlabel('Iteration')
plt.suptitle(f'MCMC Chains (Acceptance Rate: {acceptance_rate:.2f})')
plt.savefig('mcmc_chains.png')
plt.close()

# Print statistics
burn_in = 5000
print("Posterior Means (last 5000 samples):")
print(f"Means: {np.mean(chain[burn_in:], axis=0)}")
print(f"True: {true_values_flat}")
