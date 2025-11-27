import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from source_varying_functions import s_function, A_matrix, rwmh, Model, log_prior_coefficients
import numpy as np
import matplotlib.pyplot as plt

# True parameters
a0_true = 4
ak_true = np.array([1])
bk_true = np.array([1])
true_params = [a0_true, ak_true, bk_true]

# Simulation settings
T = 10
Nt = 10
Nx = 20
Lx = 5

# Constants
physical_constants = {
    'RHO_CH4': 0.656,
    'U': 5.0,
    'wind_vector': np.array([0,1]),
    'SIGMA_H': 10.0,
    'SIGMA_V': 10.0,
    'N_REFL': 5,
    'P': 1000.0,
    'XS': 0.0,
    'YS': 0.0,
    'ZS': 0,
    'Z': 0,
    'a_H': 1,
    'b_H': 1,
    'w': 1,
    'a_V': 1,
    'b_V': 1,
    'h': 1,
    'gamma_H': 1,
    'gamma_V': 1
}

beta = 1
sigma_epsilon = 0.01
model = Model(beta, sigma_epsilon, s_function, physical_constants)

# Generate data
print("Generating data...")
data = model.gen_data(T, Nt, Nx, Lx, ak_true, bk_true, a0_true)

# Helper to structure coeffs
def structure_coeffs(flat_coeff):
    a0 = flat_coeff[0]
    ak = np.array([flat_coeff[1]])
    bk = np.array([flat_coeff[2]])
    return [a0, ak, bk]

# Define log posterior
def log_posterior(coeff):
    structured_coeff = structure_coeffs(coeff)
    lp = log_prior_coefficients(structured_coeff)
    ll = model.log_likelihood_y(structured_coeff, T, Nt, Nx, data)
    return lp + ll

# Initial point
initial_point = np.array([0.0, 0.0, 0.0]) 

# Run MCMC
n_steps = 10000
proposal_variance = 0.01

print(f"Running MCMC for {n_steps} steps...")
chain, acceptance_rate = rwmh(initial_point, proposal_variance, n_steps, log_posterior)

print(f"Acceptance rate: {acceptance_rate}")
final_point = chain[-1]
print(f"Final sample: {final_point}")

# Analysis
true_flat = np.array([a0_true, ak_true[0], bk_true[0]])
structured_true = structure_coeffs(true_flat)
structured_final = structure_coeffs(final_point)

ll_true = model.log_likelihood_y(structured_true, T, Nt, Nx, data)
lp_true = log_prior_coefficients(structured_true)

ll_final = model.log_likelihood_y(structured_final, T, Nt, Nx, data)
lp_final = log_prior_coefficients(structured_final)

print("-" * 30)
print(f"True Params: {true_flat}")
print(f"Log Likelihood at Truth: {ll_true}")
print(f"Log Prior at Truth: {lp_true}")
print(f"Total Log Posterior at Truth: {ll_true + lp_true}")

print("-" * 30)
print(f"Final Params: {final_point}")
print(f"Log Likelihood at Final: {ll_final}")
print(f"Log Prior at Final: {lp_final}")
print(f"Total Log Posterior at Final: {ll_final + lp_final}")
print("-" * 30)

# Check if data generation matches likelihood expectation
# Re-calculate residuals at truth manually
data_reshaped = data['Y'].reshape(Nt, -1)
log_likelihood_check = 0
var = sigma_epsilon**2
times = np.linspace(0, T, Nt)
for i, t in enumerate(times):
    st = s_function(t, ak_true, bk_true, a0_true)
    mu = A_matrix(data['X1'], data['X2'], physical_constants)*st + beta
    y_obs = data_reshaped[i]
    sq_residuals = (y_obs - mu.flatten())**2
    log_likelihood_check += -0.5 * np.sum(sq_residuals) / var

print(f"Manual Log Likelihood Check at Truth: {log_likelihood_check}")
