import numpy as np
from math import cos, sin

def s_function(t,ak,bk,a0):
    n_coeff = ak.shape[0]
    constant = a0
    cosines = [cos(2*np.pi*(k+1)*t) for k in range(n_coeff)]
    sines = [sin(2*np.pi*(k+1)*t) for k in range(n_coeff)]
    return np.dot(ak,cosines) + np.dot(bk,sines) + a0

def A_matrix(x_1s,x_2s,x_1,x_2):
    return np.exp(-((x_1s-x_1)**2+(x_2s-x_2)**2))

def rwmh(start_point, proposal_variance, n_steps, log_posterior):
    """
    Random Walk Metropolis-Hastings Sampler.

    Args:
        start_point (np.array): Starting point for the chain.
        proposal_variance (float or np.array): Variance of the Gaussian proposal distribution.
        n_steps (int): Number of iterations.
        log_posterior (function): Function that returns the log posterior probability of a point.

    Returns:
        chain (np.array): The MCMC chain.
        acceptance_rate (float): The acceptance rate of the chain.
    """
    current_point = np.array(start_point)
    current_log_prob = log_posterior(current_point)
    
    chain = [current_point]
    accepted_count = 0
    
    for _ in range(n_steps):
        # Propose a new point
        proposal = current_point + np.random.normal(0, np.sqrt(proposal_variance), size=current_point.shape)
        proposal_log_prob = log_posterior(proposal)
        
        # Calculate acceptance probability
        # We use log scale for numerical stability
        acceptance_ratio = proposal_log_prob - current_log_prob
        
        # Accept or reject
        if np.log(np.random.rand()) < acceptance_ratio:
            current_point = proposal
            current_log_prob = proposal_log_prob
            accepted_count += 1
            
        chain.append(current_point)
        
    acceptance_rate = accepted_count / n_steps
    return np.array(chain), acceptance_rate


def log_likelihood_y(coeff, data, x_1s, x_2s, beta, sigma_epsilon, A_matrix):
    # Unpack coefficients
    a0 = coeff[0]
    ak = np.atleast_1d(coeff[1])
    bk = np.atleast_1d(coeff[2])
    
    # Grid parameters
    T = 10
    nt = 100
    nx = 100
    
    # Create grids
    times = np.linspace(0, T, nt)
    x_1 = np.linspace(-1, 1, nx)
    x_2 = np.linspace(-1, 1, nx)
    X_1, X_2 = np.meshgrid(x_1, x_2)
    
    # Flatten spatial coordinates
    X1_flat = X_1.flatten()
    X2_flat = X_2.flatten()
    
    # Calculate A matrix (spatial component)
    A = A_matrix(x_1s, x_2s, X1_flat, X2_flat)
    
    # Reshape data to (nt, nx*nx)
    data_reshaped = data.reshape(nt, -1)
    
    log_likelihood = 0
    var = sigma_epsilon**2
    
    # Iterate over time steps
    for i, t in enumerate(times):
        # Calculate source function value at time t
        st = s_function(t, ak, bk, a0)
        
        # Calculate expected value mu(x, t)
        mu = A * st + beta
        
        # Get observed data for this time step
        y_obs = data_reshaped[i]
        
        # Update log likelihood
        sq_residuals = (y_obs - mu)**2
        log_likelihood += -0.5 * np.sum(sq_residuals) / var
        
    return log_likelihood
