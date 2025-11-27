import numpy as np
from math import cos, sin

def s_function(t,ak,bk,a0):
    n_coeff = ak.shape[0]
    constant = a0
    cosines = [cos(2*np.pi*(k+1)*t) for k in range(n_coeff)]
    sines = [sin(2*np.pi*(k+1)*t) for k in range(n_coeff)]
    return np.dot(ak,cosines) + np.dot(bk,sines) + a0

# def A_matrix(x_1s,x_2s,x_1,x_2):
#     return np.exp(-((x_1s-x_1)**2+(x_2s-x_2)**2))

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




def A_matrix(x1, x2, constants):
    """
    Calculates the coupling matrix A using the Gaussian plume formula.
    
    Args:
        x: Sensor x-coordinate
        y: Sensor y-coordinate
        constants: Constants dictionary
        
    Returns:
        Concentration per unit source rate.
    """
    #Extract constants
    RHO_CH4 = constants['RHO_CH4']
    U = constants['U']
    wind_vector = constants['wind_vector']
    SIGMA_H = constants['SIGMA_H']
    SIGMA_V = constants['SIGMA_V']
    N_REFL = constants['N_REFL']
    P = constants['P']
    XS = constants['XS'] #Source x-coordinate
    YS = constants['YS'] #Source y-coordinate
    ZS = constants['ZS'] #Source height
    Z = constants['Z'] #Sensor height
    
    vec = (x1-XS,x2-YS)
    wind_vec_perp = np.array([wind_vector[1],-wind_vector[0]])
    wind_perp_normalized = wind_vec_perp / np.linalg.norm(wind_vec_perp)
    
    delta_V = np.dot(vec,wind_perp_normalized)
    delta_H = Z-ZS

    term1 = (10**6 / RHO_CH4) * (1 / (2 * np.pi * U * SIGMA_H * SIGMA_V)) * np.exp(-delta_H**2 / (2 * SIGMA_H**2))
    expV =np.exp(-delta_V**2 / (2 * SIGMA_V**2))
    
    sum_refl = 0
    for j in range(1, N_REFL + 1):
        # First reflection term
        num1 = (2 * np.floor((j + 1) / 2) * P + (-1)**j * (delta_V + Z) - Z)**2
        exp1 = np.exp(-0.5 * num1 / SIGMA_V**2)
        
        # Second reflection term
        num2 = (2 * np.floor(j / 2) * P + (-1)**(j - 1) * (delta_V + Z) + Z)**2
        exp2 = np.exp(-0.5 * num2 / SIGMA_V**2)
        
        sum_refl += exp1 + exp2
        
    term2 = expV + sum_refl
    
    return term1 * term2



class Model:
    def __init__(self,x_1s,x_2s,beta,sigma_epsilon,s_function, physical_constants):
        self.x_1s = x_1s
        self.x_2s = x_2s
        self.beta = beta
        self.sigma_epsilon = sigma_epsilon
        self.s_function = s_function
        self.A_matrix = A_matrix(self.x_1s,self.x_2s,physical_constants)
    # Observation at (x_1,x_2) at time t
    def y(self,x_1,x_2,t,ak,bk,a0):
        return self.A_matrix*self.s_function(t,ak,bk,a0)+self.beta+np.random.normal(0,self.sigma_epsilon)
    # Generate data at Nt time steps
    def gen_data(self,T,Nt,Nx,Lx,ak,bk,a0):
        x_1 = np.linspace(-Lx, Lx, Nx)
        x_2 = np.linspace(-Lx, Lx, Nx)
        X_1, X_2 = np.meshgrid(x_1, x_2)
        Y = np.array([])
        for t in np.linspace(0,T,Nt):
            Yt = np.array([self.y(x_1, x_2, t,ak,bk,a0) for x_1, x_2 in zip(X_1.flatten(), X_2.flatten())]).reshape(X_1.shape)
            Y = np.append(Y,Yt)
        return {'X1': X_1, 'X2': X_2, 'Y': Y}
        
    # Calculates log_likelihood of data given
    def log_likelihood_y(self,coeff, T, Nt, Nx, data):
        # Unpack coefficients
        a0 = coeff[0]
        ak = np.atleast_1d(coeff[1])
        bk = np.atleast_1d(coeff[2])
        
        # Create grids
        times = np.linspace(0, T, Nt)

        
        # Reshape data to (nt, nx*nx)
        data_reshaped = data['Y'].reshape(Nt, -1)
        
        log_likelihood = 0
        var = self.sigma_epsilon**2
        
        # Iterate over time steps
        for i, t in enumerate(times):
            # Calculate source function value at time t
            st = self.s_function(t, ak, bk, a0)
            
            # Calculate expected value mu(x, t)
            mu = self.A_matrix * st + self.beta
            
            # Get observed data for this time step
            y_obs = data_reshaped[i]
            
            # Update log likelihood
            sq_residuals = (y_obs - mu)**2
            log_likelihood += -0.5 * np.sum(sq_residuals) / var
            
        return log_likelihood



# Define log prior of source coefficients
def log_prior_coefficients(coeff):
    a0 = coeff[0]
    ak = np.atleast_1d(coeff[1])
    bk = np.atleast_1d(coeff[2])
    n_coeff = len(ak)
    variance_k = [1/(1+(k+1)**2) for k in range(n_coeff)]
    return -1/2 * np.sum((ak)**2/variance_k) -1/2 * np.sum((bk)**2/variance_k) -1/2 * (a0)**2