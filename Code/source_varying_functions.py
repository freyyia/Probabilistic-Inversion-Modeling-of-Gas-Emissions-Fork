import numpy as np
from math import cos, sin



def wind_function(t, wa0, wa, wb):
    wind = np.array(wa0)
    for k in range(len(wa)):
        wind += wa[k] * cos(2 * np.pi * (k + 1) * t) + wb[k] * sin(2 * np.pi * (k + 1) * t)
    return wind

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





def A_matrix(x_sensor, y_sensor, constants, wind_vector=None):
    """
    Calculates the concentration using the Gaussian plume formula with reflections.
    Supports both scalar and vectorized (grid) inputs for x_sensor and y_sensor.
    """
    # 1. Extract Geometry & Wind
    XS, YS, ZS = constants['XS'], constants['YS'], constants['ZS'] # Source
    Z = constants['Z']  # Sensor Height
    
    if wind_vector is not None:
        U_speed = np.linalg.norm(wind_vector)
        if U_speed > 1e-6:
            u_vec = wind_vector / U_speed
        else:
            # Handle zero wind case if necessary, or assume small non-zero
            u_vec = np.array([1.0, 0.0]) 
    else:
        u_vec = np.array(constants['wind_vector'])
        u_vec = u_vec / np.linalg.norm(u_vec) # Ensure unit vector
        U_speed = constants['U']
    
    # 2. Coordinate Rotation
    # Ensure inputs are arrays for uniform processing
    x_in = np.atleast_1d(x_sensor)
    y_in = np.atleast_1d(y_sensor)
    
    dx = x_in - XS
    dy = y_in - YS
    
    wind_vec_perp = np.array([-u_vec[1], u_vec[0]]) 
    
    # Project to get Downwind (dist_R) and Crosswind (dist_H) distances
    dist_R = dx * u_vec[0] + dy * u_vec[1]
    dist_H = dx * wind_vec_perp[0] + dy * wind_vec_perp[1]
    dist_V = Z - ZS 

    a_H, b_H = constants['a_H'], constants['b_H']
    a_V, b_V = constants['a_V'], constants['b_V']
    w, h = constants['w'], constants['h']
    gamma_H = constants['gamma_H']
    gamma_V = constants['gamma_V']

    # Mask for downwind distances
    valid_mask = dist_R > 0.1
    
    # Initialize result array
    result = np.zeros_like(dist_R)
    
    # Extract valid elements
    dist_R_valid = dist_R[valid_mask]
    dist_H_valid = dist_H[valid_mask]
    
    if dist_R_valid.size > 0:
        sigma_H = a_H * (dist_R_valid * np.tan(gamma_H))**b_H + w
        sigma_V = a_V * (dist_R_valid * np.tan(gamma_V))**b_V + h
        
        rho_ch4 = constants['RHO_CH4']
        pre_factor = (100 / rho_ch4) / (2 * np.pi * U_speed * sigma_H * sigma_V)
        
        term_horizontal = np.exp(-(dist_H_valid**2) / (2 * sigma_H**2))
        term_vertical_base = np.exp(-(dist_V**2) / (2 * sigma_V**2)) 
        
        N_REFL = constants['N_REFL']
        P = constants['P']
        H = ZS 
        
        sum_refl = np.zeros_like(sigma_V)
        for j in range(1, N_REFL + 1):
            k1 = 2 * np.floor((j + 1) / 2) * P
            num1 = (k1 + ((-1)**j * Z) - H)**2
            exp1 = np.exp(-num1 / (2 * sigma_V**2))
            
            k2 = 2 * np.floor(j / 2) * P
            num2 = (k2 + ((-1)**(j - 1) * Z) + H)**2
            exp2 = np.exp(-num2 / (2 * sigma_V**2))
            
            sum_refl += (exp1 + exp2)

        term_vertical_total = term_vertical_base + sum_refl
        
        # Assign back to result
        result[valid_mask] = pre_factor * term_horizontal * term_vertical_total
    
    # Return scalar if input was scalar
    if np.isscalar(x_sensor) and np.isscalar(y_sensor):
        return result[0]
    else:
        return result.reshape(np.shape(x_sensor))



class Model:
    def __init__(self,beta,sigma_epsilon,s_function, physical_constants):
        self.beta = beta
        self.sigma_epsilon = sigma_epsilon
        self.s_function = s_function
        self.physical_constants = physical_constants
    # Observation at (x_1,x_2) at time t
    def y(self,x_1,x_2,t,ak,bk,a0):
        wa0 = self.physical_constants.get('wa0')
        wa = self.physical_constants.get('wa')
        wb = self.physical_constants.get('wb')
        if wa0 is not None and wa is not None and wb is not None:
            current_wind = wind_function(t, wa0, wa, wb)
            A = A_matrix(x_1, x_2, self.physical_constants, wind_vector=current_wind)
        else:
            A = A_matrix(x_1, x_2, self.physical_constants)
        return A * self.s_function(t, ak, bk, a0) + self.beta + np.random.normal(0, self.sigma_epsilon)
    # Generate data at Nt time steps
    def gen_data(self,T,Nt,Nx,Lx,ak,bk,a0):
        x_1 = np.linspace(-Lx, Lx, Nx)
        x_2 = np.linspace(-Lx, Lx, Nx)
        X_1, X_2 = np.meshgrid(x_1, x_2)
        Y_list = []
        for t in np.linspace(0,T,Nt):
            Yt = np.array([self.y(x_1, x_2, t,ak,bk,a0) for x_1, x_2 in zip(X_1.flatten(), X_2.flatten())]).reshape(X_1.shape)
            Y_list.append(Yt)
        Y = np.array(Y_list)
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
            # Calculate expected value mu(x, t)
            wa0 = self.physical_constants.get('wa0')
            wa = self.physical_constants.get('wa')
            wb = self.physical_constants.get('wb')
            if wa0 is not None and wa is not None and wb is not None:
                current_wind = wind_function(t, wa0, wa, wb)
                A = A_matrix(data['X1'], data['X2'], self.physical_constants, wind_vector=current_wind)
            else:
                A = A_matrix(data['X1'], data['X2'], self.physical_constants)
                
            mu = A*st + self.beta
            
            # Get observed data for this time step
            y_obs = data_reshaped[i]
            
            # Update log likelihood
            # mu is (Nx, Nx), y_obs is (Nx*Nx,)
            sq_residuals = (y_obs - mu.flatten())**2
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