import numpy as np
from math import cos, sin



def wind_function(t, wa0, wa, wb):
    wind = np.array(wa0)
    for k in range(len(wa)):
        wind += wa[k] * cos(2 * np.pi * (k + 1) * t) + wb[k] * sin(2 * np.pi * (k + 1) * t)
    return wind

def s_function(t,ak,bk,a0):
    ak = np.array(ak)
    bk = np.array(bk)
    a0 = np.array(a0)
    
    if ak.ndim == 1:
        n_coeff = ak.shape[0]
        cosines = np.array([cos(2*np.pi*(k+1)*t) for k in range(n_coeff)])
        sines = np.array([sin(2*np.pi*(k+1)*t) for k in range(n_coeff)])
        return np.dot(ak,cosines) + np.dot(bk,sines) + a0
    else:
        # ak is (N_sources, n_coeff)
        n_coeff = ak.shape[1]
        cosines = np.array([cos(2*np.pi*(k+1)*t) for k in range(n_coeff)])
        sines = np.array([sin(2*np.pi*(k+1)*t) for k in range(n_coeff)])
        # np.dot(ak, cosines) -> (N_sources,)
        return np.dot(ak, cosines) + np.dot(bk, sines) + a0

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
    
    # Now XS can be a list of x coordinates
    dxlist = []
    dylist = []
    XS = np.atleast_1d(XS)
    YS = np.atleast_1d(YS)
    for i in range(len(XS)):
        dxlist.append(x_in - XS[i])
        dylist.append(y_in - YS[i])
    
    wind_vec_perp = np.array([-u_vec[1], u_vec[0]]) 
    
    # Project to get Downwind (dist_R) and Crosswind (dist_H) distances
    dist_R = np.array(dxlist) * u_vec[0] + np.array(dylist) * u_vec[1]
    dist_H = np.array(dxlist) * wind_vec_perp[0] + np.array(dylist) * wind_vec_perp[1]
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
    
    # Return result. Shape is (N_sources, N_pixels)
    # If input was reshaped, we might want to reshape back, but since we have multiple sources,
    # returning (N_sources, ...) is appropriate.
    # If x_sensor was scalar, result is (N_sources, 1) -> (N_sources,)
    if np.isscalar(x_sensor) and np.isscalar(y_sensor):
        return result.flatten()
    else:
        # If x_sensor was (N,), result is (N_sources, N)
        # If x_sensor was (N, M), result is (N_sources, N*M) because we flattened inputs?
        # Wait, inputs were np.atleast_1d.
        # If x_sensor was grid (N, N), flattened to (N*N).
        # We want to preserve the spatial shape if possible, but with N_sources dimension.
        # Let's return (N_sources, original_shape)
        original_shape = np.shape(x_sensor)
        if len(original_shape) == 0:
             return result.flatten()
        return result.reshape((len(XS),) + original_shape)



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
            
        # A is (N_sources, ...). 
        # st is scalar or (N_sources,)
        
        if np.ndim(self.s_function(t, ak, bk, a0)) > 0:
             # st is vector (independent sources)
             st = self.s_function(t, ak, bk, a0)
             # Reshape st to broadcast against A
             # A shape: (N_sources, ...)
             st_reshaped = st.reshape((len(st),) + (1,) * (A.ndim - 1))
             mu = np.sum(A * st_reshaped, axis=0) + self.beta
        else:
             # st is scalar (shared source function)
             st = self.s_function(t, ak, bk, a0)
             if A.ndim > np.ndim(x_1): # Check if we have extra dimension for sources
                  A_sum = np.sum(A, axis=0)
             else:
                  A_sum = A
             mu = A_sum * st + self.beta
             
        return mu + np.random.normal(0, self.sigma_epsilon)
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
            
            # A is (N_sources, ...). 
            # st is scalar or (N_sources,)
            if np.ndim(st) > 0:
                # st is vector
                st_reshaped = st.reshape((len(st),) + (1,) * (A.ndim - 1))
                mu = np.sum(A * st_reshaped, axis=0) + self.beta
            else:
                # st is scalar
                if A.ndim > data['X1'].ndim:
                    A_sum = np.sum(A, axis=0)
                else:
                    A_sum = A
                mu = A_sum*st + self.beta
            
            # Get observed data for this time step
            y_obs = data_reshaped[i]
            
            # Update log likelihood
            # mu is (Nx, Nx), y_obs is (Nx*Nx,)
            sq_residuals = (y_obs - mu.flatten())**2
            log_likelihood += -0.5 * np.sum(sq_residuals) / var
            
        return log_likelihood



# Define log prior of source coefficients
def log_prior_coefficients(coeff):
    a0 = np.array(coeff[0])
    ak = np.array(coeff[1])
    bk = np.array(coeff[2])
    
    # Flatten to handle both 1D and 2D cases
    ak_flat = ak.flatten()
    bk_flat = bk.flatten()
    a0_flat = a0.flatten()
    
    # Variance depends on k index. 
    # If 2D, ak is (N_sources, n_coeff). k corresponds to column index.
    if ak.ndim == 2:
        n_coeff = ak.shape[1]
        variance_k = np.array([1/(1+(k+1)**2) for k in range(n_coeff)])
        # Broadcast variance_k to (N_sources, n_coeff)
        variance_matrix = np.tile(variance_k, (ak.shape[0], 1))
        term_ak = np.sum(ak**2 / variance_matrix)
        term_bk = np.sum(bk**2 / variance_matrix)
    else:
        n_coeff = len(ak)
        variance_k = np.array([1/(1+(k+1)**2) for k in range(n_coeff)])
        term_ak = np.sum(ak**2 / variance_k)
        term_bk = np.sum(bk**2 / variance_k)
        
    return -0.5 * term_ak - 0.5 * term_bk - 0.5 * np.sum(a0**2)