import numpy as np
from math import cos, sin
from typing import List, Dict, Union, Optional, Tuple, Any

def wind_function(t: Union[float, np.ndarray], wa0: np.ndarray, wa: List[np.ndarray], wb: List[np.ndarray]) -> np.ndarray:
    """
    Calculates the wind vector at time t.
    Vectorized over t.
    
    Args:
        t: Time scalar or array of shape (Nt,)
        wa0: Base wind vector (2,)
        wa: List of cosine coefficients, each (2,)
        wb: List of sine coefficients, each (2,)
        
    Returns:
        Wind vector(s). Shape (2,) if t is scalar, or (Nt, 2) if t is array.
    """
    t = np.atleast_1d(t)
    wind = np.zeros((len(t), 2)) + wa0
    
    # Vectorized loop over coefficients
    # wa and wb are lists of arrays.
    for k in range(len(wa)):
        # k is 0-indexed, frequency is k+1
        freq = 2 * np.pi * (k + 1)
        
        # cos_vals: (Nt, 1)
        cos_vals = np.cos(freq * t)[:, None]
        sin_vals = np.sin(freq * t)[:, None]
        
        # wa[k] is (2,)
        # wind += wa[k] * cos + wb[k] * sin
        wind += wa[k] * cos_vals + wb[k] * sin_vals
        
    if wind.shape[0] == 1:
        return wind[0]
    return wind

def s_function(t: Union[float, np.ndarray], ak: np.ndarray, bk: np.ndarray, a0: np.ndarray) -> np.ndarray:
    """
    Source function s(t) using Fourier series and Softplus for positivity.
    s(t) = ln(1 + exp( Fourier(t) ))
    Uses np.logaddexp(0, x) for numerical stability.
    
    Args:
        t: Time scalar or array (Nt,)
        ak: Cosine coeffs. (N_sources, K) or (K,)
        bk: Sine coeffs. (N_sources, K) or (K,)
        a0: Bias. (N_sources,) or (1,)
        
    Returns:
        Source intensity. Shape (N_sources,) if t scalar, or (Nt, N_sources) if t array.
    """
    t = np.atleast_1d(t) # (Nt,)
    ak = np.atleast_1d(ak)
    bk = np.atleast_1d(bk)
    a0 = np.atleast_1d(a0)
    
    # Standardize ak, bk to (N_sources, K)
    if ak.ndim == 1:
        ak = ak[None, :] # (1, K)
        bk = bk[None, :]
        a0 = a0 # (1,)
    
    N_sources, K = ak.shape
    Nt = len(t)
    
    # Frequencies: 1 to K
    ks = np.arange(1, K + 1)
    freqs = 2 * np.pi * ks # (K,)
    
    # Time-Frequency grid: (Nt, K)
    # t[:, None] * freqs[None, :]
    tf = t[:, None] * freqs[None, :]
    
    cos_mat = np.cos(tf) # (Nt, K)
    sin_mat = np.sin(tf) # (Nt, K)
    
    # Linear predictor: (Nt, N_sources)
    # sum_k (ak * cos + bk * sin)
    # ak: (N_sources, K)
    # cos_mat: (Nt, K)
    # We want result (Nt, N_sources)
    # Einstein summation: 'nk,tk->tn'
    linear_pred = np.einsum('sk,tk->ts', ak, cos_mat) + np.einsum('sk,tk->ts', bk, sin_mat)
    
    # Add bias a0 (N_sources,)
    linear_pred += a0[None, :]
    
    # Softplus with stability
    # log(1 + exp(x)) = logaddexp(0, x)
    result = np.logaddexp(0, linear_pred)
    
    if Nt == 1:
        return result[0] # (N_sources,)
    return result # (Nt, N_sources)

def A_matrix(x_sensor: np.ndarray, y_sensor: np.ndarray, constants: Dict, wind_vector: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Calculates the concentration using the Gaussian plume formula.
    Supports vectorization over time if wind_vector is (Nt, 2).
    
    Args:
        x_sensor: (N_pixels,)
        y_sensor: (N_pixels,)
        constants: Physics constants
        wind_vector: (2,) or (Nt, 2)
        
    Returns:
        Concentration matrix. 
        If wind_vector is (2,), returns (N_sources, N_pixels).
        If wind_vector is (Nt, 2), returns (Nt, N_sources, N_pixels).
    """
    XS = np.atleast_1d(constants['XS']) # (N_sources,)
    YS = np.atleast_1d(constants['YS'])
    ZS = constants['ZS']
    Z = constants['Z']
    
    # Handle wind vector
    if wind_vector is None:
        u_vec = np.array([[1.0, 0.0]]) # (1, 2)
        U_speed = np.array([constants['U']])
    else:
        wind_vector = np.atleast_2d(wind_vector) # (Nt, 2) or (1, 2)
        U_speed = np.linalg.norm(wind_vector, axis=1) # (Nt,)
        # Avoid division by zero
        mask_zero = U_speed < 1e-6
        U_speed[mask_zero] = 1.0 # Dummy value
        u_vec = wind_vector / U_speed[:, None]
        u_vec[mask_zero] = np.array([1.0, 0.0])
        
    Nt = u_vec.shape[0]
    
    x_in = np.atleast_1d(x_sensor).flatten() # (N_pixels,)
    y_in = np.atleast_1d(y_sensor).flatten()
    N_pixels = len(x_in)
    N_sources = len(XS)
    
    # dx, dy: (N_sources, N_pixels)
    dx = x_in[None, :] - XS[:, None]
    dy = y_in[None, :] - YS[:, None]
    
    # Expand for time: (1, N_sources, N_pixels)
    dx = dx[None, :, :]
    dy = dy[None, :, :]
    
    # u_vec: (Nt, 2) -> (Nt, 1, 1, 2) components
    ux = u_vec[:, 0][:, None, None]
    uy = u_vec[:, 1][:, None, None]
    
    # Perpendicular wind: (-uy, ux)
    vx = -uy
    vy = ux
    
    # Projections
    # dist_R: (Nt, N_sources, N_pixels)
    dist_R = dx * ux + dy * uy
    dist_H = dx * vx + dy * vy
    dist_V = Z - ZS
    
    # Mask
    valid_mask = dist_R > 0.1
    
    result = np.zeros_like(dist_R)
    
    # We can't easily use boolean indexing for assignment if we want to keep shape.
    # Instead, use where or calculation on all (with safe ops).
    # For efficiency with mask, we might need to loop or use masked arrays.
    # Given the complexity of the formula, let's calculate on valid points only?
    # But valid points differ per time/source/pixel.
    # Let's use np.where to avoid invalid power ops.
    
    dR_safe = np.where(valid_mask, dist_R, 1.0) # Avoid <= 0 for power
    
    a_H, b_H = constants['a_H'], constants['b_H']
    a_V, b_V = constants['a_V'], constants['b_V']
    w, h = constants['w'], constants['h']
    gamma_H, gamma_V = constants['gamma_H'], constants['gamma_V']
    
    sigma_H = a_H * (dR_safe * np.tan(gamma_H))**b_H + w
    sigma_V = a_V * (dR_safe * np.tan(gamma_V))**b_V + h
    
    # Pre-factor
    # U_speed is (Nt,). Expand to (Nt, 1, 1)
    U_exp = U_speed[:, None, None]
    pre_factor = (100 / constants['RHO_CH4']) / (2 * np.pi * U_exp * sigma_H * sigma_V)
    
    term_H = np.exp(-(dist_H**2) / (2 * sigma_H**2))
    term_V_base = np.exp(-(dist_V**2) / (2 * sigma_V**2))
    
    # Reflections
    N_REFL = constants.get('N_REFL', 0) # Default 0 if not present? Usually 5?
    # Check if N_REFL is in constants, if not assume 0 or check code
    # presentation_plots doesn't define N_REFL in consts dict!
    # But source_varying_functions used it.
    # Wait, presentation_plots consts: 'a_H', 'b_H', etc. No N_REFL.
    # But A_matrix in source_varying_functions used it.
    # Let's assume N_REFL=0 if not in constants, or use a default.
    # presentation_plots A_matrix implementation didn't have reflections loop?
    # Let's check presentation_plots A_matrix again.
    # It DOES NOT have the reflection loop.
    # source_varying_functions A_matrix DOES have it.
    # I should probably keep the reflection loop but default N_REFL to 0 if not provided, to match presentation_plots behavior if needed, or better yet, improve it.
    # But wait, presentation_plots A_matrix:
    # term_V = np.exp(-(dist_V**2) / (2 * sigma_V**2))
    # No reflection loop.
    # source_varying_functions A_matrix:
    # Has reflection loop.
    # I will include the reflection loop but make it optional based on constants.
    
    term_V_total = term_V_base
    if 'N_REFL' in constants:
        N_REFL = constants['N_REFL']
        P = constants['P']
        H_height = ZS
        sum_refl = np.zeros_like(sigma_V)
        for j in range(1, N_REFL + 1):
             k1 = 2 * np.floor((j + 1) / 2) * P
             num1 = (k1 + ((-1)**j * Z) - H_height)**2
             exp1 = np.exp(-num1 / (2 * sigma_V**2))
             
             k2 = 2 * np.floor(j / 2) * P
             num2 = (k2 + ((-1)**(j - 1) * Z) + H_height)**2
             exp2 = np.exp(-num2 / (2 * sigma_V**2))
             sum_refl += (exp1 + exp2)
        term_V_total += sum_refl
        
    conc = pre_factor * term_H * term_V_total
    
    # Apply mask
    result = np.where(valid_mask, conc, 0.0)
    
    # Squeeze if single time step
    if result.shape[0] == 1:
        result = result[0] # (N_sources, N_pixels)
        
    return result

def log_prior_coefficients(coeff: List[np.ndarray]) -> float:
    a0 = np.array(coeff[0])
    ak = np.array(coeff[1])
    bk = np.array(coeff[2])
    
    # Variance depends on k index. 
    if ak.ndim == 2:
        n_coeff = ak.shape[1]
        variance_k = np.array([1/(1+(k+1)**2) for k in range(n_coeff)])
        variance_matrix = np.tile(variance_k, (ak.shape[0], 1))
        term_ak = np.sum(ak**2 / variance_matrix)
        term_bk = np.sum(bk**2 / variance_matrix)
    else:
        n_coeff = len(ak)
        variance_k = np.array([1/(1+(k+1)**2) for k in range(n_coeff)])
        term_ak = np.sum(ak**2 / variance_k)
        term_bk = np.sum(bk**2 / variance_k)
        
    return -0.5 * term_ak - 0.5 * term_bk - 0.5 * np.sum(a0**2)

def rwmh(start_point: np.ndarray, proposal_variance: float, n_steps: int, log_posterior: Any) -> Tuple[np.ndarray, float]:
    current_point = np.array(start_point)
    current_log_prob = log_posterior(current_point)
    
    chain = [current_point]
    accepted_count = 0
    
    for _ in range(n_steps):
        proposal = current_point + np.random.normal(0, np.sqrt(proposal_variance), size=current_point.shape)
        proposal_log_prob = log_posterior(proposal)
        
        if proposal_log_prob == -np.inf:
            acceptance_ratio = -np.inf
        else:
            acceptance_ratio = proposal_log_prob - current_log_prob
        
        if np.log(np.random.rand()) < acceptance_ratio:
            current_point = proposal
            current_log_prob = proposal_log_prob
            accepted_count += 1
            
        chain.append(current_point)
        
    acceptance_rate = accepted_count / n_steps
    return np.array(chain), acceptance_rate

class Model:
    def __init__(self, beta: float, sigma_epsilon: float, physical_constants: Dict):
        self.beta = beta
        self.sigma_epsilon = sigma_epsilon
        self.consts = physical_constants

    def gen_data(self, T: float, Nt: int, Nx: int, Lx: float, ak: np.ndarray, bk: np.ndarray, a0: np.ndarray) -> Dict:
        x = np.linspace(-Lx, Lx, Nx)
        y = np.linspace(-Lx, Lx, Nx)
        X, Y = np.meshgrid(x, y)
        X_flat, Y_flat = X.flatten(), Y.flatten()
        
        times = np.linspace(0, T, Nt)
        
        # Vectorized generation
        # Wind: (Nt, 2)
        winds = wind_function(times, self.consts['wa0'], self.consts['wa'], self.consts['wb'])
        
        # A: (Nt, N_sources, N_pixels)
        A = A_matrix(X_flat, Y_flat, self.consts, wind_vector=winds)
        
        # st: (Nt, N_sources)
        st = s_function(times, ak, bk, a0)
        
        # mu: (Nt, N_pixels)
        # sum_sources (A * st)
        # A: (Nt, N_sources, N_pixels)
        # st: (Nt, N_sources) -> (Nt, N_sources, 1)
        st_exp = st[:, :, None]
        mu = np.sum(A * st_exp, axis=1) + self.beta
        
        # Add noise
        obs = mu + np.random.normal(0, self.sigma_epsilon, size=mu.shape)
        
        return {
            'Y': obs, # (Nt, N_pixels)
            'X1': X, 'X2': Y, 
            'times': times, 
            'true_s': st,
            'X_flat': X_flat, 'Y_flat': Y_flat
        }

    def log_likelihood(self, params_dict: Dict, data: Dict) -> float:
        Y_obs = data['Y'] # (Nt, N_pixels)
        times = data['times']
        X1_f, X2_f = data['X_flat'], data['Y_flat']
        
        p_a0 = params_dict['a0']
        p_ak = params_dict.get('ak', np.zeros_like(p_a0))
        p_bk = params_dict.get('bk', np.zeros_like(p_a0))
        p_XS = params_dict['XS']
        p_YS = params_dict['YS']
        
        # Bounds check
        if np.any(np.abs(p_XS) > 6.0) or np.any(np.abs(p_YS) > 6.0):
            return -np.inf

        current_consts = self.consts.copy()
        current_consts['XS'] = p_XS
        current_consts['YS'] = p_YS
        
        var = self.sigma_epsilon**2
        
        # Vectorized calculation
        # 1. Winds (Nt, 2)
        winds = wind_function(times, current_consts['wa0'], current_consts['wa'], current_consts['wb'])
        
        # 2. A matrix (Nt, N_sources, N_pixels)
        A = A_matrix(X1_f, X2_f, current_consts, wind_vector=winds)
        
        # 3. Source term (Nt, N_sources)
        st = s_function(times, p_ak, p_bk, p_a0)
        
        # 4. Expected mu (Nt, N_pixels)
        st_exp = st[:, :, None]
        mu = np.sum(A * st_exp, axis=1) + self.beta
        
        # 5. Residuals
        resid = Y_obs - mu
        log_ll = -0.5 * np.sum(resid**2) / var
            
        return log_ll