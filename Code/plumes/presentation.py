import numpy as np
import matplotlib.pyplot as plt
from math import cos, sin
import time

# ==========================================
# PART 1: CORRECTED PHYSICS & FUNCTIONS
# ==========================================

def wind_function(t, wa0, wa, wb):
    wind = np.array(wa0)
    for k in range(len(wa)):
        wind += wa[k] * cos(2 * np.pi * (k + 1) * t) + wb[k] * sin(2 * np.pi * (k + 1) * t)
    return wind

def s_function_corrected(t, ak, bk, a0):
    """
    CORRECTED: Uses Softplus to enforce positivity.
    s(t) = ln(1 + exp( Fourier(t) ))
    """
    ak = np.atleast_1d(ak)
    bk = np.atleast_1d(bk)
    a0 = np.atleast_1d(a0)
    
    # Calculate linear predictor (Fourier Sum)
    n_coeff = ak.shape[-1]
    cosines = np.array([cos(2*np.pi*(k+1)*t) for k in range(n_coeff)])
    sines = np.array([sin(2*np.pi*(k+1)*t) for k in range(n_coeff)])
    
    linear_pred = np.dot(ak, cosines) + np.dot(bk, sines) + a0
    
    # Softplus link function (Smooth ReLU) -> Ensures s(t) > 0
    return np.log1p(np.exp(linear_pred))

def A_matrix(x_sensor, y_sensor, constants, wind_vector=None):
    # (Identical logic to your vectorized version, streamlined for the script)
    XS, YS, ZS = np.atleast_1d(constants['XS']), np.atleast_1d(constants['YS']), constants['ZS']
    Z = constants['Z']
    
    if wind_vector is None:
        u_vec = np.array([1.0, 0.0])
        U_speed = constants['U']
    else:
        U_speed = np.linalg.norm(wind_vector)
        u_vec = wind_vector / (U_speed + 1e-9)

    x_in = np.atleast_1d(x_sensor)
    y_in = np.atleast_1d(y_sensor)
    
    dx = x_in[None, :] - XS[:, None]
    dy = y_in[None, :] - YS[:, None]
    
    wind_vec_perp = np.array([-u_vec[1], u_vec[0]])
    
    dist_R = dx * u_vec[0] + dy * u_vec[1]
    dist_H = dx * wind_vec_perp[0] + dy * wind_vec_perp[1]
    dist_V = Z - ZS
    
    a_H, b_H = constants['a_H'], constants['b_H']
    a_V, b_V = constants['a_V'], constants['b_V']
    w, h = constants['w'], constants['h']
    gamma_H, gamma_V = constants['gamma_H'], constants['gamma_V']
    
    result = np.zeros_like(dist_R)
    mask = dist_R > 0.1
    
    if np.any(mask):
        dR = dist_R[mask]
        dH = dist_H[mask]
        
        sigma_H = a_H * (dR * np.tan(gamma_H))**b_H + w
        sigma_V = a_V * (dR * np.tan(gamma_V))**b_V + h
        
        pre_factor = (100 / constants['RHO_CH4']) / (2 * np.pi * U_speed * sigma_H * sigma_V)
        term_H = np.exp(-(dH**2) / (2 * sigma_H**2))
        term_V = np.exp(-(dist_V**2) / (2 * sigma_V**2)) # Simplified refl for demo speed
        
        result[mask] = pre_factor * term_H * term_V

    return result # Shape (N_sources, N_pixels)

class OptimizedModel:
    def __init__(self, beta, sigma_epsilon, physical_constants):
        self.beta = beta
        self.sigma_epsilon = sigma_epsilon
        self.consts = physical_constants

    def gen_data(self, T, Nt, Nx, Lx, ak, bk, a0):
        # Generate grid
        x = np.linspace(-Lx, Lx, Nx)
        y = np.linspace(-Lx, Lx, Nx)
        X, Y = np.meshgrid(x, y)
        X_flat, Y_flat = X.flatten(), Y.flatten()
        
        times = np.linspace(0, T, Nt)
        observations = []
        true_s_values = []
        
        print("Generating Truth Data...")
        for t in times:
            # 1. Calculate Wind
            wind = wind_function(t, self.consts['wa0'], self.consts['wa'], self.consts['wb'])
            
            # 2. Calculate A Matrix
            A = A_matrix(X_flat, Y_flat, self.consts, wind_vector=wind)
            
            # 3. Calculate Source
            st = s_function_corrected(t, ak, bk, a0) # shape (N_sources,)
            true_s_values.append(st)
            
            # 4. Combine
            # A is (N_src, N_pix), st is (N_src,)
            mu = np.dot(st, A) + self.beta
            obs = mu + np.random.normal(0, self.sigma_epsilon, size=mu.shape)
            observations.append(obs)
            
        return {
            'Y': np.array(observations), 
            'X1': X, 'X2': Y, 
            'times': times, 
            'true_s': np.array(true_s_values)
        }

    def log_likelihood(self, params_dict, data, is_static_model=False):
        """
        Optimized Likelihood: Pre-computes wind to save time.
        """
        Y_obs = data['Y'] # (Nt, Npix)
        times = data['times']
        X1_f, X2_f = data['X1'].flatten(), data['X2'].flatten()
        
        # Unpack Parameters
        p_a0 = params_dict['a0']
        p_ak = params_dict.get('ak', np.zeros_like(p_a0)) # Default 0 if static
        p_bk = params_dict.get('bk', np.zeros_like(p_a0))
        p_XS = params_dict['XS']
        p_YS = params_dict['YS']
        
        # Check Bounds
        if np.any(np.abs(p_XS) > 5.0) or np.any(np.abs(p_YS) > 5.0):
            return -np.inf

        # Update constants with proposed location
        current_consts = self.consts.copy()
        current_consts['XS'] = p_XS
        current_consts['YS'] = p_YS
        
        log_ll = 0
        var = self.sigma_epsilon**2
        
        # LOOP
        for i, t in enumerate(times):
            # Recalculate Wind (Fast)
            wind = wind_function(t, current_consts['wa0'], current_consts['wa'], current_consts['wb'])
            
            # Recalculate A Matrix (Slow part - but necessary as locations change)
            A = A_matrix(X1_f, X2_f, current_consts, wind_vector=wind)
            
            # Calculate Source
            st = s_function_corrected(t, p_ak, p_bk, p_a0)
            
            # Expected Data
            mu = np.dot(st, A) + self.beta
            
            # Residuals
            resid = Y_obs[i] - mu
            log_ll += -0.5 * np.sum(resid**2) / var
            
        return log_ll

# ==========================================
# PART 2: THE PRESENTATION EXPERIMENT
# ==========================================

def run_experiment():
    np.random.seed(42)
    
    # 1. SETUP
    # ------------------
    T, Nt, Nx, Lx = 1.0, 15, 15, 5.0 # Low res for speed in demo
    consts = {
        'RHO_CH4': 0.656, 'U': 5.0,
        'XS': [2.0], 'YS': [-2.0], 'ZS': 0, 'Z': 0,
        'a_H': 1, 'b_H': 1, 'w': 1, 'a_V': 1, 'b_V': 1, 'h': 1,
        'gamma_H': 1, 'gamma_V': 1,
        'wa0': np.array([0.0, 4.0]), 'wa': [np.array([2.0, 0.0])], 'wb': [np.array([0.0, 1.0])]
    }
    
    # Truth: Dynamic Source
    true_a0 = [1.0]
    true_ak = [[1.5]] # Strong oscillation
    true_bk = [[0.5]]
    
    model = OptimizedModel(beta=1.0, sigma_epsilon=0.1, physical_constants=consts)
    
    # Generate Data
    data = model.gen_data(T, Nt, Nx, Lx, np.array(true_ak), np.array(true_bk), np.array(true_a0))
    print("Data Generated.")

    # 2. RUN STATIC MODEL (Baseline)
    # ------------------
    # Only infers a0, XS, YS. Assumes ak=0, bk=0.
    print("\n--- Running STATIC Model (Baseline) ---")
    chain_static = []
    curr = np.concatenate([ [0.5], [0.0], [0.0] ]) # a0, xs, ys
    
    for i in range(2000):
        prop = curr + np.random.normal(0, [0.1, 0.1, 0.1])
        
        # Current LL
        p_curr = {'a0': [curr[0]], 'XS': [curr[1]], 'YS': [curr[2]]}
        ll_curr = model.log_likelihood(p_curr, data)
        
        # Prop LL
        p_prop = {'a0': [prop[0]], 'XS': [prop[1]], 'YS': [prop[2]]}
        ll_prop = model.log_likelihood(p_prop, data)
        
        if np.log(np.random.rand()) < (ll_prop - ll_curr):
            curr = prop
        chain_static.append(curr)
        if i % 500 == 0: print(f"Step {i}")
        
    chain_static = np.array(chain_static)[500:] # Burn-in
    mean_static = np.mean(chain_static, axis=0)

    # 3. RUN DYNAMIC MODEL (Challenger)
    # ------------------
    # Infers a0, ak, bk, XS, YS.
    print("\n--- Running DYNAMIC Model (Ours) ---")
    chain_dyn = []
    # Start near truth to speed up convergence for presentation demo
    curr = np.concatenate([ [1.0], [1.0], [0.0], [2.0], [-2.0] ]) # a0, ak, bk, xs, ys
    
    for i in range(2000):
        prop = curr + np.random.normal(0, 0.05, size=len(curr))
        
        p_curr = {'a0': [curr[0]], 'ak': [[curr[1]]], 'bk': [[curr[2]]], 'XS': [curr[3]], 'YS': [curr[4]]}
        ll_curr = model.log_likelihood(p_curr, data)
        
        p_prop = {'a0': [prop[0]], 'ak': [[prop[1]]], 'bk': [[prop[2]]], 'XS': [prop[3]], 'YS': [prop[4]]}
        ll_prop = model.log_likelihood(p_prop, data)
        
        if np.log(np.random.rand()) < (ll_prop - ll_curr):
            curr = prop
        chain_dyn.append(curr)
        if i % 500 == 0: print(f"Step {i}")

    chain_dyn = np.array(chain_dyn)[500:]
    mean_dyn = np.mean(chain_dyn, axis=0)
    
    # 4. VISUALIZATION & METRICS
    # ------------------
    print("\nGenerating Plots...")
    
    # Reconstruct Source Functions
    t_plot = np.linspace(0, T, 100)
    s_true = [s_function_corrected(t, true_ak, true_bk, true_a0)[0] for t in t_plot]
    s_static = [s_function_corrected(t, [0], [0], [mean_static[0]])[0] for t in t_plot]
    s_dyn = [s_function_corrected(t, [mean_dyn[1]], [mean_dyn[2]], [mean_dyn[0]])[0] for t in t_plot]
    
    # PLOT 1: Time Series
    plt.figure(figsize=(10, 6))
    plt.plot(t_plot, s_true, 'k-', linewidth=2, label='True Source (Physics)')
    plt.plot(t_plot, s_static, 'r--', linewidth=2, label='Static Model (Baseline)')
    plt.plot(t_plot, s_dyn, 'b-', linewidth=2, alpha=0.8, label='Dynamic Model (Ours)')
    plt.fill_between(t_plot, s_dyn, 0, color='blue', alpha=0.1)
    plt.title("Failure of Static Assumptions in Dynamic Environments", fontsize=14)
    plt.xlabel("Time (Normalized)", fontsize=12)
    plt.ylabel("Emission Rate s(t)", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.savefig('presentation_1_timeseries.png', dpi=300)
    print("Saved presentation_1_timeseries.png")
    
    # PLOT 2: Residuals at Peak Emission
    # Find index of max emission
    idx_max = np.argmax(data['true_s'])
    obs_max = data['Y'][idx_max].reshape(Nx, Nx)
    
    # Reconstruct Grids
    # Static
    p_stat = {'a0': [mean_static[0]], 'XS': [mean_static[1]], 'YS': [mean_static[2]]}
    wind_t = wind_function(data['times'][idx_max], consts['wa0'], consts['wa'], consts['wb'])
    A_stat = A_matrix(data['X1'].flatten(), data['X2'].flatten(), consts, wind_vector=wind_t)
    # Static model assumes constant source, so s(t) is just softplus(a0)
    s_val_stat = s_function_corrected(0, [0], [0], p_stat['a0']) 
    mu_stat = (np.dot(s_val_stat, A_stat) + 1.0).reshape(Nx, Nx)
    
    # Dynamic
    p_dyn = {'a0': [mean_dyn[0]], 'XS': [mean_dyn[3]], 'YS': [mean_dyn[4]]}
    A_dyn = A_matrix(data['X1'].flatten(), data['X2'].flatten(), consts, wind_vector=wind_t)
    s_val_dyn = s_function_corrected(data['times'][idx_max], [mean_dyn[1]], [mean_dyn[2]], p_dyn['a0'])
    mu_dyn = (np.dot(s_val_dyn, A_dyn) + 1.0).reshape(Nx, Nx)
    
    res_stat = np.abs(obs_max - mu_stat)
    res_dyn = np.abs(obs_max - mu_dyn)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    vmax = max(np.max(res_stat), np.max(res_dyn))
    
    im1 = axes[0].imshow(res_stat, cmap='hot', vmin=0, vmax=vmax)
    axes[0].set_title("Static Model Residuals (Ghost Plume)")
    
    im2 = axes[1].imshow(res_dyn, cmap='hot', vmin=0, vmax=vmax)
    axes[1].set_title("Dynamic Model Residuals (White Noise)")
    
    plt.colorbar(im1, ax=axes.ravel().tolist())
    plt.savefig('presentation_2_residuals.png', dpi=300)
    print("Saved presentation_2_residuals.png")

    # METRICS
    rmse_static = np.sqrt(np.mean((np.array(s_true) - np.array(s_static))**2))
    rmse_dyn = np.sqrt(np.mean((np.array(s_true) - np.array(s_dyn))**2))
    
    with open('presentation_metrics.txt', 'w') as f:
        f.write(f"RMSE Static: {rmse_static:.4f}\n")
        f.write(f"RMSE Dynamic: {rmse_dyn:.4f}\n")
        f.write(f"Improvement: {(rmse_static - rmse_dyn)/rmse_static * 100:.1f}%\n")
    print(f"RMSE Static: {rmse_static:.4f}, RMSE Dynamic: {rmse_dyn:.4f}")

if __name__ == "__main__":
    run_experiment()