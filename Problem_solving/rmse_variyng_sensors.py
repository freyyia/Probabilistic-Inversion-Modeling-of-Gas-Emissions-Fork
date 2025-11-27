import numpy as np
import matplotlib.pyplot as plt
from math import cos, sin
import time

# ==========================================
# PART 1: PHYSICS & MODEL (Re-used)
# ==========================================

def wind_function(t, wa0, wa, wb):
    wind = np.array(wa0)
    for k in range(len(wa)):
        wind += wa[k] * cos(2 * np.pi * (k + 1) * t) + wb[k] * sin(2 * np.pi * (k + 1) * t)
    return wind

def s_function_corrected(t, ak, bk, a0):
    ak = np.atleast_1d(ak)
    bk = np.atleast_1d(bk)
    a0 = np.atleast_1d(a0)
    n_coeff = ak.shape[-1]
    cosines = np.array([cos(2*np.pi*(k+1)*t) for k in range(n_coeff)])
    sines = np.array([sin(2*np.pi*(k+1)*t) for k in range(n_coeff)])
    linear_pred = np.dot(ak, cosines) + np.dot(bk, sines) + a0
    return np.log1p(np.exp(linear_pred))

def A_matrix(x_sensor, y_sensor, constants, wind_vector=None):
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
    
    result = np.zeros_like(dist_R)
    mask = dist_R > 0.1
    
    if np.any(mask):
        dR = dist_R[mask]
        dH = dist_H[mask]
        # Fixed dispersion for speed/stability in this loop test
        sigma_H = constants['a_H'] * (dR * np.tan(constants['gamma_H']))**constants['b_H'] + constants['w']
        sigma_V = constants['a_V'] * (dR * np.tan(constants['gamma_V']))**constants['b_V'] + constants['h']
        
        pre_factor = (100 / constants['RHO_CH4']) / (2 * np.pi * U_speed * sigma_H * sigma_V)
        term_H = np.exp(-(dH**2) / (2 * sigma_H**2))
        term_V = np.exp(-(dist_V**2) / (2 * sigma_V**2))
        result[mask] = pre_factor * term_H * term_V

    return result

class OptimizedModel:
    def __init__(self, beta, sigma_epsilon, physical_constants):
        self.beta = beta
        self.sigma_epsilon = sigma_epsilon
        self.consts = physical_constants

    def gen_data(self, T, Nt, Nx, Lx, ak, bk, a0):
        x = np.linspace(-Lx, Lx, Nx)
        y = np.linspace(-Lx, Lx, Nx)
        X, Y = np.meshgrid(x, y)
        X_flat, Y_flat = X.flatten(), Y.flatten()
        times = np.linspace(0, T, Nt)
        observations = []
        true_s_values = []
        
        for t in times:
            wind = wind_function(t, self.consts['wa0'], self.consts['wa'], self.consts['wb'])
            A = A_matrix(X_flat, Y_flat, self.consts, wind_vector=wind)
            st = s_function_corrected(t, ak, bk, a0)
            true_s_values.append(st)
            mu = np.dot(st, A) + self.beta
            observations.append(mu + np.random.normal(0, self.sigma_epsilon, size=mu.shape))
            
        return {
            'Y': np.array(observations), 'X1': X, 'X2': Y, 
            'times': times, 'true_s': np.array(true_s_values),
            'X_flat': X_flat, 'Y_flat': Y_flat
        }

    def log_likelihood(self, params_dict, data):
        Y_obs = data['Y']
        times = data['times']
        X1_f, X2_f = data['X_flat'], data['Y_flat']
        
        p_a0 = params_dict['a0']
        p_ak = params_dict.get('ak', np.zeros_like(p_a0))
        p_bk = params_dict.get('bk', np.zeros_like(p_a0))
        p_XS = params_dict['XS']
        p_YS = params_dict['YS']
        
        if np.any(np.abs(p_XS) > 5.0) or np.any(np.abs(p_YS) > 5.0):
            return -np.inf

        current_consts = self.consts.copy()
        current_consts['XS'] = p_XS
        current_consts['YS'] = p_YS
        
        log_ll = 0
        var = self.sigma_epsilon**2
        
        for i, t in enumerate(times):
            wind = wind_function(t, current_consts['wa0'], current_consts['wa'], current_consts['wb'])
            A = A_matrix(X1_f, X2_f, current_consts, wind_vector=wind)
            st = s_function_corrected(t, p_ak, p_bk, p_a0)
            mu = np.dot(st, A) + self.beta
            resid = Y_obs[i] - mu
            log_ll += -0.5 * np.sum(resid**2) / var
            
        return log_ll

# ==========================================
# PART 2: RMSE SCALING LOOP
# ==========================================

def run_scaling_analysis():
    np.random.seed(42)
    
    # Grid sizes to test (Nx x Nx sensors)
    # 2x2=4, 3x3=9, 4x4=16, 5x5=25, 6x6=36
    grid_sizes = [2, 3, 4, 5, 6] 
    
    rmse_static_list = []
    rmse_dyn_list = []
    
    # Common Setup
    T, Nt, Lx = 1.0, 10, 5.0 # Keep Nt small for speed
    consts = {
        'RHO_CH4': 0.656, 'U': 5.0,
        'XS': [2.0], 'YS': [-2.0], 'ZS': 0, 'Z': 0,
        'a_H': 1, 'b_H': 1, 'w': 1, 'a_V': 1, 'b_V': 1, 'h': 1,
        'gamma_H': 1, 'gamma_V': 1,
        'wa0': np.array([0.0, 4.0]), 'wa': [np.array([2.0, 0.0])], 'wb': [np.array([0.0, 1.0])]
    }
    true_a0, true_ak, true_bk = [1.0], [[1.5]], [[0.5]]
    
    model = OptimizedModel(beta=1.0, sigma_epsilon=0.1, physical_constants=consts)
    
    print(f"Starting RMSE Analysis across {len(grid_sizes)} grid configurations...")
    
    for Nx in grid_sizes:
        num_sensors = Nx * Nx
        print(f"\nTesting Grid: {Nx}x{Nx} ({num_sensors} sensors)")
        
        # 1. Generate Data for this specific grid
        data = model.gen_data(T, Nt, Nx, Lx, np.array(true_ak), np.array(true_bk), np.array(true_a0))
        
        # 2. Fit Static Model
        # Short chain for speed (approximate result is fine for trend)
        chain_static = []
        curr = np.concatenate([[0.5], [0.0], [0.0]])
        for i in range(600): # Fast burn
            prop = curr + np.random.normal(0, 0.1, 3)
            curr_ll = model.log_likelihood({'a0': [curr[0]], 'XS': [curr[1]], 'YS': [curr[2]]}, data)
            prop_ll = model.log_likelihood({'a0': [prop[0]], 'XS': [prop[1]], 'YS': [prop[2]]}, data)
            if np.log(np.random.rand()) < (prop_ll - curr_ll): curr = prop
            if i > 200: chain_static.append(curr) # Keep last 400
        mean_static = np.mean(chain_static, axis=0)
        
        # 3. Fit Dynamic Model
        chain_dyn = []
        # Warm start to ensure it finds the mode quickly
        curr = np.concatenate([[1.0], [1.0], [0.0], [2.0], [-2.0]]) 
        for i in range(600):
            prop = curr + np.random.normal(0, 0.05, 5)
            curr_ll = model.log_likelihood({'a0': [curr[0]], 'ak': [[curr[1]]], 'bk': [[curr[2]]], 'XS': [curr[3]], 'YS': [curr[4]]}, data)
            prop_ll = model.log_likelihood({'a0': [prop[0]], 'ak': [[prop[1]]], 'bk': [[prop[2]]], 'XS': [prop[3]], 'YS': [prop[4]]}, data)
            if np.log(np.random.rand()) < (prop_ll - curr_ll): curr = prop
            if i > 200: chain_dyn.append(curr)
        mean_dyn = np.mean(chain_dyn, axis=0)
        
        # 4. Compute RMSE
        t_eval = np.linspace(0, T, 50)
        s_true = np.array([s_function_corrected(t, true_ak, true_bk, true_a0)[0] for t in t_eval])
        
        # Static prediction (constant line)
        s_pred_stat = np.array([s_function_corrected(0, [0], [0], [mean_static[0]])[0] for t in t_eval])
        # Dynamic prediction
        s_pred_dyn = np.array([s_function_corrected(t, [mean_dyn[1]], [mean_dyn[2]], [mean_dyn[0]])[0] for t in t_eval])
        
        rmse_stat = np.sqrt(np.mean((s_true - s_pred_stat)**2))
        rmse_dyn = np.sqrt(np.mean((s_true - s_pred_dyn)**2))
        
        rmse_static_list.append(rmse_stat)
        rmse_dyn_list.append(rmse_dyn)
        
        print(f"  -> RMSE Static: {rmse_stat:.3f}, Dynamic: {rmse_dyn:.3f}")

    # ==========================================
    # PART 3: CREATIVE PLOTTING
    # ==========================================
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # PLOT 1: RMSE Scaling
    sensors = [n*n for n in grid_sizes]
    axes[0].plot(sensors, rmse_static_list, 'r--o', label='Static Model (Baseline)', linewidth=2)
    axes[0].plot(sensors, rmse_dyn_list, 'b-o', label='Dynamic Model (Ours)', linewidth=2)
    axes[0].set_xlabel("Number of Sensors (Observations)", fontsize=12)
    axes[0].set_ylabel("Source RMSE", fontsize=12)
    axes[0].set_title("Inversion Accuracy vs. Data Density", fontsize=14)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=11)
    
    # PLOT 2: Sensor Densification Visual
    # Show the grid for the largest Nx (background) and smallest Nx (foreground)
    
    # Largest Grid
    Nx_max = grid_sizes[-1]
    x_max = np.linspace(-Lx, Lx, Nx_max)
    y_max = np.linspace(-Lx, Lx, Nx_max)
    X_max, Y_max = np.meshgrid(x_max, y_max)
    
    # Smallest Grid
    Nx_min = grid_sizes[0]
    x_min = np.linspace(-Lx, Lx, Nx_min)
    y_min = np.linspace(-Lx, Lx, Nx_min)
    X_min, Y_min = np.meshgrid(x_min, y_min)
    
    # Mid Grid
    Nx_mid = grid_sizes[2]
    x_mid = np.linspace(-Lx, Lx, Nx_mid)
    y_mid = np.linspace(-Lx, Lx, Nx_mid)
    X_mid, Y_mid = np.meshgrid(x_mid, y_mid)
    
    axes[1].scatter(X_max, Y_max, c='lightgray', s=50, label=f'High Density ({Nx_max}x{Nx_max})')
    axes[1].scatter(X_mid, Y_mid, c='skyblue', s=100, marker='s', label=f'Med Density ({Nx_mid}x{Nx_mid})')
    axes[1].scatter(X_min, Y_min, c='red', s=150, marker='*', label=f'Low Density ({Nx_min}x{Nx_min})')
    
    # Add Source Location
    axes[1].plot(consts['XS'][0], consts['YS'][0], 'kX', markersize=15, label='True Source')
    
    axes[1].set_title("Sensor Grid Densification", fontsize=14)
    axes[1].set_xlim(-Lx-1, Lx+1)
    axes[1].set_ylim(-Lx-1, Lx+1)
    axes[1].legend(loc='upper right')
    axes[1].grid(True, linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('presentation_3_scaling.png', dpi=300)
    print("\nSaved presentation_3_scaling.png")

if __name__ == "__main__":
    run_scaling_analysis()