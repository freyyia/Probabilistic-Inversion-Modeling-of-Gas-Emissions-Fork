import numpy as np
import matplotlib.pyplot as plt
from math import cos, sin

# ==========================================
# PART 1: PHYSICS & MODEL (Standardized)
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
        
        for t in times:
            wind = wind_function(t, self.consts['wa0'], self.consts['wa'], self.consts['wb'])
            A = A_matrix(X_flat, Y_flat, self.consts, wind_vector=wind)
            st = s_function_corrected(t, ak, bk, a0)
            mu = np.dot(st, A) + self.beta
            observations.append(mu + np.random.normal(0, self.sigma_epsilon, size=mu.shape))
            
        return {'Y': np.array(observations), 'times': times, 'X_flat': X_flat, 'Y_flat': Y_flat}

    def log_likelihood(self, params_dict, data):
        Y_obs = data['Y']
        times = data['times']
        X1_f, X2_f = data['X_flat'], data['Y_flat']
        
        p_a0 = params_dict['a0']
        p_ak = params_dict.get('ak', np.zeros_like(p_a0))
        p_bk = params_dict.get('bk', np.zeros_like(p_a0))
        p_XS = params_dict['XS']
        p_YS = params_dict['YS']
        
        if np.any(np.abs(p_XS) > 6.0) or np.any(np.abs(p_YS) > 6.0): return -np.inf

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
# PART 2: LOCATION ACCURACY ANALYSIS
# ==========================================

def run_location_analysis():
    np.random.seed(101) # New seed for variety
    
    # Grid sizes (N x N)
    grid_sizes = [2, 3, 4, 5, 6] 
    
    loc_error_static = []
    loc_error_dyn = []
    
    # To store coordinates for the spatial plot
    est_locs_static = [] # List of (x, y) tuples
    est_locs_dyn = []
    
    # Setup
    T, Nt, Lx = 1.0, 10, 5.0
    consts = {
        'RHO_CH4': 0.656, 'U': 5.0,
        'XS': [2.0], 'YS': [-2.0], 'ZS': 0, 'Z': 0,
        'a_H': 1, 'b_H': 1, 'w': 1, 'a_V': 1, 'b_V': 1, 'h': 1,
        'gamma_H': 1, 'gamma_V': 1,
        'wa0': np.array([0.0, 4.0]), 'wa': [np.array([2.0, 0.0])], 'wb': [np.array([0.0, 1.0])]
    }
    true_loc = np.array([consts['XS'][0], consts['YS'][0]])
    true_a0, true_ak, true_bk = [1.0], [[1.5]], [[0.5]]
    
    model = OptimizedModel(beta=1.0, sigma_epsilon=0.1, physical_constants=consts)
    
    print(f"Analyzing Source Localization Accuracy...")
    
    for Nx in grid_sizes:
        print(f"  Testing {Nx}x{Nx} grid...")
        data = model.gen_data(T, Nt, Nx, Lx, np.array(true_ak), np.array(true_bk), np.array(true_a0))
        
        # 1. Static Model
        chain_static = []
        curr = np.concatenate([[0.5], [0.0], [0.0]]) # Start at center (0,0)
        for i in range(500):
            prop = curr + np.random.normal(0, 0.1, 3)
            curr_ll = model.log_likelihood({'a0': [curr[0]], 'XS': [curr[1]], 'YS': [curr[2]]}, data)
            prop_ll = model.log_likelihood({'a0': [prop[0]], 'XS': [prop[1]], 'YS': [prop[2]]}, data)
            if np.log(np.random.rand()) < (prop_ll - curr_ll): curr = prop
            if i > 200: chain_static.append(curr)
        mean_static = np.mean(chain_static, axis=0)
        est_static = np.array([mean_static[1], mean_static[2]])
        
        # 2. Dynamic Model
        chain_dyn = []
        # Start offset from truth to show convergence capability
        curr = np.concatenate([[1.0], [1.0], [0.0], [1.0], [-1.0]]) 
        for i in range(500):
            prop = curr + np.random.normal(0, 0.08, 5)
            curr_ll = model.log_likelihood({'a0': [curr[0]], 'ak': [[curr[1]]], 'bk': [[curr[2]]], 'XS': [curr[3]], 'YS': [curr[4]]}, data)
            prop_ll = model.log_likelihood({'a0': [prop[0]], 'ak': [[prop[1]]], 'bk': [[prop[2]]], 'XS': [prop[3]], 'YS': [prop[4]]}, data)
            if np.log(np.random.rand()) < (prop_ll - curr_ll): curr = prop
            if i > 200: chain_dyn.append(curr)
        mean_dyn = np.mean(chain_dyn, axis=0)
        est_dyn = np.array([mean_dyn[3], mean_dyn[4]])
        
        # 3. Calculate Errors
        err_stat = np.linalg.norm(est_static - true_loc)
        err_dyn = np.linalg.norm(est_dyn - true_loc)
        
        loc_error_static.append(err_stat)
        loc_error_dyn.append(err_dyn)
        est_locs_static.append(est_static)
        est_locs_dyn.append(est_dyn)

    # ==========================================
    # PART 3: VISUALIZATION
    # ==========================================
    
    # PLOT 1: The "Target" Plot (Spatial Convergence)
    plt.figure(figsize=(8, 8))
    
    # Plot True Source (Bullseye)
    plt.plot(true_loc[0], true_loc[1], 'k*', markersize=20, label='True Source Location')
    
    # Extract X and Y coords
    stat_x = [p[0] for p in est_locs_static]
    stat_y = [p[1] for p in est_locs_static]
    dyn_x = [p[0] for p in est_locs_dyn]
    dyn_y = [p[1] for p in est_locs_dyn]
    
    # Scale point sizes by grid density (Larger points = More Sensors)
    sizes = [s**2 * 5 for s in grid_sizes]
    
    # Plot Estimates with connecting lines to show trajectory
    plt.plot(stat_x, stat_y, 'r--', alpha=0.3)
    plt.plot(dyn_x, dyn_y, 'b--', alpha=0.3)
    
    scatter_stat = plt.scatter(stat_x, stat_y, c=grid_sizes, cmap='Reds', s=100, edgecolors='k', label='Static Model Estimates')
    scatter_dyn = plt.scatter(dyn_x, dyn_y, c=grid_sizes, cmap='Blues', s=100, edgecolors='k', label='Dynamic Model Estimates')
    
    # Add a colorbar to indicate sensor count
    cbar = plt.colorbar(scatter_dyn, fraction=0.046, pad=0.04)
    cbar.set_label('Grid Size (N x N)', rotation=270, labelpad=15)
    
    plt.title(f"Source Localization: Convergence to Truth ({true_loc[0]}, {true_loc[1]})", fontsize=14)
    plt.xlabel("X Coordinate (m)")
    plt.ylabel("Y Coordinate (m)")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper right')
    plt.axis('equal')
    
    # Zoom in reasonably around the action
    all_x = stat_x + dyn_x + [true_loc[0]]
    all_y = stat_y + dyn_y + [true_loc[1]]
    plt.xlim(min(all_x)-1, max(all_x)+1)
    plt.ylim(min(all_y)-1, max(all_y)+1)
    
    plt.savefig('presentation_4_location_target.png', dpi=300)
    print("Saved presentation_4_location_target.png")
    
    # PLOT 2: Location Error vs Sensor Count
    plt.figure(figsize=(10, 6))
    sensors = [n*n for n in grid_sizes]
    
    plt.plot(sensors, loc_error_static, 'r--o', linewidth=2, label='Static Model Error')
    plt.plot(sensors, loc_error_dyn, 'b-o', linewidth=2, label='Dynamic Model Error')
    
    plt.fill_between(sensors, loc_error_dyn, 0, color='blue', alpha=0.1)
    
    plt.title("Distance to True Source vs. Data Density", fontsize=14)
    plt.xlabel("Number of Sensors", fontsize=12)
    plt.ylabel("Euclidean Error (m)", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.savefig('presentation_5_location_error.png', dpi=300)
    print("Saved presentation_5_location_error.png")

if __name__ == "__main__":
    run_location_analysis()