import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from math import cos, sin
import os

# Ensure figures directory exists
OUTPUT_DIR = 'Problem_solving/figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# PART 1: PHYSICS & CORE FUNCTIONS
# ==========================================

def wind_function(t, wa0, wa, wb):
    wind = np.array(wa0)
    for k in range(len(wa)):
        wind += wa[k] * cos(2 * np.pi * (k + 1) * t) + wb[k] * sin(2 * np.pi * (k + 1) * t)
    return wind

def s_function(t, ak, bk, a0):
    """
    Source function s(t) using Fourier series and Softplus for positivity.
    s(t) = ln(1 + exp( Fourier(t) ))
    """
    ak = np.atleast_1d(ak)
    bk = np.atleast_1d(bk)
    a0 = np.atleast_1d(a0)
    
    # Handle multiple sources if ak is 2D
    if ak.ndim == 2:
        n_coeff = ak.shape[1]
        cosines = np.array([cos(2*np.pi*(k+1)*t) for k in range(n_coeff)])
        sines = np.array([sin(2*np.pi*(k+1)*t) for k in range(n_coeff)])
        linear_pred = np.dot(ak, cosines) + np.dot(bk, sines) + a0
    else:
        n_coeff = len(ak)
        cosines = np.array([cos(2*np.pi*(k+1)*t) for k in range(n_coeff)])
        sines = np.array([sin(2*np.pi*(k+1)*t) for k in range(n_coeff)])
        linear_pred = np.dot(ak, cosines) + np.dot(bk, sines) + a0
    
    return np.log1p(np.exp(linear_pred))

def A_matrix(x_sensor, y_sensor, constants, wind_vector=None):
    """
    Calculates the concentration using the Gaussian plume formula.
    """
    XS, YS, ZS = np.atleast_1d(constants['XS']), np.atleast_1d(constants['YS']), constants['ZS']
    Z = constants['Z']
    
    if wind_vector is None:
        u_vec = np.array([1.0, 0.0])
        U_speed = constants['U']
    else:
        U_speed = np.linalg.norm(wind_vector)
        if U_speed < 1e-6:
            u_vec = np.array([1.0, 0.0])
        else:
            u_vec = wind_vector / U_speed

    x_in = np.atleast_1d(x_sensor)
    y_in = np.atleast_1d(y_sensor)
    
    # Broadcasting for multiple sources
    # dx, dy shape: (N_sources, N_sensors)
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

    return result # Shape (N_sources, N_sensors)

def log_prior_coefficients(coeff):
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

def rwmh(start_point, proposal_variance, n_steps, log_posterior):
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
            st = s_function(t, ak, bk, a0)
            true_s_values.append(st)
            
            # A is (N_src, N_pix), st is (N_src,) or scalar
            if np.ndim(st) > 0:
                mu = np.dot(st, A) + self.beta
            else:
                mu = np.sum(A, axis=0) * st + self.beta
                
            obs = mu + np.random.normal(0, self.sigma_epsilon, size=mu.shape)
            observations.append(obs)
            
        return {
            'Y': np.array(observations), 
            'X1': X, 'X2': Y, 
            'times': times, 
            'true_s': np.array(true_s_values),
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
        
        # Bounds check
        if np.any(np.abs(p_XS) > 6.0) or np.any(np.abs(p_YS) > 6.0):
            return -np.inf

        current_consts = self.consts.copy()
        current_consts['XS'] = p_XS
        current_consts['YS'] = p_YS
        
        log_ll = 0
        var = self.sigma_epsilon**2
        
        # Pre-calculate wind if possible, but here we loop
        for i, t in enumerate(times):
            wind = wind_function(t, current_consts['wa0'], current_consts['wa'], current_consts['wb'])
            A = A_matrix(X1_f, X2_f, current_consts, wind_vector=wind)
            st = s_function(t, p_ak, p_bk, p_a0)
            
            if np.ndim(st) > 0:
                mu = np.dot(st, A) + self.beta
            else:
                mu = np.sum(A, axis=0) * st + self.beta
                
            resid = Y_obs[i] - mu
            log_ll += -0.5 * np.sum(resid**2) / var
            
        return log_ll

# ==========================================
# PART 2: PLOTTING & DEMOS
# ==========================================

def run_constant_source_demo():
    print("Generating Constant Source Demo...")
    T, Nt, Nx, Lx = 1.0, 20, 30, 5.0
    consts = {
        'RHO_CH4': 0.656, 'U': 5.0,
        'XS': [0.0], 'YS': [0.0], 'ZS': 0, 'Z': 0,
        'a_H': 1, 'b_H': 1, 'w': 1, 'a_V': 1, 'b_V': 1, 'h': 1,
        'gamma_H': 1, 'gamma_V': 1,
        'wa0': np.array([0.0, 5.0]), 'wa': [np.array([1.0, 0.0])], 'wb': [np.array([0.0, 1.0])]
    }
    
    # Constant source: ak=0, bk=0
    ak = np.array([[0.0]])
    bk = np.array([[0.0]])
    a0 = np.array([1.0])
    
    model = Model(beta=1.0, sigma_epsilon=0.01, physical_constants=consts)
    data = model.gen_data(T, Nt, Nx, Lx, ak, bk, a0)
    
    # Plot snapshot
    plt.figure()
    plt.contourf(data['X1'], data['X2'], data['Y'][0].reshape(Nx, Nx), levels=20, cmap='viridis')
    plt.colorbar()
    plt.title("Constant Source Plume Snapshot")
    plt.savefig(os.path.join(OUTPUT_DIR, 'demo_constant_source_snapshot.png'))
    plt.close()
    
    # Animation
    fig, ax = plt.subplots()
    def update(frame):
        ax.clear()
        t = frame * T / (Nt - 1)
        contour = ax.contourf(data['X1'], data['X2'], data['Y'][frame].reshape(Nx, Nx), levels=20, cmap='viridis')
        ax.set_title(f'Constant Source Model at t={t:.2f}')
        return contour,
    
    ani = animation.FuncAnimation(fig, update, frames=Nt, interval=200)
    ani.save(os.path.join(OUTPUT_DIR, 'demo_constant_source.gif'), writer='pillow')
    print(f"Saved {OUTPUT_DIR}/demo_constant_source.gif")


def run_varying_source_demo():
    print("Generating Varying Source Demo (Multiple Sources)...")
    T, Nt, Nx, Lx = 1.0, 20, 30, 5.0
    consts = {
        'RHO_CH4': 0.656, 'U': 5.0,
        'XS': [2.0, -2.0], 'YS': [-2.0, 2.0], 'ZS': 0, 'Z': 0,
        'a_H': 1, 'b_H': 1, 'w': 1, 'a_V': 1, 'b_V': 1, 'h': 1,
        'gamma_H': 1, 'gamma_V': 1,
        'wa0': np.array([0.0, 5.0]), 'wa': [np.array([1.0, 0.0])], 'wb': [np.array([0.0, 1.0])]
    }
    
    # Varying source
    ak = np.array([[1.0], [0.5]])
    bk = np.array([[0.5], [-0.5]])
    a0 = np.array([1.0, 1.0])
    
    model = Model(beta=1.0, sigma_epsilon=0.01, physical_constants=consts)
    data = model.gen_data(T, Nt, Nx, Lx, ak, bk, a0)
    
    # Plot snapshot
    plt.figure()
    plt.contourf(data['X1'], data['X2'], data['Y'][0].reshape(Nx, Nx), levels=20, cmap='viridis')
    plt.colorbar()
    plt.title("Varying Source Plume Snapshot")
    plt.savefig(os.path.join(OUTPUT_DIR, 'demo_varying_source_snapshot.png'))
    plt.close()
    
    # Animation
    fig, ax = plt.subplots()
    def update(frame):
        ax.clear()
        t = frame * T / (Nt - 1)
        contour = ax.contourf(data['X1'], data['X2'], data['Y'][frame].reshape(Nx, Nx), levels=20, cmap='viridis')
        ax.set_title(f'Varying Source Model at t={t:.2f}')
        return contour,
    
    ani = animation.FuncAnimation(fig, update, frames=Nt, interval=200)
    ani.save(os.path.join(OUTPUT_DIR, 'demo_varying_source.gif'), writer='pillow')
    print(f"Saved {OUTPUT_DIR}/demo_varying_source.gif")


def run_inference_and_plots():
    print("Running Inference and Generating Plots...")
    np.random.seed(42)
    
    # Setup
    T, Nt, Nx, Lx = 1.0, 15, 15, 5.0
    consts = {
        'RHO_CH4': 0.656, 'U': 5.0,
        'XS': [2.0], 'YS': [-2.0], 'ZS': 0, 'Z': 0,
        'a_H': 1, 'b_H': 1, 'w': 1, 'a_V': 1, 'b_V': 1, 'h': 1,
        'gamma_H': 1, 'gamma_V': 1,
        'wa0': np.array([0.0, 4.0]), 'wa': [np.array([2.0, 0.0])], 'wb': [np.array([0.0, 1.0])]
    }
    
    true_a0 = np.array([1.0])
    true_ak = np.array([[1.5]])
    true_bk = np.array([[0.5]])
    
    model = Model(beta=1.0, sigma_epsilon=0.1, physical_constants=consts)
    data = model.gen_data(T, Nt, Nx, Lx, true_ak, true_bk, true_a0)
    
    # --- MCMC ---
    # Parameters: a0, ak, bk, XS, YS (5 params)
    # Flattened: [a0, ak, bk, XS, YS]
    
    def log_posterior(flat_coeff):
        # Unpack
        p_a0 = flat_coeff[0:1]
        p_ak = flat_coeff[1:2].reshape(1,1)
        p_bk = flat_coeff[2:3].reshape(1,1)
        p_XS = flat_coeff[3:4]
        p_YS = flat_coeff[4:5]
        
        params = {'a0': p_a0, 'ak': p_ak, 'bk': p_bk, 'XS': p_XS, 'YS': p_YS}
        
        # Prior
        lp_coeff = log_prior_coefficients([p_a0, p_ak, p_bk])
        
        # Likelihood
        ll = model.log_likelihood(params, data)
        
        return lp_coeff + ll

    # Initial point near truth
    initial_point = np.array([1.0, 1.0, 0.0, 2.0, -2.0])
    
    print("Running MCMC...")
    chain, acc = rwmh(initial_point, 0.05, 5000, log_posterior)
    print(f"MCMC Acceptance Rate: {acc}")
    
    burn_in = 1000
    chain_burned = chain[burn_in:]
    mean_params = np.mean(chain_burned, axis=0)
    
    # --- PLOT 1: MCMC Chains ---
    plt.figure(figsize=(10, 12))
    labels = ['a0', 'ak', 'bk', 'XS', 'YS']
    true_vals = [true_a0[0], true_ak[0,0], true_bk[0,0], consts['XS'][0], consts['YS'][0]]
    
    for i in range(5):
        plt.subplot(5, 1, i+1)
        plt.plot(chain[:, i], label='Chain')
        plt.axhline(true_vals[i], color='r', linestyle='--', label='True')
        plt.ylabel(labels[i])
        plt.legend()
    plt.xlabel('Iteration')
    plt.suptitle('MCMC Chains')
    plt.savefig(os.path.join(OUTPUT_DIR, 'presentation_mcmc_chains.png'))
    plt.close()
    
    # --- PLOT 2: 95% CI for Source Intensity ---
    print("Generating 95% CI Plot...")
    t_plot = np.linspace(0, T, 100)
    s_samples = []
    
    # Sample from chain
    indices = np.random.choice(len(chain_burned), size=200, replace=False)
    for idx in indices:
        sample = chain_burned[idx]
        p_a0 = sample[0:1]
        p_ak = sample[1:2].reshape(1,1)
        p_bk = sample[2:3].reshape(1,1)
        s_t = [s_function(t, p_ak, p_bk, p_a0)[0] for t in t_plot]
        s_samples.append(s_t)
        
    s_samples = np.array(s_samples)
    s_mean = np.mean(s_samples, axis=0)
    s_lower = np.percentile(s_samples, 2.5, axis=0)
    s_upper = np.percentile(s_samples, 97.5, axis=0)
    s_true = [s_function(t, true_ak, true_bk, true_a0)[0] for t in t_plot]
    
    plt.figure(figsize=(10, 6))
    plt.plot(t_plot, s_true, 'k-', linewidth=2, label='True Source')
    plt.plot(t_plot, s_mean, 'b--', linewidth=2, label='Posterior Mean')
    plt.fill_between(t_plot, s_lower, s_upper, color='blue', alpha=0.2, label='95% CI')
    plt.xlabel('Time')
    plt.ylabel('Source Intensity s(t)')
    plt.title('Posterior Source Intensity with 95% CI')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, 'presentation_source_ci.png'))
    plt.close()
    
    # --- PLOT 3: Ghost Residuals (Static vs Dynamic) ---
    print("Generating Ghost Residuals Plot...")
    # Fit Static Model (a0, XS, YS only)
    def log_posterior_static(flat_coeff):
        p_a0 = flat_coeff[0:1]
        p_XS = flat_coeff[1:2]
        p_YS = flat_coeff[2:3]
        params = {'a0': p_a0, 'XS': p_XS, 'YS': p_YS} # ak, bk default to 0
        lp = -0.5 * np.sum(p_a0**2) # Simple prior
        ll = model.log_likelihood(params, data)
        return lp + ll
        
    chain_static, _ = rwmh(np.array([0.5, 0.0, 0.0]), 0.05, 2000, log_posterior_static)
    mean_static = np.mean(chain_static[500:], axis=0)
    
    # Find max emission time
    idx_max = np.argmax(data['true_s'])
    t_max = data['times'][idx_max]
    obs_max = data['Y'][idx_max].reshape(Nx, Nx)
    
    # Reconstruct Static
    p_stat_a0 = mean_static[0:1]
    p_stat_XS = mean_static[1:2]
    p_stat_YS = mean_static[2:3]
    consts_stat = consts.copy()
    consts_stat['XS'] = p_stat_XS
    consts_stat['YS'] = p_stat_YS
    wind_max = wind_function(t_max, consts['wa0'], consts['wa'], consts['wb'])
    A_stat = A_matrix(data['X_flat'], data['Y_flat'], consts_stat, wind_vector=wind_max)
    s_stat = s_function(t_max, [0], [0], p_stat_a0)
    mu_stat = (np.dot(s_stat, A_stat) + 1.0).reshape(Nx, Nx)
    
    # Reconstruct Dynamic (using mean_params from earlier)
    p_dyn_a0 = mean_params[0:1]
    p_dyn_ak = mean_params[1:2].reshape(1,1)
    p_dyn_bk = mean_params[2:3].reshape(1,1)
    p_dyn_XS = mean_params[3:4]
    p_dyn_YS = mean_params[4:5]
    consts_dyn = consts.copy()
    consts_dyn['XS'] = p_dyn_XS
    consts_dyn['YS'] = p_dyn_YS
    A_dyn = A_matrix(data['X_flat'], data['Y_flat'], consts_dyn, wind_vector=wind_max)
    s_dyn = s_function(t_max, p_dyn_ak, p_dyn_bk, p_dyn_a0)
    mu_dyn = (np.dot(s_dyn, A_dyn) + 1.0).reshape(Nx, Nx)
    
    res_stat = np.abs(obs_max - mu_stat)
    res_dyn = np.abs(obs_max - mu_dyn)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    vmax = max(np.max(res_stat), np.max(res_dyn))
    im1 = axes[0].imshow(res_stat, cmap='hot', vmin=0, vmax=vmax)
    axes[0].set_title("Static Model Residuals (Ghost Plume)")
    im2 = axes[1].imshow(res_dyn, cmap='hot', vmin=0, vmax=vmax)
    axes[1].set_title("Dynamic Model Residuals")
    plt.colorbar(im1, ax=axes.ravel().tolist())
    plt.savefig(os.path.join(OUTPUT_DIR, 'presentation_ghost_residuals.png'))
    plt.close()
    
    # --- PLOT 4: Side-by-Side Animation ---
    print("Generating Side-by-Side Animation...")
    # Reconstruct full dynamic model data
    model_dyn = Model(1.0, 0.1, consts_dyn)
    data_dyn = model_dyn.gen_data(T, Nt, Nx, Lx, p_dyn_ak, p_dyn_bk, p_dyn_a0)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    def update_side_by_side(frame):
        axes[0].clear()
        axes[1].clear()
        
        # True Data
        axes[0].contourf(data['X1'], data['X2'], data['Y'][frame].reshape(Nx, Nx), levels=20, cmap='viridis')
        axes[0].set_title(f"True Data (t={data['times'][frame]:.2f})")
        
        # Recovered Data
        axes[1].contourf(data_dyn['X1'], data_dyn['X2'], data_dyn['Y'][frame].reshape(Nx, Nx), levels=20, cmap='viridis')
        axes[1].set_title(f"Recovered Model (t={data['times'][frame]:.2f})")
        
        return axes
        
    ani = animation.FuncAnimation(fig, update_side_by_side, frames=Nt, interval=200)
    ani.save(os.path.join(OUTPUT_DIR, 'presentation_comparison.gif'), writer='pillow')
    print(f"Saved {OUTPUT_DIR}/presentation_comparison.gif")


def run_rmse_scaling_analysis():
    print("Running RMSE Scaling Analysis (Varying Sensors)...")
    np.random.seed(42)
    grid_sizes = [2, 3, 4, 5, 6]
    
    rmse_intensity_static = []
    rmse_intensity_dyn = []
    rmse_loc_static = []
    rmse_loc_dyn = []
    
    # Store estimated locations for convergence plot
    est_locs_static = []
    est_locs_dyn = []
    
    T, Nt, Lx = 1.0, 10, 5.0
    consts = {
        'RHO_CH4': 0.656, 'U': 5.0,
        'XS': [2.0], 'YS': [-2.0], 'ZS': 0, 'Z': 0,
        'a_H': 1, 'b_H': 1, 'w': 1, 'a_V': 1, 'b_V': 1, 'h': 1,
        'gamma_H': 1, 'gamma_V': 1,
        'wa0': np.array([0.0, 4.0]), 'wa': [np.array([2.0, 0.0])], 'wb': [np.array([0.0, 1.0])]
    }
    true_a0, true_ak, true_bk = [1.0], [[1.5]], [[0.5]]
    true_loc = np.array([consts['XS'][0], consts['YS'][0]])
    
    model = Model(beta=1.0, sigma_epsilon=0.1, physical_constants=consts)
    
    for Nx in grid_sizes:
        print(f"  Grid {Nx}x{Nx}...")
        data = model.gen_data(T, Nt, Nx, Lx, np.array(true_ak), np.array(true_bk), np.array(true_a0))
        
        # Static
        def lp_stat(fc):
            return -0.5*np.sum(fc[0]**2) + model.log_likelihood({'a0':fc[0:1], 'XS':fc[1:2], 'YS':fc[2:3]}, data)
        chain_stat, _ = rwmh(np.array([0.5, 0.0, 0.0]), 0.1, 500, lp_stat)
        mean_stat = np.mean(chain_stat[200:], axis=0)
        
        # Dynamic
        def lp_dyn(fc):
             p = {'a0':fc[0:1], 'ak':fc[1:2].reshape(1,1), 'bk':fc[2:3].reshape(1,1), 'XS':fc[3:4], 'YS':fc[4:5]}
             return log_prior_coefficients([p['a0'], p['ak'], p['bk']]) + model.log_likelihood(p, data)
        chain_dyn, _ = rwmh(np.array([1.0, 1.0, 0.0, 2.0, -2.0]), 0.05, 500, lp_dyn)
        mean_dyn = np.mean(chain_dyn[200:], axis=0)
        
        # 1. Intensity RMSE
        t_eval = np.linspace(0, T, 50)
        s_true = np.array([s_function(t, true_ak, true_bk, true_a0)[0] for t in t_eval])
        s_stat = np.array([s_function(t, [0], [0], [mean_stat[0]])[0] for t in t_eval])
        s_dyn = np.array([s_function(t, [mean_dyn[1]], [mean_dyn[2]], [mean_dyn[0]])[0] for t in t_eval])
        
        rmse_intensity_static.append(np.sqrt(np.mean((s_true - s_stat)**2)))
        rmse_intensity_dyn.append(np.sqrt(np.mean((s_true - s_dyn)**2)))
        
        # 2. Location RMSE (Euclidean Distance)
        loc_stat = mean_stat[1:3]
        loc_dyn = mean_dyn[3:5]
        rmse_loc_static.append(np.linalg.norm(loc_stat - true_loc))
        rmse_loc_dyn.append(np.linalg.norm(loc_dyn - true_loc))
        
        est_locs_static.append(loc_stat)
        est_locs_dyn.append(loc_dyn)
        
    # Plotting RMSE
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sensors = [n*n for n in grid_sizes]
    
    # Plot 1: Intensity RMSE
    axes[0].plot(sensors, rmse_intensity_static, 'r--o', label='Static Model')
    axes[0].plot(sensors, rmse_intensity_dyn, 'b-o', label='Dynamic Model')
    axes[0].set_xlabel('Number of Sensors')
    axes[0].set_ylabel('RMSE (Source Intensity)')
    axes[0].set_title('Source Intensity Recovery vs Sensors')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot 2: Location RMSE
    axes[1].plot(sensors, rmse_loc_static, 'r--o', label='Static Model')
    axes[1].plot(sensors, rmse_loc_dyn, 'b-o', label='Dynamic Model')
    axes[1].set_xlabel('Number of Sensors')
    axes[1].set_ylabel('Location Error (m)')
    axes[1].set_title('Source Location Error vs Sensors')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'presentation_rmse_scaling.png'))
    plt.close()
    print(f"Saved {OUTPUT_DIR}/presentation_rmse_scaling.png")
    
    # --- PLOT 3: Spatial Convergence (Target Plot) ---
    print("Generating Spatial Convergence Plot...")
    plt.figure(figsize=(10, 10))
    
    # Plot True Source (Bullseye)
    plt.plot(true_loc[0], true_loc[1], 'k*', markersize=20, label='True Source')
    
    # Extract X and Y coords
    stat_x = [p[0] for p in est_locs_static]
    stat_y = [p[1] for p in est_locs_static]
    dyn_x = [p[0] for p in est_locs_dyn]
    dyn_y = [p[1] for p in est_locs_dyn]
    
    # Connect lines
    plt.plot(stat_x, stat_y, 'r--', alpha=0.3)
    plt.plot(dyn_x, dyn_y, 'b--', alpha=0.3)
    
    sensors = [n*n for n in grid_sizes]
    norm = plt.Normalize(min(sensors), max(sensors))
    
    # Static Estimates (Empty Circles)
    for i, (x, y) in enumerate(zip(stat_x, stat_y)):
        n_s = sensors[i]
        color = plt.cm.Reds(norm(n_s))
        # Edge color is mapped, face is empty
        plt.scatter(x, y, s=150, edgecolors=color, facecolors='none', linewidths=2, zorder=3)
        plt.annotate(f"{n_s}", (x, y), xytext=(-15, 10), textcoords='offset points', fontsize=10, color='darkred')
        
    # Dynamic Estimates (Filled Circles)
    for i, (x, y) in enumerate(zip(dyn_x, dyn_y)):
        n_s = sensors[i]
        color = plt.cm.Blues(norm(n_s))
        # Face color is mapped
        plt.scatter(x, y, s=150, facecolors=color, edgecolors='k', zorder=3)
        plt.annotate(f"{n_s}", (x, y), xytext=(5, 10), textcoords='offset points', fontsize=10, color='darkblue')

    # Custom Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='*', color='w', markerfacecolor='k', markersize=15, label='True Source'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='none', markeredgecolor='r', markersize=10, markeredgewidth=2, label='Static (Empty)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='b', markeredgecolor='k', markersize=10, label='Dynamic (Filled)')
    ]
    
    plt.title(f"Source Localization Convergence\n(Labels = Number of Sensors)", fontsize=14)
    plt.xlabel("X Coordinate (m)")
    plt.ylabel("Y Coordinate (m)")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(handles=legend_elements, loc='upper right')
    plt.axis('equal')
    
    # Zoom in
    all_x = stat_x + dyn_x + [true_loc[0]]
    all_y = stat_y + dyn_y + [true_loc[1]]
    plt.xlim(min(all_x)-1, max(all_x)+1)
    plt.ylim(min(all_y)-1, max(all_y)+1)
    
    plt.savefig(os.path.join(OUTPUT_DIR, 'presentation_location_convergence.png'), dpi=300)
    plt.close()
    print(f"Saved {OUTPUT_DIR}/presentation_location_convergence.png")

if __name__ == "__main__":
    run_constant_source_demo()
    run_varying_source_demo()
    run_inference_and_plots()
    run_rmse_scaling_analysis()
