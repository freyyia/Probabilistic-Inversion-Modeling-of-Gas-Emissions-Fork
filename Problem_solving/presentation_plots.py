import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from math import cos, sin
import os
import sys

# Ensure figures directory exists
OUTPUT_DIR = 'Problem_solving/figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Add current directory to path to allow imports if run from root
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ==========================================
# PART 1: PHYSICS & CORE FUNCTIONS
# ==========================================

# Import core functions and Model from the refactored module
try:
    from source_varying_functions import wind_function, s_function, A_matrix, log_prior_coefficients, rwmh, Model, AdaptiveMetropolis
except ImportError:
    # Fallback if path manipulation didn't work as expected
    from Problem_solving.source_varying_functions import wind_function, s_function, A_matrix, log_prior_coefficients, rwmh, Model, AdaptiveMetropolis

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
    
    # Initial point near truth
    initial_point = np.array([1.0, 1.0, 0.0, 2.0, -2.0])
    
    # Pre-calculate winds for efficiency
    winds = wind_function(data['times'], consts['wa0'], consts['wa'], consts['wb'])
    
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
        ll = model.log_likelihood(params, data, precomputed_winds=winds)
        
        return lp_coeff + ll

    print("Running Adaptive MCMC...")
    sampler = AdaptiveMetropolis(
        target_log_prob=log_posterior,
        start_point=initial_point,
        t0=1000
    )
    chain, acc = sampler.sample(n_steps=10000)
    print(f"MCMC Acceptance Rate: {acc:.3f}")
    print(f"Learned Covariance Diagonal: {np.diag(sampler.cov)}")
    
    burn_in = 2000
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
    
    # --- Fit Static Model (Moved up for Plot 2) ---
    print("Fitting Static Model...")
    def log_posterior_static(flat_coeff):
        p_a0 = flat_coeff[0:1]
        p_XS = flat_coeff[1:2]
        p_YS = flat_coeff[2:3]
        params = {'a0': p_a0, 'XS': p_XS, 'YS': p_YS} # ak, bk default to 0
        lp = -0.5 * np.sum(p_a0**2) # Simple prior
        ll = model.log_likelihood(params, data, precomputed_winds=winds)
        return lp + ll
        
    sampler_static = AdaptiveMetropolis(
        target_log_prob=log_posterior_static,
        start_point=np.array([0.5, 0.0, 0.0]),
        t0=500
    )
    chain_static, _ = sampler_static.sample(n_steps=3000)
    mean_static = np.mean(chain_static[1000:], axis=0)

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
        s_t = s_function(t_plot, p_ak, p_bk, p_a0)
        # s_function returns (Nt, 1) here, flatten to (Nt,)
        s_samples.append(s_t.flatten())
        
    s_samples = np.array(s_samples)
    s_mean = np.mean(s_samples, axis=0)
    s_lower = np.percentile(s_samples, 2.5, axis=0)
    s_upper = np.percentile(s_samples, 97.5, axis=0)
    s_true = s_function(t_plot, true_ak, true_bk, true_a0).flatten()
    
    # Calculate Static Model Source
    p_stat_a0 = mean_static[0:1]
    s_static_vals = s_function(t_plot, np.array([0.0]), np.array([0.0]), p_stat_a0).flatten()
    
    plt.figure(figsize=(10, 6))
    plt.plot(t_plot, s_true, 'k-', linewidth=2, label='True Source')
    plt.plot(t_plot, s_mean, 'b--', linewidth=2, label='Posterior Mean (Dynamic)')
    plt.plot(t_plot, s_static_vals, 'r:', linewidth=3, label='Static Model')
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
    # Static model already fitted above (mean_static)
    
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
    s_stat = s_function(t_max, np.array([0.0]), np.array([0.0]), p_stat_a0)
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
    
    # --- PLOT 5: Side-by-Side Comparison (Snapshot) ---
    print("Generating Side-by-Side Comparison (Dynamic Data)...")
    # Use max emission time for snapshot
    t_snap = t_max
    wind_snap = wind_function(t_snap, consts['wa0'], consts['wa'], consts['wb'])
    
    # True Data
    A_true = A_matrix(data['X_flat'], data['Y_flat'], consts, wind_vector=wind_snap)
    s_true_val = s_function(t_snap, true_ak, true_bk, true_a0)
    mu_true = (np.dot(s_true_val, A_true) + 1.0).reshape(Nx, Nx)
    
    # Dynamic Model Mean
    A_dyn = A_matrix(data['X_flat'], data['Y_flat'], consts_dyn, wind_vector=wind_snap)
    s_dyn_val = s_function(t_snap, p_dyn_ak, p_dyn_bk, p_dyn_a0)
    mu_dyn_snap = (np.dot(s_dyn_val, A_dyn) + 1.0).reshape(Nx, Nx)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    vmax = max(np.max(mu_true), np.max(mu_dyn_snap))
    im1 = axes[0].contourf(data['X1'], data['X2'], mu_true, levels=20, cmap='viridis', vmin=0, vmax=vmax)
    axes[0].set_title("True Data (Dynamic Source)")
    im2 = axes[1].contourf(data['X1'], data['X2'], mu_dyn_snap, levels=20, cmap='viridis', vmin=0, vmax=vmax)
    axes[1].set_title("Recovered Model (Dynamic Mean)")
    plt.colorbar(im1, ax=axes.ravel().tolist())
    plt.savefig(os.path.join(OUTPUT_DIR, 'dynamic_data_comparison.png'))
    plt.close()
    print(f"Saved {OUTPUT_DIR}/dynamic_data_comparison.png")


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
        
        # Pre-calculate winds
        winds = wind_function(data['times'], consts['wa0'], consts['wa'], consts['wb'])
        
        # Static
        def lp_stat(fc):
            return -0.5*np.sum(fc[0]**2) + model.log_likelihood({'a0':fc[0:1], 'XS':fc[1:2], 'YS':fc[2:3]}, data, precomputed_winds=winds)
        
        sampler_stat = AdaptiveMetropolis(lp_stat, np.array([0.5, 0.0, 0.0]), t0=200)
        chain_stat, _ = sampler_stat.sample(1000)
        mean_stat = np.mean(chain_stat[500:], axis=0)
        
        # Dynamic
        def lp_dyn(fc):
             p = {'a0':fc[0:1], 'ak':fc[1:2].reshape(1,1), 'bk':fc[2:3].reshape(1,1), 'XS':fc[3:4], 'YS':fc[4:5]}
             return log_prior_coefficients([p['a0'], p['ak'], p['bk']]) + model.log_likelihood(p, data, precomputed_winds=winds)
        
        sampler_dyn = AdaptiveMetropolis(lp_dyn, np.array([1.0, 1.0, 0.0, 2.0, -2.0]), t0=200)
        chain_dyn, _ = sampler_dyn.sample(1000)
        mean_dyn = np.mean(chain_dyn[500:], axis=0)
        
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


def run_static_data_analysis():
    print("Running Static Data Analysis (Dynamic Model on Static Data)...")
    np.random.seed(42)
    
    # Setup
    T, Nt, Nx, Lx = 1.0, 15, 15, 5.0
    consts = {
        'RHO_CH4': 0.656, 'U': 5.0,
        'XS': [2.0], 'YS': [-2.0], 'ZS': 0, 'Z': 0,
        'a_H': 1, 'b_H': 1, 'w': 1, 'a_V': 1, 'b_V': 1, 'h': 1,
        'gamma_H': 1, 'gamma_V': 1,
        # CONSTANT WIND for Static Analysis
        'wa0': np.array([0.0, 4.0]), 'wa': [np.array([0.0, 0.0])], 'wb': [np.array([0.0, 0.0])]
    }
    
    # STATIC Source: ak=0, bk=0
    true_a0 = np.array([1.0])
    true_ak = np.array([[0.0]])
    true_bk = np.array([[0.0]])
    
    model = Model(beta=1.0, sigma_epsilon=0.1, physical_constants=consts)
    data = model.gen_data(T, Nt, Nx, Lx, true_ak, true_bk, true_a0)
    
    # Pre-calculate winds
    winds = wind_function(data['times'], consts['wa0'], consts['wa'], consts['wb'])
    
    # --- 1. Fit Dynamic Model ---
    print("  Fitting Dynamic Model...")
    initial_point = np.array([1.0, 0.1, 0.1, 2.0, -2.0]) # Start with slight variation
    
    def log_posterior_dyn(flat_coeff):
        p_a0 = flat_coeff[0:1]
        p_ak = flat_coeff[1:2].reshape(1,1)
        p_bk = flat_coeff[2:3].reshape(1,1)
        p_XS = flat_coeff[3:4]
        p_YS = flat_coeff[4:5]
        params = {'a0': p_a0, 'ak': p_ak, 'bk': p_bk, 'XS': p_XS, 'YS': p_YS}
        lp = log_prior_coefficients([p_a0, p_ak, p_bk])
        ll = model.log_likelihood(params, data, precomputed_winds=winds)
        return lp + ll

    sampler_dyn = AdaptiveMetropolis(log_posterior_dyn, initial_point, t0=1000)
    chain_dyn, acc_dyn = sampler_dyn.sample(10000)
    print(f"  Dynamic MCMC Acceptance: {acc_dyn:.3f}")
    
    burn_in = 2000
    chain_dyn_burned = chain_dyn[burn_in:]
    mean_dyn = np.mean(chain_dyn_burned, axis=0)
    
    # --- 2. Fit Static Model ---
    print("  Fitting Static Model...")
    def log_posterior_stat(flat_coeff):
        p_a0 = flat_coeff[0:1]
        p_XS = flat_coeff[1:2]
        p_YS = flat_coeff[2:3]
        params = {'a0': p_a0, 'XS': p_XS, 'YS': p_YS}
        lp = -0.5 * np.sum(p_a0**2)
        ll = model.log_likelihood(params, data, precomputed_winds=winds)
        return lp + ll
        
    sampler_stat = AdaptiveMetropolis(log_posterior_stat, np.array([1.0, 2.0, -2.0]), t0=500)
    chain_stat, acc_stat = sampler_stat.sample(5000)
    print(f"  Static MCMC Acceptance: {acc_stat:.3f}")
    
    chain_stat_burned = chain_stat[1000:]
    mean_stat = np.mean(chain_stat_burned, axis=0)

    # --- Plot 1: Source Recovery ---
    print("  Generating Source Recovery Plot...")
    t_plot = np.linspace(0, T, 100)
    
    # True Source (Static)
    s_true = s_function(t_plot, true_ak, true_bk, true_a0).flatten()
    
    # Dynamic Estimate
    s_dyn_samples = []
    indices = np.random.choice(len(chain_dyn_burned), size=200, replace=False)
    for idx in indices:
        sample = chain_dyn_burned[idx]
        s_t = s_function(t_plot, sample[1:2].reshape(1,1), sample[2:3].reshape(1,1), sample[0:1])
        s_dyn_samples.append(s_t.flatten())
    s_dyn_samples = np.array(s_dyn_samples)
    s_dyn_mean = np.mean(s_dyn_samples, axis=0)
    s_dyn_lower = np.percentile(s_dyn_samples, 2.5, axis=0)
    s_dyn_upper = np.percentile(s_dyn_samples, 97.5, axis=0)
    
    # Static Estimate & CI
    s_stat_samples = []
    indices_stat = np.random.choice(len(chain_stat_burned), size=200, replace=False)
    for idx in indices_stat:
        sample = chain_stat_burned[idx]
        # Static model: ak=0, bk=0
        s_t = s_function(t_plot, np.array([[0.0]]), np.array([[0.0]]), sample[0:1])
        s_stat_samples.append(s_t.flatten())
    s_stat_samples = np.array(s_stat_samples)
    s_stat_mean = np.mean(s_stat_samples, axis=0)
    s_stat_lower = np.percentile(s_stat_samples, 2.5, axis=0)
    s_stat_upper = np.percentile(s_stat_samples, 97.5, axis=0)
    
    plt.figure(figsize=(10, 6))
    plt.plot(t_plot, s_true, 'k-', linewidth=2, label='True Source (Static)')
    
    # Dynamic
    plt.plot(t_plot, s_dyn_mean, 'b--', linewidth=2, label='Dynamic Model Mean')
    plt.fill_between(t_plot, s_dyn_lower, s_dyn_upper, color='blue', alpha=0.2, label='Dynamic 95% CI')
    
    # Static
    plt.plot(t_plot, s_stat_mean, 'r:', linewidth=3, label='Static Model Mean')
    plt.fill_between(t_plot, s_stat_lower, s_stat_upper, color='red', alpha=0.2, label='Static 95% CI')
    
    plt.xlabel('Time')
    plt.ylabel('Source Intensity s(t)')
    plt.title('Recovery of Static Source (Constant Wind)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, 'static_data_source_recovery.png'))
    plt.close()
    print(f"Saved {OUTPUT_DIR}/static_data_source_recovery.png")
    
    # --- Plot 2: Dynamic Parameters Chains ---
    print("  Generating MCMC Chains Plot...")
    plt.figure(figsize=(10, 10))
    labels = ['a0', 'ak', 'bk', 'XS', 'YS']
    true_vals = [true_a0[0], true_ak[0,0], true_bk[0,0], consts['XS'][0], consts['YS'][0]]
    
    for i in range(5):
        plt.subplot(5, 1, i+1)
        plt.plot(chain_dyn[:, i], label='Chain')
        plt.axhline(true_vals[i], color='r', linestyle='--', label='True')
        plt.ylabel(labels[i])
        plt.legend()
    plt.xlabel('Iteration')
    plt.suptitle('Dynamic Model Chains on Static Data')
    plt.savefig(os.path.join(OUTPUT_DIR, 'static_data_mcmc_chains.png'))
    plt.close()
    print(f"Saved {OUTPUT_DIR}/static_data_mcmc_chains.png")
    
    # --- Plot 3: Side-by-Side Comparison (Snapshot) ---
    print("  Generating Side-by-Side Comparison...")
    # Reconstruct Dynamic Model Mean
    p_dyn_a0 = mean_dyn[0:1]
    p_dyn_ak = mean_dyn[1:2].reshape(1,1)
    p_dyn_bk = mean_dyn[2:3].reshape(1,1)
    p_dyn_XS = mean_dyn[3:4]
    p_dyn_YS = mean_dyn[4:5]
    
    consts_dyn = consts.copy()
    consts_dyn['XS'] = p_dyn_XS
    consts_dyn['YS'] = p_dyn_YS
    
    # Use max time for snapshot
    t_snap = T
    wind_snap = wind_function(t_snap, consts['wa0'], consts['wa'], consts['wb'])
    
    # True Data (Static)
    A_true = A_matrix(data['X_flat'], data['Y_flat'], consts, wind_vector=wind_snap)
    s_true_val = s_function(t_snap, true_ak, true_bk, true_a0)
    mu_true = (np.dot(s_true_val, A_true) + 1.0).reshape(Nx, Nx)
    
    # Dynamic Model Mean
    A_dyn = A_matrix(data['X_flat'], data['Y_flat'], consts_dyn, wind_vector=wind_snap)
    s_dyn_val = s_function(t_snap, p_dyn_ak, p_dyn_bk, p_dyn_a0)
    mu_dyn = (np.dot(s_dyn_val, A_dyn) + 1.0).reshape(Nx, Nx)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    vmax = max(np.max(mu_true), np.max(mu_dyn))
    im1 = axes[0].contourf(data['X1'], data['X2'], mu_true, levels=20, cmap='viridis', vmin=0, vmax=vmax)
    axes[0].set_title("True Data (Static Source)")
    im2 = axes[1].contourf(data['X1'], data['X2'], mu_dyn, levels=20, cmap='viridis', vmin=0, vmax=vmax)
    axes[1].set_title("Recovered Model (Dynamic Mean)")
    plt.colorbar(im1, ax=axes.ravel().tolist())
    plt.savefig(os.path.join(OUTPUT_DIR, 'static_data_comparison.png'))
    plt.close()
    print(f"Saved {OUTPUT_DIR}/static_data_comparison.png")

def run_static_rmse_scaling_analysis():
    print("Running Static RMSE Scaling Analysis (Static Data)...")
    np.random.seed(42)
    grid_sizes = [2, 3, 4, 5, 6]
    
    rmse_intensity_static = []
    rmse_intensity_dyn = []
    rmse_loc_static = []
    rmse_loc_dyn = []
    
    est_locs_static = []
    est_locs_dyn = []
    
    T, Nt, Lx = 1.0, 10, 5.0
    consts = {
        'RHO_CH4': 0.656, 'U': 5.0,
        'XS': [2.0], 'YS': [-2.0], 'ZS': 0, 'Z': 0,
        'a_H': 1, 'b_H': 1, 'w': 1, 'a_V': 1, 'b_V': 1, 'h': 1,
        'gamma_H': 1, 'gamma_V': 1,
        # CONSTANT WIND for Static Analysis
        'wa0': np.array([0.0, 4.0]), 'wa': [np.array([0.0, 0.0])], 'wb': [np.array([0.0, 0.0])]
    }
    # Static Source
    true_a0 = np.array([1.0])
    true_ak = np.array([[0.0]])
    true_bk = np.array([[0.0]])
    true_loc = np.array([consts['XS'][0], consts['YS'][0]])
    
    model = Model(beta=1.0, sigma_epsilon=0.1, physical_constants=consts)
    
    for Nx in grid_sizes:
        print(f"  Grid {Nx}x{Nx}...")
        data = model.gen_data(T, Nt, Nx, Lx, true_ak, true_bk, true_a0)
        winds = wind_function(data['times'], consts['wa0'], consts['wa'], consts['wb'])
        
        # Static Model Fit
        def lp_stat(fc):
            return -0.5*np.sum(fc[0]**2) + model.log_likelihood({'a0':fc[0:1], 'XS':fc[1:2], 'YS':fc[2:3]}, data, precomputed_winds=winds)
        
        sampler_stat = AdaptiveMetropolis(lp_stat, np.array([0.5, 0.0, 0.0]), t0=200)
        chain_stat, _ = sampler_stat.sample(1000)
        mean_stat = np.mean(chain_stat[500:], axis=0)
        
        # Dynamic Model Fit
        def lp_dyn(fc):
             p = {'a0':fc[0:1], 'ak':fc[1:2].reshape(1,1), 'bk':fc[2:3].reshape(1,1), 'XS':fc[3:4], 'YS':fc[4:5]}
             return log_prior_coefficients([p['a0'], p['ak'], p['bk']]) + model.log_likelihood(p, data, precomputed_winds=winds)
        
        sampler_dyn = AdaptiveMetropolis(lp_dyn, np.array([1.0, 0.1, 0.1, 2.0, -2.0]), t0=200)
        chain_dyn, _ = sampler_dyn.sample(1000)
        mean_dyn = np.mean(chain_dyn[500:], axis=0)
        
        # Metrics
        t_eval = np.linspace(0, T, 50)
        s_true = np.array([s_function(t, true_ak, true_bk, true_a0)[0] for t in t_eval])
        s_stat = np.array([s_function(t, [0], [0], [mean_stat[0]])[0] for t in t_eval])
        s_dyn = np.array([s_function(t, [mean_dyn[1]], [mean_dyn[2]], [mean_dyn[0]])[0] for t in t_eval])
        
        rmse_intensity_static.append(np.sqrt(np.mean((s_true - s_stat)**2)))
        rmse_intensity_dyn.append(np.sqrt(np.mean((s_true - s_dyn)**2)))
        
        loc_stat = mean_stat[1:3]
        loc_dyn = mean_dyn[3:5]
        rmse_loc_static.append(np.linalg.norm(loc_stat - true_loc))
        rmse_loc_dyn.append(np.linalg.norm(loc_dyn - true_loc))
        
        est_locs_static.append(loc_stat)
        est_locs_dyn.append(loc_dyn)
        
    # Plotting RMSE
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sensors = [n*n for n in grid_sizes]
    
    axes[0].plot(sensors, rmse_intensity_static, 'r--o', label='Static Model')
    axes[0].plot(sensors, rmse_intensity_dyn, 'b-o', label='Dynamic Model')
    axes[0].set_xlabel('Number of Sensors')
    axes[0].set_ylabel('RMSE (Source Intensity)')
    axes[0].set_title('Static Data: Intensity Recovery')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(sensors, rmse_loc_static, 'r--o', label='Static Model')
    axes[1].plot(sensors, rmse_loc_dyn, 'b-o', label='Dynamic Model')
    axes[1].set_xlabel('Number of Sensors')
    axes[1].set_ylabel('Location Error (m)')
    axes[1].set_title('Static Data: Location Error')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'static_data_rmse_scaling.png'))
    plt.close()
    print(f"Saved {OUTPUT_DIR}/static_data_rmse_scaling.png")
    
    # Plotting Convergence
    plt.figure(figsize=(10, 10))
    plt.plot(true_loc[0], true_loc[1], 'k*', markersize=20, label='True Source')
    
    stat_x = [p[0] for p in est_locs_static]
    stat_y = [p[1] for p in est_locs_static]
    dyn_x = [p[0] for p in est_locs_dyn]
    dyn_y = [p[1] for p in est_locs_dyn]
    
    plt.plot(stat_x, stat_y, 'r--', alpha=0.3)
    plt.plot(dyn_x, dyn_y, 'b--', alpha=0.3)
    
    norm = plt.Normalize(min(sensors), max(sensors))
    
    for i, (x, y) in enumerate(zip(stat_x, stat_y)):
        n_s = sensors[i]
        color = plt.cm.Reds(norm(n_s))
        plt.scatter(x, y, s=150, edgecolors=color, facecolors='none', linewidths=2, zorder=3)
        plt.annotate(f"{n_s}", (x, y), xytext=(-15, 10), textcoords='offset points', fontsize=10, color='darkred')
        
    for i, (x, y) in enumerate(zip(dyn_x, dyn_y)):
        n_s = sensors[i]
        color = plt.cm.Blues(norm(n_s))
        plt.scatter(x, y, s=150, facecolors=color, edgecolors='k', zorder=3)
        plt.annotate(f"{n_s}", (x, y), xytext=(5, 10), textcoords='offset points', fontsize=10, color='darkblue')

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='*', color='w', markerfacecolor='k', markersize=15, label='True Source'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='none', markeredgecolor='r', markersize=10, markeredgewidth=2, label='Static (Empty)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='b', markeredgecolor='k', markersize=10, label='Dynamic (Filled)')
    ]
    
    plt.title(f"Static Data: Localization Convergence\n(Labels = Number of Sensors)", fontsize=14)
    plt.xlabel("X Coordinate (m)")
    plt.ylabel("Y Coordinate (m)")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(handles=legend_elements, loc='upper right')
    plt.axis('equal')
    
    all_x = stat_x + dyn_x + [true_loc[0]]
    all_y = stat_y + dyn_y + [true_loc[1]]
    plt.xlim(min(all_x)-1, max(all_x)+1)
    plt.ylim(min(all_y)-1, max(all_y)+1)
    
    plt.savefig(os.path.join(OUTPUT_DIR, 'static_data_location_convergence.png'), dpi=300)
    plt.close()
    print(f"Saved {OUTPUT_DIR}/static_data_location_convergence.png")

if __name__ == "__main__":
    run_constant_source_demo()
    run_varying_source_demo()
    run_inference_and_plots()
    run_rmse_scaling_analysis()
    run_static_data_analysis()
    run_static_rmse_scaling_analysis()
