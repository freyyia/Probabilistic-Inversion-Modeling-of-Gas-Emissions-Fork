
#%%
import sourceinversion.atmospheric_measurements as gp
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

#%%
#Parameters
Lx = 110.0
Ly = 110.0
x1_s = 50.0
x2_s = 50.0
w_speed = 6
w_direction = -70
Nt = 100
T = 1
sigma_s = 0.1
sigma_w = 10
rho_w = 0.5

grid = gp.Grid(
    x_range = (jnp.array(0.0), jnp.array(Lx)), 
    y_range = (jnp.array(0.0), jnp.array(Ly)),
    z_range= (jnp.array(0.0), jnp.array(0)),
    dx = jnp.array(0.1),
    dy = jnp.array(0.1),
    dz = jnp.array(1),
)

source_location = gp.SourceLocation(
    source_location_x = jnp.array([x1_s]),
    source_location_y = jnp.array([x2_s]),
    source_location_z = jnp.array([0.0]),
)
extra = [x1_s, x2_s, 0.0]

#Wind field
wind_field = gp.WindField(
    Ornstein_Uhlenbeck = False,
    initial_wind_speed = jnp.array(w_speed),
    initial_wind_direction = jnp.array(w_direction),
    end_wind_direction = jnp.array(-w_direction),
    number_of_time_steps = jnp.array(Nt),
    time_step = jnp.array(T),
    wind_speed_temporal_std = jnp.array(sigma_s),
    wind_direction_temporal_std = jnp.array(sigma_w),
    wind_temporal_correlation = jnp.array(rho_w),
    wind_speed_seed = 2,
    wind_direction_seed = 4,
)

#define atmospheric state
atmospheric_state = gp.AtmosphericState(
    emission_rate = jnp.array([0.00039]),                           # 0.00039kg/s = 1.41kg/h. To scale parameter like distances (0.00039 * 100_000) = 39.0
    source_half_width = jnp.array(0.5),                             # source is a square of 2m side length
    max_abl = jnp.array(1000.0),
    background_mean = jnp.array(2.0),       
    background_std = jnp.array(1e-2),       
    background_seed = jnp.array(56),
    background_filter = "power_law",        
    Gaussian_filter_kernel = 1,
    horizontal_opening_angle= 10.0,
    vertical_opening_angle = 10.0,
    a_horizontal = 1.0,
    a_vertical = 1.0,          
    b_horizontal = 1.0,
    b_vertical = 1.0,  
)

#define sensors settings
layout = "grid"
number_of_sensors = 36
# define a grid of points
p1 = (110,5,0)
p2 = (70,5,0)
p3 = (100,5,0)
p4 = (70,70,0)
sensor_location = gp.SensorsSettings.grid_of_sensors(
    p1, p2, p3, p4, number_of_sensors, layout)

# Plot the points in the grid
gp.SensorsSettings.plot_points_3d(sensor_location, extra, False)


sensors_settings =  gp.SensorsSettings(
    layout = layout,
    sensor_number = jnp.array(number_of_sensors),
    measurement_error_var = jnp.array(1e-6),
    sensor_seed = jnp.array(5),
    measurement_error_seed = jnp.array(420),
    sensor_locations = sensor_location,
)


#setup the gaussian plume model
gaussianplume = gp.GaussianPlume(grid, source_location, wind_field,
                                atmospheric_state, sensors_settings)

#this just extracts some parameters so that they don't have to be recomputed every time
fixed = gaussianplume.fixed_objects_of_gridfree_coupling_matrix()

# this computed the coupling matrix using the gaussian plume model
A = gaussianplume.temporal_gridfree_coupling_matrix(fixed)

# np.save('A_matrix.npy', A)

#%%

gaussianplume.gaussian_plume_plot(1)


background = gp.BackgroundGas(grid, source_location, atmospheric_state)
background.background_plot(save=False, format='png')
sensors = gp.Sensors(gaussianplume, background, sensors_settings)
# True source and atmospheric parameter values and sensor measurement data
truth = sensors.temporal_sensors_measurements(grided=False, beam=False)
# Data
data = truth[0]


#%%

colors = plt.cm.tab10(np.linspace(0, 1, 6))
plt.figure()
for i in range(36):
    color = colors[i // 6]
    plt.plot(data.reshape(36,-1)[i,:], 'o', color=color, label=f'Sensor {i+1}' if i < 6 else "")

# %%
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from source_varying_functions import s_function, A_matrix, rwmh, log_likelihood_y
import numpy as np
import matplotlib.pyplot as plt
from math import cos, sin
from collections.abc import Iterable
# s(t)=sum_k a_k cost(2pi kt) + sum_k b_k sin(2pi kt) + a_0



#Test s_function
a0 =4
ak = np.array([1])
bk = np.array([1])
t = 1

print(s_function(t,ak,bk,a0))
#Plots source function over [0,2pi]
t = np.linspace(0,0.5,100)
plt.plot(t,list(map(lambda t: s_function(t,ak,bk,a0),t)))
plt.show()


#Defines the coupling matrix that  maps source (x_s,y_s) to concentration at (x,y)
# 

Asaved = np.load('A_matrix.npy')


beta = 1
sigma_epsilon = 0.01
# Define model class y(t,x)=A(x)s(t)+beta+epsilon
class Model:
    def __init__(self,x1_s,x2_s,beta,sigma_epsilon,s_function, A_matrix):
        self.x1_s = x1_s
        self.x2_s = x2_s
        self.beta = beta
        self.sigma_epsilon = sigma_epsilon
        self.s_function = s_function
        self.A_matrix = A_matrix
    def y(self,x_1,x_2,t):
        return self.A_matrix(self.x1_s,self.x2_s,x_1,x_2)*self.s_function(t,ak,bk,a0)+self.beta+np.random.normal(0,self.sigma_epsilon)

#%%
#Tests the model
model = Model(x1_s,x2_s,beta,sigma_epsilon,s_function,A_matrix)
t=0.5
x_1 = 51
x_2 = 51
y = model.y(x_1,x_2,t)
print(y)
#Plots over spatial grid for fixed time t
x_1 = np.linspace(49,51,100)
x_2 = np.linspace(49,51,100)
X_1,X_2 = np.meshgrid(x_1,x_2)
Y = np.array([model.y(x_1,x_2,t) for x_1,x_2 in zip(X_1.flatten(),X_2.flatten())]).reshape(X_1.shape)
plt.contourf(X_1,X_2,Y)
plt.colorbar()
plt.show()

#%%
# Now varies time in plot too
import matplotlib.animation as animation

# Set up the figure and axis
fig, ax = plt.subplots()
x_1 = np.linspace(49, 51, 100)
x_2 = np.linspace(49, 51, 100)
X_1, X_2 = np.meshgrid(x_1, x_2)


# Initial plot
t_start = 0
Y = np.array([model.y(x_1, x_2, t_start) for x_1, x_2 in zip(X_1.flatten(), X_2.flatten())]).reshape(X_1.shape)
contour = ax.contourf(X_1, X_2, Y, levels=20, cmap='viridis')
cbar = fig.colorbar(contour)
ax.set_title(f'Observation Model at t={t_start:.2f}')

def update(frame):
    ax.clear()
    t = frame * 0.01  # Time step
    Y = np.array([model.y(x_1, x_2, t) for x_1, x_2 in zip(X_1.flatten(), X_2.flatten())]).reshape(X_1.shape)
    contour = ax.contourf(X_1, X_2, Y, levels=20, cmap='viridis')
    ax.set_title(f'Observation Model at t={t:.2f}')
    return contour,

# Create animation
ani = animation.FuncAnimation(fig, update, frames=50, interval=200)

# Save animation
try:
    ani.save('observation_model_animation.gif', writer='pillow')
    print("Animation saved as observation_model_animation.gif")
except Exception as e:
    print(f"Could not save animation: {e}")
    plt.show()

#%%
# Generate data
def gen_data(T):
    s_values = map(lambda t: s_function(t,ak,bk,a0), np.linspace(0,T,
            gaussianplume.wind_field.number_of_time_steps))

    measurements =  np.matmul(A, s_values) + 1 + np.random.normal(0, 0.1, 
    gaussianplume.sensors_settings.sensor_number * 
    gaussianplume.wind_field.number_of_time_steps)
    return measurements




# Function for RWMHS given starting point, variance, time steps and posterior


#%%
# Generate data  on grid
data = gen_data(10)
#Define log prior
def log_prior_coefficients(coeff):
    a0 = coeff[0]
    ak = np.atleast_1d(coeff[1])
    bk = np.atleast_1d(coeff[2])
    n_coeff = len(ak)
    variance_k = [1/(1+(k+1)**2) for k in range(n_coeff)]
    return -1/2 * np.sum((ak)**2/variance_k) -1/2 * np.sum((bk)**2/variance_k) -1/2 * (a0)**2

coefficients = [a0,ak,bk]
print(log_prior_coefficients(coefficients))
#y=As+beta+epsilon
# %%

# Test log_likelihood_y
ll = log_likelihood_y(coefficients, data, x1_s, x2_s, beta, sigma_epsilon, A_matrix)
print(f"Log likelihood: {ll}")



def log_posterior(coeff):
    return log_prior_coefficients(coeff) + log_likelihood_y(coeff, data, x1_s, x2_s, beta, sigma_epsilon, A_matrix)
    
#Run rwmh
initial_point = [0,0,0]
chain,acceptance_rate = rwmh(initial_point,1,10000,log_posterior)
print(chain)
print(acceptance_rate)
