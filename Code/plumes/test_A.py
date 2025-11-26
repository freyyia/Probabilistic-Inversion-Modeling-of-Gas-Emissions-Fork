
import sourceinversion.atmospheric_measurements as gp
import jax.numpy as jnp

#define grid
grid = gp.Grid(
    x_range = (jnp.array(0.0), jnp.array(110.0)), 
    y_range = (jnp.array(0.0), jnp.array(110.0)),
    z_range= (jnp.array(0), jnp.array(0)),
    dx = jnp.array(0.1),
    dy = jnp.array(0.1),
    dz = jnp.array(1),
)

#define source location
source_location = gp.SourceLocation(
    source_location_x = jnp.array([50.0]),
    source_location_y = jnp.array([50.0]),
    source_location_z = jnp.array([5.0]),
)

#define wind field
wind_field = gp.WindField(
    Ornstein_Uhlenbeck = False,
    initial_wind_speed = jnp.array(6.0),
    initial_wind_direction = jnp.array(-70),
    end_wind_direction = jnp.array(70),
    number_of_time_steps = jnp.array(100),
    time_step = jnp.array(1.0),
    wind_speed_temporal_std = jnp.array(0.1),
    wind_direction_temporal_std = jnp.array(10.0),
    wind_temporal_correlation = jnp.array(0.5),
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
p1 = (100,5,1)
p2 = (100,5,20)
p3 = (100,95,1)
p4 = (100,95,20)
sensor_location = gp.SensorsSettings.grid_of_sensors(
    p1, p2, p3, p4, number_of_sensors, layout)

# Plot the points in the grid
gp.SensorsSettings.plot_points_3d(sensor_location, False)


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



