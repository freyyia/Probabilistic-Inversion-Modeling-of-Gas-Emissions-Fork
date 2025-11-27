
import sys
import os
import numpy as np

# Add the directory to path so we can import varying_source
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'Code/plumes')))

# Import A_matrix from varying_source
# We need to make sure we can import it. Since varying_source is a script, importing it might run it.
# But we can try.
# Alternatively, I will copy the A_matrix function and constants here to test them in isolation,
# or just modify varying_source to only run main code if __name__ == "__main__".
# But modifying varying_source structure might be too much.
# Let's try to exec the file and get the function.

file_path = 'Code/plumes/varying_source.py'
with open(file_path) as f:
    code = f.read()

# Execute the code in a dictionary
# Mock source_varying_functions
import sys
import unittest
from unittest.mock import MagicMock
mock_svf = MagicMock()
# Setup return values for imported functions if needed
# s_function is used in varying_source.py
def mock_s_function(t, ak, bk, a0):
    return 1.0
mock_svf.s_function = mock_s_function
mock_svf.A_matrix = MagicMock() # This will be shadowed by our implementation anyway
mock_svf.rwmh = MagicMock(return_value=([], 0.5)) # Return empty chain and acceptance rate
mock_svf.log_likelihood_y = MagicMock(return_value=0.0)

sys.modules['source_varying_functions'] = mock_svf

global_vars = {'__file__': file_path}

with unittest.mock.patch('matplotlib.pyplot.show'), unittest.mock.patch('matplotlib.pyplot.plot'), unittest.mock.patch('matplotlib.pyplot.contourf'), unittest.mock.patch('matplotlib.animation.FuncAnimation'), unittest.mock.patch('matplotlib.pyplot.colorbar'):

    exec(code, global_vars)

A_matrix = global_vars['A_matrix']
RHO_CH4 = global_vars['RHO_CH4']
U = global_vars['U']
SIGMA_H = global_vars['SIGMA_H']
SIGMA_V = global_vars['SIGMA_V']
H = global_vars['H']
Z = global_vars['Z']

print("Constants loaded:")
print(f"RHO_CH4={RHO_CH4}, U={U}, SIGMA_H={SIGMA_H}, SIGMA_V={SIGMA_V}, H={H}, Z={Z}")

# Test case 1: Directly downwind (y=y_s), same height as sensor (z is fixed at Z)
# But wait, z is fixed at Z=2 in the code.
# So if we are at y=y_s, delta_H = 0.
# delta_V = Z - H = 2 - 50 = -48.
# This is still far vertically.
# Let's test with x_s=0, y_s=0.
# Sensor at x=100, y=0.
x_s, y_s = 0, 0
x, y = 100, 0

val = A_matrix(x_s, y_s, x, y)
print(f"A_matrix({x_s}, {y_s}, {x}, {y}) = {val}")

# Calculate expected value manually
delta_H = 0
delta_V = Z - H # -48
term1 = (10**6 / RHO_CH4) * (1 / (2 * np.pi * U * SIGMA_H * SIGMA_V)) * np.exp(0)
# term1 = (1e6 / 0.656) * (1 / (2 * pi * 5 * 10 * 10))
#       = 1.524e6 * (1 / 3141.59) approx 485

term2_main = np.exp(-delta_V**2 / (2 * SIGMA_V**2))
# exp(-(-48)^2 / 200) = exp(-2304 / 200) = exp(-11.52) approx 1e-5

# Reflections
# j=1:
# num1 = (2*1*1000 + (-1)*(-48+50) - 50)^2 = (2000 - 2 - 50)^2 = (1948)^2
# exp1 = exp(-0.5 * 1948^2 / 100) -> 0
# num2 = (0 + 1*(-48+50) + 50)^2 = (2 + 50)^2 = 52^2 = 2704
# exp2 = exp(-0.5 * 2704 / 100) = exp(-13.52) approx 1.3e-6

# So total should be term1 * (term2_main + small_reflections)
# 485 * (1e-5 + 1.3e-6) approx 485 * 1.13e-5 approx 0.005

print(f"Expected approx: {485 * (np.exp(-11.52) + np.exp(-13.52))}")

