import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from source_varying_functions import A_matrix
import numpy as np

# Constants
physical_constants = {
    'RHO_CH4': 0.656,
    'U': 5.0,
    'wind_vector': np.array([0,1]),
    'SIGMA_H': 10.0,
    'SIGMA_V': 10.0,
    'N_REFL': 5,
    'P': 1000.0,
    'XS': 0.0,
    'YS': 0.0,
    'ZS': 0,
    'Z': 0,
    'a_H': 1,
    'b_H': 1,
    'w': 1,
    'a_V': 1,
    'b_V': 1,
    'h': 1,
    'gamma_H': 1,
    'gamma_V': 1
}

# Create a grid of points
Nx = 20
Lx = 5
x_1 = np.linspace(-Lx, Lx, Nx)
x_2 = np.linspace(-Lx, Lx, Nx)
X_1, X_2 = np.meshgrid(x_1, x_2)

# Calculate A_matrix using array input
print("Calculating A_matrix with array input...")
A_array = A_matrix(X_1, X_2, physical_constants)

# Calculate A_matrix using scalar input
print("Calculating A_matrix with scalar input...")
A_scalar = np.zeros_like(A_array)
for i in range(A_array.shape[0]):
    for j in range(A_array.shape[1]):
        A_scalar[i, j] = A_matrix(X_1[i, j], X_2[i, j], physical_constants)

# Compare
diff = np.abs(A_array - A_scalar)
max_diff = np.max(diff)
print(f"Max difference: {max_diff}")

if max_diff > 1e-10:
    print("MISMATCH DETECTED!")
    # Find where mismatch is
    indices = np.where(diff > 1e-10)
    print(f"Indices of mismatch: {indices}")
    print(f"Array value at mismatch: {A_array[indices][0]}")
    print(f"Scalar value at mismatch: {A_scalar[indices][0]}")
    print(f"X1 at mismatch: {X_1[indices][0]}")
    print(f"X2 at mismatch: {X_2[indices][0]}")
else:
    print("Consistency check passed.")
