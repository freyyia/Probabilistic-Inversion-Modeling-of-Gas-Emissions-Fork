
import sys
import os
import numpy as np

# Add path to import source_varying_functions
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from source_varying_functions import A_matrix

def test_A_matrix_vectorization():
    # Mock constants
    physical_constants = {
        'RHO_CH4': 0.656,
        'U': 5.0,
        'wind_vector': np.array([0, 1]),
        'SIGMA_H': 10.0,
        'SIGMA_V': 10.0,
        'N_REFL': 5,
        'P': 1000.0,
        'XS': 0.0,
        'YS': 0.0,
        'ZS': 0,
        'Z': 0,
        'a_H': 1, 'b_H': 1, 'w': 1,
        'a_V': 1, 'b_V': 1, 'h': 1,
        'gamma_H': 1, 'gamma_V': 1
    }

    # Create grid
    Nx = 10
    x = np.linspace(-5, 5, Nx)
    y = np.linspace(-5, 5, Nx)
    X1, X2 = np.meshgrid(x, y)

    print(f"Testing A_matrix with inputs of shape {X1.shape}")

    try:
        result = A_matrix(X1, X2, physical_constants)
        print(f"Result shape: {result.shape}")
        if result.shape != X1.shape:
            print("FAIL: Result shape mismatch")
        else:
            print("SUCCESS: Result shape matches input")
            print("Sample value:", result[0,0])
    except Exception as e:
        print(f"FAIL: A_matrix raised exception: {e}")

if __name__ == "__main__":
    test_A_matrix_vectorization()
