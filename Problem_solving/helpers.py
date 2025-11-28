import numpy as np
from math import cos, sin
from scipy.integrate import dblquad
import scipy

def s_function(t,ak,bk,a0):
    n_coeff = ak.shape[0]
    cosines = [cos(2*np.pi*(k+1)*t) for k in range(n_coeff)]
    sines = [sin(2*np.pi*(k+1)*t) for k in range(n_coeff)]
    return np.dot(ak,cosines) + np.dot(bk,sines) + a0


def box_mask(X, Y, ax, bx, ay, by):
    mask = ((X >= ax) & (X <= bx) & (Y >= ay) & (Y <= by))
    return 1 - mask 

def gaussian_2d(x, y, mu_x, mu_y, sigma):
    """Isotropic 2D Gaussian centered at (mu_x, mu_y)."""
    coeff = 1 / (2 * np.pi * sigma**2)
    exponent = -((x - mu_x)**2 + (y - mu_y)**2) / (2 * sigma**2)
    return coeff * np.exp(exponent)


def A_obstacle(x_1s,x_2s,x_1,x_2, x_1o1, x_2o1, x_1o2, x_2o2, h=2, sigma=2):
    # dist1 = np.sqrt((x_1s - x_1o1)**2 + (x_2s - x_2o1)**2)
    # dist2 = np.sqrt((x_1s - x_1o2)**2 + (x_2s - x_2o2)**2)

    return (1/((2* np.pi**2 * 0.5**2))*(np.exp(-((x_1o1-x_1)**2+(x_2o1-x_2)**2)) +
    np.exp(-((x_1o2-x_1)**2+(x_2o2-x_2)**2))) *
    box_mask(x_1, x_2, x_1o1, x_1o2, x_2o1, x_2o2+h))
 
def A_source(x_1s, x_2s, x_1, x_2, sigma=2):
    """
    2D Gaussian centered at (x_1, x_2), evaluated at (x_1s, x_2s)
    with isotropic standard deviation = sigma.
    """
    coeff = 1 / (2 * np.pi * sigma**2)
    exponent = -((x_1s - x_1)**2 + (x_2s - x_2)**2) / (2 * sigma**2)
    return coeff * np.exp(exponent)

def A_obstacle_norm(x_1s,x_2s,x1,x2, x_1o1, x_2o1, x_1o2, x_2o2, dom_start=0, dom_end=10):

    f = lambda y, x: A_source(x_1s,x_2s,y,x)
    # result_box,_ = dblquad(f, x_1o1  + (x_2o1 - x_1o1)/2 , dom_end, 
    #                      x_1o1  + (x_2o1 - x_1o1)/2, dom_end, epsabs=1e-4)
    result_box,_ = dblquad(f, x_1o1, x_2o1, x_1o2, x_2o2 , epsabs=1e-4)
    # print("Integral over domain (source only part):", result_box)
    return (A_source(x_1s,x_2s,x1,x2) * (1-result_box) * 
        (box_mask(x1, x2, x_1o1, x_1o2, x_2o1, dom_end))
        + A_obstacle(x_1s,x_2s,x1,x2, x_1o1, x_2o1, x_1o2, x_2o2) * result_box/2 )

