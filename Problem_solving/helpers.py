import numpy as np
from math import cos, sin
from scipy.integrate import dblquad
import scipy

def s_function(t,ak,bk,a0):
    n_coeff = ak.shape[0]
    constant = a0
    cosines = [cos(2*np.pi*(k+1)*t) for k in range(n_coeff)]
    sines = [sin(2*np.pi*(k+1)*t) for k in range(n_coeff)]
    return np.dot(ak,cosines) + np.dot(bk,sines) + a0

# def A_matrix(x_1s,x_2s,x_1,x_2):
#     return np.exp(-((x_1s-x_1)**2+(x_2s-x_2)**2))

def box_mask(X, Y, ax, bx, ay, by):
    mask = ((X >= ax) & (X <= bx) & (Y >= ay) & (Y <= by))
    return 1 - mask 

# def box_mask(x, y, ax, bx, ay, by):
#     """1 outside the box, 0 inside:  [ax,bx] × [ay,by]."""
#     return ((x < ax) | (x > bx) | (y < ay) | (y > by)).astype(float)


def gaussian_2d(x, y, mu_x, mu_y, sigma):
    """Isotropic 2D Gaussian centered at (mu_x, mu_y)."""
    coeff = 1 / (2 * np.pi * sigma**2)
    exponent = -((x - mu_x)**2 + (y - mu_y)**2) / (2 * sigma**2)
    return coeff * np.exp(exponent)

# def A_obstacle(
#                mu_x, mu_y, 
#                x,y,
#                o1_x, o1_y, 
#                o2_x, o2_y,
#                sigma=2,
#                h=2):
#     """
#     Returns obstacle influence:
#     Sum of Gaussians at two obstacle points,
#     multiplied by a mask that blocks a rectangular region.
#     """
    
#     # Gaussian contributions from the two obstacles
#     G1 = gaussian_2d(mu_x, mu_y, o1_x, o1_y,  sigma)
#     G2 = gaussian_2d( mu_x, mu_y, o2_x, o2_y, sigma)
#     Gsum = G1 + G2
    
#     # Mask box defined by obstacle coordinates (+h padding)
#     # M = box_mask(x, y,
#     #              ax=min(o1_x, o2_x),
#     #              bx=max(o1_x, o2_x),
#     #              ay=min(o1_y, o2_y),
#     #              by=max(o1_y, o2_y) + h)
#     M = box_mask(x, y, o1_x, o2_x, o1_y, o1_y+h)
    
#     return Gsum * M


def A_obstacle(x_1s,x_2s,x_1,x_2, x_1o1, x_2o1, x_1o2, x_2o2, h=2, sigma=2):
    # dist1 = np.sqrt((x_1s - x_1o1)**2 + (x_2s - x_2o1)**2)
    # dist2 = np.sqrt((x_1s - x_1o2)**2 + (x_2s - x_2o2)**2)
    # sum_dist = dist1 + dist2
    # A = gaussian_2d(x_1o1, x_2o1, x_1, x_2, sigma)
    # B = gaussian_2d(x_1o2, x_2o2, x_1, x_2, sigma)

    return (1/((2* np.pi**2 * 0.5**2))*(np.exp(-((x_1o1-x_1)**2+(x_2o1-x_2)**2)) +
    np.exp(-((x_1o2-x_1)**2+(x_2o2-x_2)**2))) *
    box_mask(x_1, x_2, x_1o1, x_1o2, x_2o1, x_2o2+h))
    # return ((A + B) *
    # box_mask(x_1, x_2, x_1o1, x_1o2, x_2o1, x_2o2+h))

# def A_source(x_1s,x_2s,x_1,x_2):
#     # return (scipy.stats.norm.pdf(x_1, loc=x_1s, scale=2) *
#     #     scipy.stats.norm.pdf(x_2, loc=x_2s, scale=2))
#     return (1/(np.sqrt(2* np.pi**2 * 2**2)  )) *np.exp(-(1/2**2)*(x_1s-x_1)**2-(1/2**2)*(x_2s-x_2)**2) 

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

