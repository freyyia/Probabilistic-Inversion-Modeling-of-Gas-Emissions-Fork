# DEfine model y(t,x)=A(x)s(t)+beta+epsilon

import numpy as np
import matplotlib.pyplot as plt
from math import cos, sin
def s_function(t,ak,bk,a0):
    n_coeff = len(ak)
    constant = a0
    cosines = [cos(2*np.pi*(k+1)*t) for k in range(n_coeff)]
    sines = [sin(2*np.pi*(k+1)*t) for k in range(n_coeff)]
    return np.dot(ak,cosines) + np.dot(bk,sines) + a0


#Test s_function
a0 =1
ak = np.array([1,2,3])
bk = np.array([4,5,6])
t = 1

print(s_function(t,ak,bk,a0))
#Plots source function over [0,2pi]
t = np.linspace(0,2*np.pi,100)
plt.plot(t,list(map(lambda t: s_function(t,ak,bk,a0),t)))
plt.show()


#Defines the coupling matrix that  maps source (x_s,y_s) to concentration at (x,y)
# 
def A_matrix(x_1s,x_2s,x_1,x_2):
    return np.exp(-((x_1s-x_1)**2+(x_2s-x_2)**2))

beta = 1
sigma_epsilon = 0.1
# Define model class y(t,x)=A(x)s(t)+beta+epsilon
class Model:
    def __init__(self,x_1s,x_2s,beta,sigma_epsilon,s_function):
        self.x_1s = x_1s
        self.x_2s = x_2s
        self.beta = beta
        self.sigma_epsilon = sigma_epsilon
        self.s_function = s_function
    def y(self,x_1,x_2,t):
        return A_matrix(self.x_1s,self.x_2s,x_1,x_2)*self.s_function(t,ak,bk,a0)+self.beta+np.random.normal(0,self.sigma_epsilon)


