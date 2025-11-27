from helpers import A_obstacle_norm, s_function, box_mask
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

#%%
# sys.path.append(os.path.abspath("../"))
a0 =4
ak = np.array([1])
bk = np.array([1])
beta = 1
sigma_epsilon = 0.01
# Define model class y(t,x)=A(x)s(t)+beta+epsilon
class ModelObstacle:
    def __init__(self,x_1s,x_2s,
                 x_1o1, x_2o1, x_1o2, x_2o2,
                 beta,sigma_epsilon,s_function, A_matrix):
        self.x_1s = x_1s
        self.x_2s = x_2s
        self.x_1o1 = x_1o1
        self.x_2o1 = x_2o1
        self.x_1o2 = x_1o2
        self.x_2o2 = x_2o2
        self.beta = beta
        self.sigma_epsilon = sigma_epsilon
        self.s_function = s_function
        self.A_matrix = A_matrix
    def y(self,x_1,x_2,t):
        #x_1s,x_2s,x1,x2, x_1o1, x_2o1, x_1o2, x_2o2, domain_start, domain_end
        return (self.A_matrix(self.x_1s,self.x_2s,x_1,x_2,
                            self.x_1o1, self.x_2o1,
                            self.x_1o2, self.x_2o2,
                            )*self.s_function(t,ak,bk,a0) +
                            self.beta+np.random.normal(0,self.sigma_epsilon))




#%%

#Tests the model
x_1s =5
x_2s =4
x_1o1 = 4
x_2o1 = 7
x_1o2 = 6
x_2o2 = 7




t=0.5
#Plots over spatial grid for fixed time t
x_1 = np.linspace(0,10,100)
x_2 = np.linspace(0,10,100)
X_1,X_2 = np.meshgrid(x_1,x_2)

model = ModelObstacle(x_1s,x_2s, x_1o1, x_2o1, x_1o2, x_2o2,
                      beta,sigma_epsilon,s_function,A_obstacle_norm)


Y = np.array([model.y(x_1,x_2,t) for x_1,x_2 in zip(X_1.flatten(),X_2.flatten())]).reshape(X_1.shape)
plt.contourf(X_1,X_2,Y)
plt.colorbar()
plt.show()

# #%%
# from scipy.integrate import dblquad
# f = lambda y, x: model.A_matrix(x_1s,x_2s,y, x, x_1o1, x_2o1, x_1o2, x_2o2)
# result, err = dblquad(f, 0, 10, 0, 10)
# print("Integral over domain:", result)

# # %%
# x = 5
# y = 0
# #model.A_matrix(x_1s,x_2s,x, y, x_1o1, x_2o1, x_1o2, x_2o2)
# #A_obstacle(x_1s,x_2s,x, y, x_1o1, x_2o1, x_1o2, x_2o2)
# box_mask(x, y, x_1o1, x_1o2, x_2o1, x_2o2+2)

# # %%
