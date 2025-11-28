from helpers import A_obstacle_norm, s_function, box_mask
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

#%%
# sys.path.append(os.path.abspath("../"))
a0 =np.array([4])
ak = np.array([1.5])
bk = np.array([0.5])
beta = 1
sigma_epsilon = 0.001
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
                            ) *self.s_function(t,ak,bk,a0) +
                             self.beta +np.random.normal(0,self.sigma_epsilon))
    # # Generate data at Nt time steps
    def gen_data(self,T,Nt,Nx,Lx,Hx):
        x_1 = np.linspace(Lx, Hx, Nx)
        x_2 = np.linspace(Lx, Hx, Nx)
        X_1, X_2 = np.meshgrid(x_1, x_2)
        Y_list = []
        for t in np.linspace(0,T,Nt):
            Yt = np.array([self.y(x_1, x_2, t) for x_1, x_2 in zip(X_1.flatten(), X_2.flatten())]).reshape(X_1.shape)
            Y_list.append(Yt)
        Y = np.array(Y_list)
        return {'X1': X_1, 'X2': X_2, 'Y': Y}




#%%

#Tests the model
x_1s =5
x_2s =5
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
plt.contourf(X_1,X_2,Y, levels=30)
plt.colorbar()
plt.show()



# %%
T=10
Nt=10
Nx=20
Lx=0
Hx=10
t_start = 0

#Plots over spatial grid for fixed time t
data = model.gen_data(T,Nt,Nx,Lx,Hx)
plt.contourf(data['X1'],data['X2'],data['Y'][0])
plt.colorbar()
# plt.show() replaced with savefig to avoid webview error
plt.savefig('source_function_plot_obstacle.png')
plt.close()
# Now varies time in plot too
#%%
import matplotlib.animation as animation

# Set up the figure and axis
fig, ax = plt.subplots()

# Initial plot
data = model.gen_data(T,Nt,Nx,Lx, Hx)
contour = ax.contourf(data['X1'], data['X2'], data['Y'][0], levels=20, cmap='viridis')
cbar = fig.colorbar(contour)
ax.set_title(f'Observation Model at t={t_start:.2f}')

def update(frame):
    ax.clear()
    t = frame * T / (Nt - 1)
    # data is already generated outside
    contour = ax.contourf(data['X1'], data['X2'], data['Y'][frame], levels=20, cmap='viridis')
    ax.set_title(f'Observation Model at t={t:.2f}')
    return contour,

# Create animation
ani = animation.FuncAnimation(fig, update, frames=Nt, interval=200)

# Save animation
try:
    ani.save('obstacle_Model_animation.gif', writer='pillow')
    print("Animation saved as obstacle_Model_animation.gif")
except Exception as e:
    print(f"Could not save animation: {e}")
    # plt.show() replaced with savefig to avoid webview error



# %%
