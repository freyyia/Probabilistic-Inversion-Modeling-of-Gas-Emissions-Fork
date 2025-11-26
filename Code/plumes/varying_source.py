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
a0 =4
ak = np.array([3,2,1])
bk = np.array([3,2,1])
t = 1

print(s_function(t,ak,bk,a0))
#Plots source function over [0,2pi]
t = np.linspace(0,0.5,100)
plt.plot(t,list(map(lambda t: s_function(t,ak,bk,a0),t)))
plt.show()


#Defines the coupling matrix that  maps source (x_s,y_s) to concentration at (x,y)
# 
def A_matrix(x_1s,x_2s,x_1,x_2):
    return np.exp(-((x_1s-x_1)**2+(x_2s-x_2)**2))

beta = 1
sigma_epsilon = 0.01
# Define model class y(t,x)=A(x)s(t)+beta+epsilon
class Model:
    def __init__(self,x_1s,x_2s,beta,sigma_epsilon,s_function, A_matrix):
        self.x_1s = x_1s
        self.x_2s = x_2s
        self.beta = beta
        self.sigma_epsilon = sigma_epsilon
        self.s_function = s_function
        self.A_matrix = A_matrix
    def y(self,x_1,x_2,t):
        return self.A_matrix(self.x_1s,self.x_2s,x_1,x_2)*self.s_function(t,ak,bk,a0)+self.beta+np.random.normal(0,self.sigma_epsilon)

#Tests the model
x_1s =0
x_2s =0
model = Model(x_1s,x_2s,beta,sigma_epsilon,s_function,A_matrix)
t=0.5
x_1 = 1
x_2 = 1
y = model.y(x_1,x_2,t)
print(y)
#Plots over spatial grid for fixed time t
x_1 = np.linspace(-1,1,100)
x_2 = np.linspace(-1,1,100)
X_1,X_2 = np.meshgrid(x_1,x_2)
Y = np.array([model.y(x_1,x_2,t) for x_1,x_2 in zip(X_1.flatten(),X_2.flatten())]).reshape(X_1.shape)
plt.contourf(X_1,X_2,Y)
plt.colorbar()
plt.show()
# Now varies time in plot too
import matplotlib.animation as animation

# Set up the figure and axis
fig, ax = plt.subplots()
x_1 = np.linspace(-1, 1, 100)
x_2 = np.linspace(-1, 1, 100)
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

# Generate data
def gen_data(model,T):
    x_1 = np.linspace(-1, 1, 100)
    x_2 = np.linspace(-1, 1, 100)
    X_1, X_2 = np.meshgrid(x_1, x_2)
    Y = np.array([])
    for t in np.linspace(0,T,100):
        Yt = np.array([model.y(x_1, x_2, t) for x_1, x_2 in zip(X_1.flatten(), X_2.flatten())]).reshape(X_1.shape)
        Y = np.append(Y,Yt)


# Function for RWMHS given starting point, variance, time steps and posterior
def rwmh(start_point, proposal_variance, n_steps, log_posterior):
    """
    Random Walk Metropolis-Hastings Sampler.

    Args:
        start_point (np.array): Starting point for the chain.
        proposal_variance (float or np.array): Variance of the Gaussian proposal distribution.
        n_steps (int): Number of iterations.
        log_posterior (function): Function that returns the log posterior probability of a point.

    Returns:
        chain (np.array): The MCMC chain.
        acceptance_rate (float): The acceptance rate of the chain.
    """
    current_point = np.array(start_point)
    current_log_prob = log_posterior(current_point)
    
    chain = [current_point]
    accepted_count = 0
    
    for _ in range(n_steps):
        # Propose a new point
        proposal = current_point + np.random.normal(0, np.sqrt(proposal_variance), size=current_point.shape)
        proposal_log_prob = log_posterior(proposal)
        
        # Calculate acceptance probability
        # We use log scale for numerical stability
        acceptance_ratio = proposal_log_prob - current_log_prob
        
        # Accept or reject
        if np.log(np.random.rand()) < acceptance_ratio:
            current_point = proposal
            current_log_prob = proposal_log_prob
            accepted_count += 1
            
        chain.append(current_point)
        
    acceptance_rate = accepted_count / n_steps
    return np.array(chain), acceptance_rate

#%%
# Generate data  on grid
data = gen_data(model,10)
#Define log prior
def log_prior_coefficients(coeff):
    a0 = coeff[0]
    ak = coeff[1]
    bk = coeff[2]
    n_coeff = len(ak)
    variance_k = [1/(1+(k+1)**2) for k in range(n_coeff)]
    return -1/2 * np.sum((ak)**2/variance_k) -1/2 * np.sum((bk)**2/variance_k) -1/2 * (a0)**2

coefficients = [a0,ak,bk]
print(log_prior_coefficients(coefficients))

# %%
def log_likelihood_y(coeff,data,model):
    y = model.y(coeff[0],coeff[1],coeff[2])