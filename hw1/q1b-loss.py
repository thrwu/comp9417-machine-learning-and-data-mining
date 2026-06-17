
import numpy as np
import matplotlib.pyplot as plt
t_var = np.load("t_var.npy")
y_var = np.load("y_var.npy")
#plt.plot(t_var, y_var)
#plt.show()

def create_W(p):
   ## generate W which is a p-2 x p matrix as defined in the question
    W = np.zeros((p-2, p))
    b = np.array([1, -2, 1])
    for i in range(p-2):
        W[i, i:i+3] = b 
    return W 

def loss(beta, y, W, L):
    ## compute loss for a given vector beta for data y, matrix W, regularization parameter L (lambda)
    # your code here 

    return 1/(2*p) * np.dot(y - beta, y - beta) + L * np.dot(W @ beta, W @ beta)



## your code here 

# Match the size of the output matrix 'y'
p = len(y_var)  

W = create_W(p)
L = 0.9
y = y_var

# Compute beta_hat
beta_hat = np.linalg.inv(np.identity(p) + 2 * L * p * W.T @ W) @ y


plt.plot(t_var, y_var, zorder=1, color='red', label='truth')
plt.plot(t_var, beta_hat, zorder=3, color='blue', 
            linewidth=2, linestyle='--', label='fit')
plt.legend(loc='best')
plt.title(f"L(beta_hat) = {loss(beta_hat, y, W, L)}")
plt.show()
