# 20th June, 2026
# Created by Tian Hong Raymond Wu

# The purpose of this program is to implement (batch) Gradient Descent to minimise a loss function. Then, we compare the performance of GD with the coordinate scheme.

import numpy as np
import matplotlib.pyplot as plt
t_var = np.load("t_var.npy")
y_var = np.load("y_var.npy")


def create_W(p):
   ## generate W which is a p-2 x p matrix as defined in the question
    W = np.zeros((p-2, p))
    b = np.array([1, -2, 1])
    for i in range(p-2):
        W[i, i:i+3] = b 
    return W 



# ==================== MAIN ====================

# Total number of outputs
p = len(y_var)  

alpha = 1
lmbda = 0.001

y = y_var.reshape(p, 1)
W = create_W(p)
beta_gd = np.ones((p, 1))
beta_coord = np.ones((p, 1))

performance_gd = np.zeros((1000, 1))
performance_coord = np.zeros((1000, 1))


# ===== True Fit =====
beta_true = np.linalg.inv(np.identity(p) + 2 * lmbda * p * W.T @ W) @ y
L_true = 1/(2*p) * (y - beta_true).T @ (y - beta_true) + lmbda * (W @ beta_true).T @ (W @ beta_true)



# ===== Gradient Descent Method =====
# 1000 epochs (1 epoch = one pass over the entire data i.e. single GD step)
for i in range(1000):

    # Loss Function
    L_gd = 1/(2*p) * (y - beta_gd).T @ (y - beta_gd) + lmbda * (W @ beta_gd).T @ (W @ beta_gd)
    
    # Store the performance for the i-th epoch
    performance_gd[i] = L_gd - L_true


    # Calculate NEXT parameter values (i.e. beta)
    sum_delta_L = 0
    for j in range(0, p):
        # Standard Basis
        e_j = np.zeros((p, 1))
        e_j[j] = 1

        # Calculate delta_L
        sum_delta_L += -(y[j]-beta_gd[j]) * e_j + 2 * lmbda * W.T @ W @ beta_gd
    
    beta_gd = beta_gd - alpha/p * sum_delta_L



# ===== Coordinate Method =====

# Run the algorithm for 1000 updates (from j = 0, 1, ..., p and then back to 0)
for i in range(1000):
    # When i = p, 2p..., loop back to 0
    j = i % p

    # Loss Function
    L_coord = 1/(2*p) * (y - beta_coord).T @ (y - beta_coord) + lmbda * (W @ beta_coord).T @ (W @ beta_coord)
    
    # Store the performance for the i-th update
    performance_coord[i] = L_coord - L_true


    # Calculate the individual beta value (index j) that gives the minimum loss
    # NOTE: Because index begins from zero, beta_coord[p] does not exist, so we need to subtract by 1 when retrieving beta_coord, y etc.
    if j == 0:
        beta_new = (y[0]/p + 4*lmbda*beta_coord[1] - 2*lmbda*beta_coord[2])/(1/p + 2*lmbda)
    
    elif j == 1:
        beta_new = (y[1]/p + 4*lmbda*beta_coord[0] + 8*lmbda*beta_coord[2] - 2*lmbda*beta_coord[3])/(1/p + 10*lmbda)

    elif j == p-2:
        beta_new = (y[p-2]/p + 4*lmbda*beta_coord[p-1] + 8*lmbda*beta_coord[p-3] - 2*lmbda*beta_coord[p-4])/(1/p + 10*lmbda)
    
    elif j == p-1:
        beta_new = (y[p-1]/p + 4*lmbda*beta_coord[p-2] - 2*lmbda*beta_coord[p-3])/(1/p + 2*lmbda)

    else:
        beta_new = (y[j]/p - 2*lmbda*beta_coord[j-2] + 6*lmbda*beta_coord[j-1] + 8*lmbda*beta_coord[j+1] - 2*lmbda*beta_coord[j+2])/(1/p + 12*lmbda)

    # Update the current beta value with the beta value that gave the minimum loss
    beta_coord[j] = beta_new


# Plot the difference
k = np.arange(0, 1000)

plt.plot(k, performance_gd, label='GD', color='blue')
plt.plot(k, performance_coord, label='Coord', color='green')
plt.title("Comparison of Coordinate Scheme and (Batch) Gradient Descent")
plt.xlabel('k')
plt.ylabel('True & Fit Difference')

plt.legend()
plt.show()
