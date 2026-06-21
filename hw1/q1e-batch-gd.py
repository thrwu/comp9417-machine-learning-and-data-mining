# 20th June, 2026
# Created by Tian Hong Raymond Wu

# The purpose of this program is to implement (batch) Gradient Descent to minimise a loss function. Nine separate step sizes are plotted and compared.

# Given a loss function and data sets output, we want to find parameters (i.e. beta = p x 1 array) such that our linear model (y = mx + b  OR  y = Ax+b) fits the data set the most. To achieve this, we are given a loss function (first term data fit, second term some sort of filtering for rapid changes in the data) that we must minimise to near zero by taking its derivative as 0.

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

# Create a figure with 3x3 subplots
fig, axs = plt.subplots(3, 3)


# Total number of outputs
p = len(y_var)  

alphas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 0.6, 1.2, 2]
lmbda = 0.001

y = y_var.reshape(p, 1)
W = create_W(p)
beta_batch_gd = np.ones((p, 1))

# Loop through every alpha and plot the output
for row in range(3):
    for col in range(3):
        # Get alpha from the array 'alphas' corresponding to 3x3 matrix (row * width + col)
        alpha = alphas[row * 3 + col]

        # 1000 epochs (1 epoch = one pass over the entire data i.e. single GD step)
        for i in range(1000):

            # Batch GD Loss Function
            L_batch_gd = 1/(2*p) * (y - beta_batch_gd).T @ (y - beta_batch_gd) + lmbda * (W @ beta_batch_gd).T @ (W @ beta_batch_gd)

            # True Fit
            beta_true = np.linalg.inv(np.identity(p) + 2 * lmbda * p * W.T @ W) @ y
            L_true = 1/(2*p) * (y - beta_true).T @ (y - beta_true) + lmbda * (W @ beta_true).T @ (W @ beta_true)

            # Plot the difference
            axs[row, col].scatter(i, L_batch_gd - L_true, color='red', marker='*', s=1)
            axs[row, col].set_title(f"alpha (step size) = {alpha}")
            axs[row, col].set_xlabel('Epoch Num.')
            axs[row, col].set_ylabel('True & Fit Difference')

            # Calculate NEXT parameter values (i.e. beta)
            sum_delta_L = 0
            for j in range(0, p):
                # Standard Basis
                e_j = np.zeros((p, 1))
                e_j[j] = 1

                # Calculate delta_L
                sum_delta_L += -(y[j]-beta_batch_gd[j]) * e_j + 2 * lmbda * W.T @ W @ beta_batch_gd
            
            beta_batch_gd = beta_batch_gd - alphas[row*3+col]/p * sum_delta_L

plt.tight_layout()
plt.show()
