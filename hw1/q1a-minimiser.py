# 17th June, 2026
# Created by Tian Hong Raymond Wu

# The purpose of this program is to perform gradient based optimisation to find the minimiser of the function (shown in instructions pdf document resembling y=Ax-b).


import numpy as np
import sys

print("PROGRAM BEGIN")

# Step Size
alpha = 0.01

# ?
gamma = 2

# Starting point
x_k = np.array([[1], [1], [1], [1]])

# Matrices part of straight line formula
A = np.array([[3, 2, 0, -1], [-1, 3, 0 , 2], [0, -4, -2, 7]])
b = np.array([[3], [1], [-4]])


x_store = []
while True:
    # Preparing delta f
    delta_f = x_k - alpha * (A.T @ A @ x_k - A.T @ b + gamma * x_k)
    
    if np.linalg.norm(delta_f) < 0.001:
        break

    # Keep a memory of all the x_k values
    x_store.append(x_k)

    # Minimiser function
    x_k_next = x_k - alpha * delta_f

    # Set x_k for next iteration
    x_k = x_k_next


for i in range(0, 5):
    print("k = " + str(i) + ",      " + "x" + " = " + str(np.round(x_store[i].T, decimals=4)))

for i in range(0, 5):
    print("k = " + str(len(x_store) - 5 + i) + ",   " + "x" " = " + str(np.round(x_store[-5+i].T, decimals=4)))

print("PROGRAM END")
