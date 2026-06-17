# 17th June, 2026
# Created by Tian Hong Raymond Wu

# The purpose of this program is to show the minimiser function that apparently gives us the minimum domain of the function (i.e. minimum x).


import sys

# Initial Input
x_k = 1

for i in range(0, int(sys.argv[1])):
    # Preparing delta f and step size (alpha)
    delta = 3 * x_k**2 * (x_k**3 + 1)**(-1/2)
    alpha = 0.1

    # Minimiser function
    x_k_next = x_k - alpha * delta

    # Set x_k for next iteration
    x_k = x_k_next
    print(delta)

# The iterated x value never reaches the true minimum because, if we plot the delta function, we see that it has a global minimum at 0 when x = 0, so the printed minimum x-value remains stuck at 0. In addition, the function y (whose derivative is delta) is discontinuous.