import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets

# Created 26th July, 2026
# In q2d, we found that directly applying feature transformation to the data can be computationally intractable (i.e. for larger inputs, the task may be practically impossible to perform). In other words, we have to iterate through every single data point and transform it, which can take an extremely long time for large inputs. A new method is implemented where we utilise h(x,x') = <phi(x),phi(x')>.

def main():
    X, y = datasets.make_circles(n_samples=200, factor=0.4, noise=0.04, random_state=13)
    colors = np.array(['orange', 'blue'])

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    # Perform cluster algorithm with linear function (i.e. attempt to 'cut' the data into two halves using a linear straight line)
    labeling = clustering_algorithm(X, np.load('init_clusters.npy'), False)
    axs[0].scatter(X[:, 0], X[:, 1], s=20, color=colors[labeling.astype(int)])
    axs[0].set_title("Linear Function")
    axs[0].set_xlabel('x1')
    axs[0].set_ylabel('x2')

    # Perform cluster algorithm with polynomial function (i.e. [1 + h(x, x')]^2)
    labeling = clustering_algorithm(X, np.load('init_clusters.npy'), True)
    axs[1].scatter(X[:, 0], X[:, 1], s=20, color=colors[labeling.astype(int)])
    axs[1].set_title("Polynomial Function")
    axs[1].set_xlabel('x1')
    axs[1].set_ylabel('x2')

    plt.show()

def clustering_algorithm(X, cluster_init, is_poly):
    T = 10          # Number of iterations for the algorithm to learn/update the clusters
    n = len(X)      # Number of elements in the (unlabeled) data X
    k_prev = cluster_init # Initial set of K clusters i.e. C_1^0, C_2^0, ..., C_K^0
    K = 2

    for t in range(T):
        k_i = np.array([]) # Get an array of the cluster centre that is closest to each data set

        for i in range(n):
            argmin_set = np.array([])
            
            # Check if the user wants to utilise a polynomial function
            if (is_poly):
                h_ii = (1 + np.inner(X[i], X[i]))**2
            else:
                h_ii = 1 + np.inner(X[i], X[i])

            for k in range(K):
                X_j = X[k_prev == k]

                h_ij = 0
                for j in range(len(X_j)):
                    if (is_poly):
                        h_ij += (1 + np.inner(X[i], X_j[j]))**2
                    else:
                        h_ij += 1 + np.inner(X[i], X_j[j])

                h_jl = 0
                for j in range(len(X_j)):
                    for l in range(len(X_j)):
                        if (is_poly):
                            h_jl += (1 + np.inner(X_j[j], X_j[l]))**2
                        else:
                            h_jl += 1 + np.inner(X_j[j], X_j[l])

                C_k_cardinal = len(X_j)

                if (C_k_cardinal):
                    argmin_set = np.append(argmin_set, h_ii - (2/C_k_cardinal)*h_ij + (1/(C_k_cardinal**2))*h_jl)

            # Find out which cluster center the data at index 'i' is closest to, and represent this info as '0' or '1'
            k_i = np.append(k_i, np.argmin(argmin_set))

        k_prev = k_i
        
    return k_i


if __name__ == "__main__":
    main()
    