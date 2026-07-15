import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets

def main():
    X, y = datasets.make_circles(n_samples=200, factor=0.4, noise=0.04, random_state=13)
    colors = np.array(['orange', 'blue'])


    labeling = clustering_algorithm(X)

    # 's' represents the size, X[:, 0] represents the x-points, and X[:, 1] represents the y-points
    plt.scatter(X[:, 0], X[:, 1], s=20, color=colors[labeling.astype(int)])
    plt.title("Randomly Labelled Points")
    plt.savefig("Randomly_Labeled.png")
    plt.show()



def clustering_algorithm(X):
    T = 10          # Number of iterations for the algorithm to learn/update the clusters
    n = len(X)      # Number of elements in the (unlabeled) data X

    # Initialise two cluster centers
    K = 2 # Number of cluster centers
    mu_0 = np.array([0, 0, 0])
    mu_1 = np.array([1, 1, 0])

    # Convert inputs from 2D to 3D
    for i in range(n):
        if (i == 0):
            phi_3d = np.array([X[i,0]**2, X[i, 1]**2, np.sqrt(2)*X[i,0]*X[i,1]])
            continue
        phi_3d = np.vstack((phi_3d, np.array([(X[i,0]**2, X[i, 1]**2, np.sqrt(2)*X[i,0]*X[i,1])])))

    for t in range(0, T):
        k_i = np.array([]) # Get an array of the cluster centre (mu) that is closest to each data set
        for i in range(0, n):
            diff_0 = phi_3d[i] - mu_0
            diff_1 = phi_3d[i] - mu_1

            # Find out which cluster center the data at index 'i' is closest to, and represent this info as '0' (mu_0) or '1' (mu_1)
            k_i = np.append(k_i, np.argmin([np.dot(diff_0, diff_0), np.dot(diff_1, diff_1)]))

        for k in range(1, K):
            # Prevent division by zeroes
            if (np.count_nonzero(k_i == 0)):
                mu_0 = 1/(np.count_nonzero(k_i == 0)) * np.sum(phi_3d[k_i == 0], axis=0) # axis=0 adds the arrays element-wise
            if (np.count_nonzero(k_i == 1)):
                mu_1 = 1/(np.count_nonzero(k_i == 1)) * np.sum(phi_3d[k_i == 1], axis=0)
    
    print(f"mu_0: {mu_0} | mu_1: {mu_1}")
    return k_i

if __name__ == "__main__":
    main()
    