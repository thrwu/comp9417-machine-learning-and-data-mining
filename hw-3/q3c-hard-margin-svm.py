import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

# Created 26th July, 2026
# Given a set of data points (x1,x2) and labels (y), first transform the points then utilise the existing SVC() package to perform the classification algorithm on the transformed data points with a hard-margin i.e. no tolerance for misclassification. Compute the Lagrange Multiplier corresponding to each data set (i.e. influence on the margin that identifies the point as either a supporting vector or not a supporting vector). Finally, compute the training error (does the separating hyperplane correctly classify/separate all data points?).

def main():
    # Get data points (x1,x2) and labels (y)
    X = np.array([[1, 0], [0, 1], [0, -1], [-1, 0], [0, 2], [0, -2], [-2, 0]])
    Y = np.array([-1, -1, -1, 1, 1, 1, 1])

    # Transform the data sets
    X_transformed = np.c_[2*X[:, 1]**2-4*X[:, 0]+1, X[:, 0]**2-2*X[:,1]-3] # np.c_[] concatenates arrays horizontally along their second axis (columns), which prevents the need to write X_transformed=X_transformed.T with a normal np.array([]) method.

    # Hyperplane equation: w0*x1+w1*x2+b=0 => x2=-w0/w1 *x1-b/w1 = -(w0*x1+b)/w1
    w, b, alphas, support_vectors, support_indices, training_error = get_hyperplane_equation(X_transformed, Y)
    print(f"Separating Hyperplane Equation (rounded to 3 decimal points): {w[0]:.3f} * x1 + {w[1]:.3f} * x2 + {b:.3f} = 0\n")
    print(f"Alpha Values: {alphas}\n")
    print(f"Support Vectors: \n{support_vectors}\n")
    print(f"Support Indices: \n{support_indices}\n")
    print(f"Training Error: {training_error}")
    x1_hyperplane = np.linspace(-3, 10, 100)
    x2_hyperplane = -(w[0] * x1_hyperplane + b)/w[1]

    # Plot
    #plt.scatter(X[:, 0], X[:, 1], s=20, color=colours, marker='s') # Plot Original (x1,x2) data points
    plt.scatter(X_transformed[:, 0], X_transformed[:, 1], s=20, color="blue", marker='o')
    plt.scatter(support_vectors[:, 0], support_vectors[:, 1], s=20, color="red", marker='o', label="Support Vectors")
    plt.plot(x1_hyperplane, x2_hyperplane, label="Separating Hyperplane")

    plt.title(f"Best Separating Hyperplane: {w[0]:.3f} * x1 + {w[1]:.3f} * x2 + {b:.3f} = 0")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.ylim(-8, 3)
    plt.legend(loc="upper left")
    plt.show()

def get_hyperplane_equation(X, Y):

    # Train Support Vector Machine with a linear transformation/kernel
    model = SVC(kernel='linear', C=1e20)
    model.fit(X, Y)

    # Extract weights (w) and bias (b)
    w = model.coef_[0]
    b = model.intercept_[0]

    # Lagrange Multipliers (alpha) -- one for each data set that represents it's 'importance' or how heavily it influences the margin or separating hyperplace. Note that model.dual_coef_ only gives back non-zero alphas for support vectors
    alphas = np.abs(model.dual_coef_[0])

    # Support Vectors that heavily influence the margin or separating hyperplace
    support_vectors = model.support_vectors_
    support_indices = model.support_

    # Get array of all alphas (including non-zero alphas for non-support vectors)
    all_alphas = np.zeros(len(X))
    all_alphas[support_indices] = np.round(alphas, 3) # model.support_ gives the index of the support_vectors in the input data set 'X'

    training_score = model.score(X, Y)
    training_error = 1.0 - training_score

    return w, b, all_alphas, support_vectors, support_indices, training_error
    

if __name__ == "__main__":
    main()
