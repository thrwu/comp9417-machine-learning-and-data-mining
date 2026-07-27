import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

# Created 26th July, 2026
# Similar to q3c. However, instead of a linear kernel, we perform the classification algorithm SVM on a polynomial kernel k(x,z) = (2+X.T*z)^2. This prevents the need to transform the data set.

def main():
    # Get data points (x1,x2) and labels (y)
    X = np.array([[1, 0], [0, 1], [0, -1], [-1, 0], [0, 2], [0, -2], [-2, 0]])
    Y = np.array([-1, -1, -1, 1, 1, 1, 1])

    # Hyperplane equation: w0*x1+w1*x2+b=0 => x2=-w0/w1 *x1-b/w1 = -(w0*x1+b)/w1
    alphas, support_vectors, support_indices, training_error = get_hyperplane_equation(X, Y)
    print(f"Alpha Values: {alphas}\n")
    print(f"Support Vectors: \n{support_vectors}\n")
    print(f"Support Indices: \n{support_indices}\n")
    print(f"Training Error: {training_error}")

def get_hyperplane_equation(X, Y):

    # Train Support Vector Machine with a polynomial transformation/kernel
    # Kernel: (2+X.T*z)^2
    # Kernel: (coef0 + X.T*z)^degree
    model = SVC(
        kernel='poly', 
        degree=2,
        coef0=2.0,
        C=1e20
    )
    model.fit(X, Y)

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

    return all_alphas, support_vectors, support_indices, training_error
    

if __name__ == "__main__":
    main()
