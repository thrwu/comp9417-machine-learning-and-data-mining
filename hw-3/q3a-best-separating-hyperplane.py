import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC

# Created 26th July, 2026
# Given a set of data points (x1,x2) and labels (y), first transform the points then plot the equation of the best separating hyper-plane (utilise SVC package from sklearn.svm).

def main():
    # Get data points (x1,x2) and labels (y)
    X = np.array([[1, 0], [0, 1], [0, -1], [-1, 0], [0, 2], [0, -2], [-2, 0]])
    Y = np.array([-1, -1, -1, 1, 1, 1, 1])

    # Transform the data sets
    X_transformed = np.c_[2*X[:, 1]**2-4*X[:, 0]+1, X[:, 0]**2-2*X[:,1]-3] # np.c_[] concatenates arrays horizontally along their second axis (columns), which prevents the need to write X_transformed=X_transformed.T with a normal np.array([]) method.

    # Hyperplane equation: w0*x1+w1*x2+b=0 => x2=-w0/w1 *x1-b/w1 = -(w0*x1+b)/w1
    w, b = get_hyperplane_equation(X_transformed, Y)
    print(f"Separating Hyperplane Equation (rounded to 3 decimal points): {w[0]:.3f} * x1 + {w[1]:.3f} * x2 + {b:.3f} = 0")
    x1_hyperplane = np.linspace(-3, 10, 100)
    x2_hyperplane = -(w[0] * x1_hyperplane + b)/w[1]

    # Plot
    colours = ['orange' if y == 1 else 'blue' for y in Y]
    #plt.scatter(X[:, 0], X[:, 1], s=20, color=colours, marker='s') # Plot Original (x1,x2) data points
    plt.scatter(X_transformed[:, 0], X_transformed[:, 1], s=20, color=colours, marker='o', label="Transformed Data Points")
    plt.plot(x1_hyperplane, x2_hyperplane, label="Separating Hyperplane")

    plt.title(f"Best Separating Hyperplane: {w[0]:.3f} * x1 + {w[1]:.3f} * x2 + {b:.3f} = 0")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.ylim(-8, 3)
    plt.legend(loc="upper left")
    plt.show()

def get_hyperplane_equation(X, Y):

    # Train Support Vector Machine with a linear transformation/kernel
    model = SVC(kernel='linear')
    model.fit(X, Y)

    # Extract weights (w) and bias (b)
    w = model.coef_[0]
    b = model.intercept_[0]

    return w, b
    

if __name__ == "__main__":
    main()
