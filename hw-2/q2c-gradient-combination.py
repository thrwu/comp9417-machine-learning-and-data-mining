from sklearn.tree import DecisionTreeRegressor
import numpy as np 
import matplotlib.pyplot as plt

# true function
def f(x):
    t1 = np.sqrt(x * (1-x))
    t2 = (2.1 * np.pi) / (x + 0.05)
    t3 = np.sin(t2)
    return t1*t3

def f_sampler(f, n=100, sigma=0.05):    
    # sample points from function f with Gaussian noise (0,sigma**2)
    xvals = np.random.uniform(low=0, high=1, size=n)
    yvals = f(xvals) + sigma * np.random.normal(0,1,size=n)

    return xvals, yvals

def perform_gradient_combination(x, y, T, adaptive):

    # Range of x-values to perform fitted model on
    xx = np.linspace(0, 1, len(y))
    xx = xx.reshape(-1, 1)

    # Initial
    f_prev = np.zeros(len(y))
    
    # Number of desired base learners
    for t in range(0, T):

        # (GC1) Residual 'r'
        # (GC2) Pseudo Data Set 'D'
        D = np.zeros(len(y))
        for i in range(0, len(y)):
            r = y[i] - f_prev[i]
            D[i] = r

        # (GC3) Fit a model to this residual using our base class F
        # Prepare machine learning base class F (decision tree that splits populations to two sub-groups for best fit)
        h = DecisionTreeRegressor(max_depth=1)
        # Perform the base class on residuals to get a model
        h = h.fit(x, D)
        # Perform the tested model on the x values
        h = h.predict(xx)

        alpha = 0
        for i in range(0, len(y)):
            alpha += h[i]
        alpha /= len(h)

        if (adaptive):
            f_curr = f_prev + alpha * h
        else:
            f_curr = f_prev + 0.1 * h
        f_prev = f_curr
                        

    return f_curr



np.random.seed(123)
X, y = f_sampler(f, 160, sigma=0.2)
X = X.reshape(-1,1)

# ADAPTIVE
fig1, axes1 = plt.subplots(nrows=5, ncols=2, layout='constrained')
axes1 = axes1.flatten() # to enumerate through

for i, ax in enumerate(axes1):
    adaptive = True
    base_learner = (i + 2) * 5
    fitted_model = perform_gradient_combination(X, y, base_learner, adaptive)
    xx = np.linspace(0,1,1000)
    ax.plot(xx, f(xx), alpha=0.5, color='red', label='truth')
    ax.scatter(X,y, marker='x', color='blue', label='observed')
    ax.plot(np.linspace(0, 1, len(fitted_model)), fitted_model, color='green', label='dt')
    ax.set_title(f"Plot F: {i*5} (Fixed)")


fig2, axes2 = plt.subplots(nrows=5, ncols=2, layout='constrained')
axes2 = axes2.flatten() # to enumerate through

for i, ax in enumerate(axes2):
    adaptive = False
    base_learner = (i + 2) * 5
    fitted_model = perform_gradient_combination(X, y, base_learner, adaptive)
    xx = np.linspace(0,1,1000)
    ax.plot(xx, f(xx), alpha=0.5, color='red', label='truth')
    ax.scatter(X,y, marker='x', color='blue', label='observed')
    ax.plot(np.linspace(0, 1, len(fitted_model)), fitted_model, color='green', label='dt')
    ax.set_title(f"Plot F: {i*5} (Adaptive)")

       
plt.tight_layout()
plt.show()