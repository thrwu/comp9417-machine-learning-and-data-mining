import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets

X, y = datasets.make_circles(n_samples=200, factor=0.4, noise=0.04, random_state=13)
colors = np.array(['orange', 'blue'])

np.random.seed(123)
random_labeling = np.random.choice([0,1], size=X.shape[0], )
plt.scatter(X[:, 0], X[:, 1], s=20, color=colors[random_labeling])
plt.title("Randomly Labelled Points")
plt.savefig("Randomly_Labeled.png")
plt.show()