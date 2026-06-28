# comp9417-machine-learning-and-data-mining

Tian Hong Raymond Wu

Created: 16th June, 2026

This is a repository containing all the code and documentation for UNSW COMP9417 (Machine Learning and Data Mining). The primary usage would be for referencing and storage.

## Homework 1

### Gradient Based Optimisation
Given a bunch of data, we may want to find the line of best fit that follows a trend. The human eye can find this line somewhat easily with some margin of error, but how do we allow the computer to find this line numerically and accurately?

Assuming that the trend of the data is linear, the line of best fit would have the following equation: 

$$y=mx+b$$

Given $(x,y)$ data points, we must find the parameters $m$ and $b$ that minimises the error (i.e. MSE between data points and the line of best fit).

The gradient based optimisation method involves finding the derivative of a loss function (i.e. an equation that fits to the data and also has a term that prevents overfitting) such as:

$$L(\beta)=\frac{1}{2p}||y-\beta||^2_2+\lambda||W\beta||^2_2$$

After finding its derivative, we then rearrange the equation in respect to the parameters (i.e. $\beta$). Finally, we equate the derivative to zero (minimum error) and solve for $\beta$.

### Coordinate Based Scheme

Similar optimisation method to gradient descent. However, we update the parameters one at a time instead of updating the entire $p$ parameters at every iteration.