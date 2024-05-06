# -*- coding: utf-8 -*-
"""
Created on Sat May  4 17:43:35 2024
School Activity
@author: Alfred Dagohoy
Pymc3 (regression)
"""

import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt

# Generate some synthetic data
np.random.seed(42)
n = 100
true_slope = 2.0
true_intercept = 1.0
x = np.linspace(0, 10, n)
y = true_slope * x + true_intercept + np.random.normal(scale=0.5, size=n)

# Define the Bayesian linear regression model
with pm.Model() as linear_model:
    # Priors
    slope = pm.Normal('slope', mu=0, sd=10)
    intercept = pm.Normal('intercept', mu=0, sd=10)
    sigma = pm.HalfNormal('sigma', sd=1)

    # Likelihood
    likelihood = pm.Normal('y', mu=slope * x + intercept, sd=sigma, observed=y)

# Perform MCMC sampling
with linear_model:
    trace = pm.sample(2000, tune=1000, cores=1)  # Adjust parameters based on your computational resources

# Plot the posterior distributions
pm.traceplot(trace)
plt.show()

# Get the summary statistics of the posterior distributions
summary = pm.summary(trace)
print(summary)

# Plot the regression line with 95% credible interval
plt.scatter(x, y, label='Data')
pm.plot_posterior_predictive_glm(trace, samples=100, eval=np.linspace(0, 10, 100), label='Posterior Predictive')
plt.plot(np.linspace(0, 10, 100), true_slope * np.linspace(0, 10, 100) + true_intercept, 'r--', label='True Regression Line')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Bayesian Linear Regression')
plt.legend()
plt.show()