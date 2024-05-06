# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 10:18:46 2024
School Activity
@author: Alfred Dagohoy
Pymc3 (forecasting/Regression)
"""

import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt

# Generate some synthetic data
np.random.seed(42)
X = np.linspace(0, 10, 100)
true_slope = 2.0
true_intercept = 1.0
true_noise = 1.0
y = true_slope * X + true_intercept + np.random.normal(scale=true_noise, size=X.shape)

# Define the Bayesian linear regression model
with pm.Model() as model:
    # Priors for the parameters
    slope = pm.Normal('slope', mu=0, sigma=10)
    intercept = pm.Normal('intercept', mu=0, sigma=10)
    noise = pm.HalfNormal('noise', sigma=10)
    
    # Expected value of outcome
    mu = slope * X + intercept
    
    # Likelihood (sampling distribution) of observations
    likelihood = pm.Normal('y', mu=mu, sigma=noise, observed=y)
    
    # Sampler
    step = pm.NUTS()

# Perform Bayesian inference
with model:
    trace = pm.sample(2000, tune=1000, cores=1, step=step)

# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(X, y, label='Data')
pm.plot_posterior_predictive_glm(trace, samples=100, label='Posterior predictive regression lines')
plt.plot(X, true_slope * X + true_intercept, 'k--', label='True regression line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Bayesian Linear Regression')
plt.legend()
plt.show()