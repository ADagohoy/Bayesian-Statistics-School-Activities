# -*- coding: utf-8 -*-
"""
Created on Mon May  6 10:57:57 2024
School activity
@author: Alfred Dagohoy
Uniform Dist and Bayesian Interference 
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform

# Generate data
np.random.seed(42)
data = np.random.uniform(0, 1, 1000)  # Generate 1000 random samples from a uniform distribution [0, 1]

# Plot the histogram of the data
plt.hist(data, bins=20, density=True, alpha=0.7, color='skyblue', edgecolor='black')
plt.title('Histogram of Uniform Distribution')
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.grid(True)
plt.show()

# Bayesian Inference with Uniform Distribution
prior_uniform = uniform(loc=0, scale=1)  # Prior belief: Uniform distribution over [0, 1]

# Likelihood function
def likelihood(data, theta):
    return np.prod(uniform.pdf(data, loc=theta, scale=1))

# Posterior calculation using Bayes' theorem
def posterior(data, prior, likelihood):
    theta_values = np.linspace(0, 1, 1000)  # Possible values of the parameter theta
    prior_prob = prior.pdf(theta_values)  # Prior probabilities
    likelihood_prob = np.array([likelihood(data, theta) for theta in theta_values])  # Likelihood probabilities
    unnormalized_posterior = prior_prob * likelihood_prob  # Unnormalized posterior
    posterior_prob = unnormalized_posterior / np.sum(unnormalized_posterior)  # Normalizing the posterior
    return theta_values, posterior_prob

# Calculate posterior
theta_values, posterior_prob = posterior(data, prior_uniform, likelihood)

# Plot the posterior distribution
plt.plot(theta_values, posterior_prob, color='green')
plt.title('Posterior Distribution')
plt.xlabel('Parameter (Theta)')
plt.ylabel('Probability Density')
plt.grid(True)
plt.show()
