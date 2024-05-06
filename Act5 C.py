# -*- coding: utf-8 -*-
"""
Created on Mon May  6 10:21:47 2024
School Activity 5
@author: Alfred Dagohoy
Sample script C
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sts
def likelihood_func(datum, mu):
    likelihood_out = sts.norm.pdf(datum, mu, scale = 0.1)
    return likelihood_out/likelihood_out.sum()

mu = np.linspace(1.65, 1.8, num = 50)
test = np.linspace(0, 2)
uniform_dist = sts.uniform.pdf(mu)
likelihood_out = likelihood_func(1.7, mu)

unnormalized_posterior = likelihood_out * uniform_dist
plt.plot (mu, unnormalized_posterior)
plt.title("$mu$ in meters")
plt.ylabel(" Unnormalized Posterior")
plt.xlabel("Unnormalized Posterior Distribution")
plt.show()