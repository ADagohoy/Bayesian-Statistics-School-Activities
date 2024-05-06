# -*- coding: utf-8 -*-
"""
Created on Mon May  6 09:41:23 2024
School Activity 5
@author: Alfred Dagohoy
Sample script B
"""

import scipy.stats as sts
import numpy as np
import matplotlib.pyplot as plt

def likelihood_func(datum, mu):
    likelihood_out = sts.norm.pdf(datum, mu, scale = 0.1)
    return likelihood_out/likelihood_out.sum()

mu = np.linspace(1.65, 1.8, num = 50)
test = np.linspace(0, 2)
uniform_dist = sts.uniform.pdf(mu)
likelihood_out = likelihood_func(1.7, mu)

plt.plot (mu, likelihood_out)
plt.title("Likelihood of $\mu$ given observation 1.7m")
plt.ylabel(" Probability Density/Likelihood")
plt.xlabel("Value of $mus$")
plt.show()