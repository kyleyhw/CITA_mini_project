import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import emcee

from question_2 import Template, Likelihood
from data_loader import Data

filepath = 'simple-cw-master/'
filename = 'cw_' + 'example' + '.csv'

Likelihood = Likelihood()

data = Data(filepath + filename)

ndim = 2
nwalkers = 32
p0 = np.random.uniform(nwalkers, ndim)

sampler = emcee.EnsembleSampler(nwalkers, ndim, Likelihood)

state = sampler.run_mcmc(p0, 100)
sampler.reset()

sampler.run_mcmc(state, 10000)

samples = sampler.get_chain(flat=True)
plt.hist(samples[:, 0], 100, color="k", histtype="step")
plt.xlabel(r"$\theta_1$")
plt.ylabel(r"$p(\theta_1)$")
plt.gca().set_yticks([])