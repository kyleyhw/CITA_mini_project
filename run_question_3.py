import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import emcee

from question_3 import LogProb
from data_loader import Data

log_prob = LogProb()

filepath = 'simple-cw-master/'
filename = 'cw_' + 'example' + '.csv'

data = Data(filepath + filename)

ndim = 2
nwalkers = 32
p0 = np.random.uniform(size=(nwalkers, ndim), low=(1e-7, 1e-5), high=(1e-5, 1e-3))

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, args=[data.times, data.signal])

sampler.run_mcmc(p0, 10)

samples = sampler.get_chain()
print(samples.shape)
print(samples[0])

epsilon_walk = samples[:, :, 0]
df_dt_walk = samples[:, :, 1]

plt.figure()
plt.plot(epsilon_walk, df_dt_walk)
plt.show()