import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import emcee

from question_3 import UniformPrior, Likelihood, LogProb, AutocorrelationTime, Template
from data_loader import Data

filepath = 'simple-cw-master/'
filename = 'cw_' + 'example' + '.csv'

data = Data(filepath + filename)

ndim = 2
nwalkers = 4
steps = 10000
# (epsilon, fdot): true value for epsilon is 1e-6, true value for fdot is 1e-4
# prior_lower_bounds = (0.9e-6, 0.9e-4)
# prior_upper_bounds = (1.1e-6, 1.1e-4)

prior_lower_bounds = (0, 0)
prior_upper_bounds = (1e-4, 1e-3)

prior = UniformPrior(lower_bounds=prior_lower_bounds, upper_bounds=prior_upper_bounds) # initialize the prior, which is uniform (1 in allowed region, 0 otherwise)
likelihood = Likelihood() # initialize likelihood function
log_prob = LogProb(likelihood=likelihood, prior=prior) # pass likelihood function and prior into log_probability, where log_prob = log_likelihood + log_prior


p0 = np.random.uniform(size=(nwalkers, ndim), low=prior_lower_bounds, high=prior_upper_bounds) # random initial guesses, using the same bounds as the prior

dtype=[('log_like', float)]
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, args=[data.times, data.signal, data.f_rot0], blobs_dtype=dtype) # initialize MCMC sampler

sampler.run_mcmc(p0, steps, progress=True) # run MCMC sampler

flat_samples = sampler.get_chain(flat=True) # get flattened samples for plotting histograms
samples = sampler.get_chain() # get samples for plotting individual chains

epsilon_points = flat_samples[:, 0]
df_dt_points = flat_samples[:, 1]

epsilon_walk = samples[:, :, 0]
df_dt_walk = samples[:, :, 1]

plt.figure() # plot histogram, contour plot, and chains all overlaid, with true value incidated

plt.plot(epsilon_walk, df_dt_walk, alpha=0.1)
sns.kdeplot(x=epsilon_points, y=df_dt_points, levels=[0.9])
plt.hist2d(x=epsilon_points, y=df_dt_points, bins=64, cmap='Greys')
plt.colorbar()

plt.scatter([1e-6], [1e-4], label='true value', color='red', marker='x')
plt.legend()

plt.xlabel(r'\epsilon')
plt.ylabel(r'\dot{f}')

plt.show()

samples = np.swapaxes(samples, 0, 1) # gives chains of parameter vectors, to incorporate Speagle definition of autocorrelation time

# for i, chain in enumerate(samples):
#     autocorrelation_time = AutocorrelationTime(chain)
#     print('autocorrelation time for chain %i:' %i, autocorrelation_time.tau)


fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1) # trace plots for each parameter

for chain in samples:
    ax1.plot(chain.T[0], alpha=0.1)
    ax2.plot(chain.T[1], alpha=0.1)

    print(chain.T[0, -1], chain.T[1, -1])

ax1.set_title('epsilon')
ax2.set_title('df_dt')
plt.show()

# blobs = sampler.get_blobs()
# print(blobs['log_like'])