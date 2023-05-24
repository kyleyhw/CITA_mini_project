import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import emcee

from question_3 import UniformPrior, Likelihood, LogProb, AutocorrelationTime
from data_loader import Data

filepath = 'simple-cw-master/'
filename = 'cw_' + 'example' + '.csv'

data = Data(filepath + filename)

ndim = 2
nwalkers = 4
steps = 100000

prior_lower_bounds = (1e-7, 1e-5)
prior_upper_bounds = (1e-5, 1e-3)

prior = UniformPrior(lower_bounds=prior_lower_bounds, upper_bounds=prior_upper_bounds)
likelihood = Likelihood()
log_prob = LogProb(likelihood=likelihood, prior=prior)


p0 = np.random.uniform(size=(nwalkers, ndim), low=prior_lower_bounds, high=prior_upper_bounds)


sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, args=[data.times, data.signal])

sampler.run_mcmc(p0, steps, progress=True)

flat_samples = sampler.get_chain(flat=True)
samples = sampler.get_chain()

epsilon_points = flat_samples[:, 0]
df_dt_points = flat_samples[:, 1]

epsilon_walk = samples[:, :, 0]
df_dt_walk = samples[:, :, 1]

plt.figure()

plt.plot(epsilon_walk, df_dt_walk, alpha=0.05)
sns.kdeplot(x=epsilon_points, y=df_dt_points, levels=[0.9])
plt.hist2d(x=epsilon_points, y=df_dt_points, bins=32, cmap='gray')
plt.colorbar()

plt.scatter([1e-6], [1e-4], label='true value', color='white', marker='o')
plt.legend()

plt.xlabel(r'\epsilon')
plt.ylabel(r'\dot{f}')

plt.show()

samples = np.swapaxes(samples, 0, 1)
for i, chain in enumerate(samples):
    autocorrelation_time = AutocorrelationTime(chain)
    print('autocorrelation time for chain %i:' %i, autocorrelation_time.tau)


fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)

for chain in samples:
    ax1.plot(chain.T[0], alpha=0.05)
    ax2.plot(chain.T[1], alpha=0.05)

ax1.set_title('epsilon')
ax2.set_title('df_dt')
plt.show()