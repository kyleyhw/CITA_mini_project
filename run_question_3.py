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
nwalkers = 32
steps = 10

prior_lower_bounds = (1e-7, 1e-5)
prior_upper_bounds = (1e-5, 1e-3)

prior = UniformPrior(lower_bounds=prior_lower_bounds, upper_bounds=prior_upper_bounds)
likelihood = Likelihood()
log_prob = LogProb(prior=prior, likelihood=likelihood)


p0 = np.random.uniform(size=(nwalkers, ndim), low=prior_lower_bounds, high=prior_upper_bounds)
print(p0)



sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, args=[data.times, data.signal])

sampler.run_mcmc(p0, steps, progress=True)

flat_samples = sampler.get_chain(flat=True)
samples = sampler.get_chain()

epsilon_points = flat_samples[:, 0]
df_dt_points = flat_samples[:, 1]

epsilon_walk = samples[:, :, 0]
df_dt_walk = samples[:, :, 1]

epsilon_logspace = np.logspace(prior_lower_bounds[0], prior_upper_bounds[0], num=32)
df_dt_logspace = np.logspace(prior_lower_bounds[1], prior_upper_bounds[1], num=32)

plt.figure()

plt.plot(epsilon_walk, df_dt_walk)
sns.kdeplot(x=epsilon_points, y=df_dt_points, levels=[0.5, 0.9])
plt.hist2d(x=epsilon_points, y=df_dt_points, bins=np.array([epsilon_logspace, df_dt_logspace]))

plt.xlabel(r'\epsilon')
plt.ylabel(r'\dot{f}')
plt.xscale('log')
plt.yscale('log')
plt.show()

chain = samples[0]
print(chain.shape)
autocorrelation_time = AutocorrelationTime(chain)
print('autocorrelation time:', autocorrelation_time.tau)