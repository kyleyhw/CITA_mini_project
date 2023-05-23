import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import emcee

from question_3 import LogProb, AutocorrelationTime
from data_loader import Data

log_prob = LogProb()

filepath = 'simple-cw-master/'
filename = 'cw_' + 'example' + '.csv'

data = Data(filepath + filename)

ndim = 2
nwalkers = 4
steps = 10

# epsilon_p0 = np.random.uniform(size=nwalkers, low=1e-7, high=1e-5)
# df_dt_p0 = np.random.uniform(size=nwalkers, low=1e-5, high=1e-3)
# p0 = np.zeros(shape=(nwalkers, ndim))
# p0[:, 0] = epsilon_p0
# p0[:, 1] = df_dt_p0


p0 = np.random.uniform(size=(nwalkers, ndim), low=(1e-7, 1e-5), high=(1e-5, 1e-3))
print(p0)



sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, args=[data.times, data.signal])

sampler.run_mcmc(p0, steps, progress=True)

flat_samples = sampler.get_chain(flat=True)
samples = sampler.get_chain()

epsilon_points = flat_samples[:, 0]
df_dt_points = flat_samples[:, 1]

epsilon_walk = samples[:, :, 0]
df_dt_walk = samples[:, :, 1]

plt.figure()

plt.plot(epsilon_walk, df_dt_walk)
sns.kdeplot(x=epsilon_points, y=df_dt_points, levels=[0.5, 0.9])
plt.hist2d(x=epsilon_points, y=df_dt_points, bins=nwalkers)

plt.xlabel(r'\epsilon')
plt.ylabel(r'\dot{f}')
plt.show()

chain = samples[0]
print(chain.shape)
autocorrelation_time = AutocorrelationTime(chain)
print(autocorrelation_time.tau)