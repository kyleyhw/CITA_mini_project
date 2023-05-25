import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from question_2 import Template, Likelihood
from data_loader import Data

filepath = 'simple-cw-master/'
filename = 'cw_' + 'example' + '.csv'

Likelihood = Likelihood()

data = Data(filepath + filename)

iterations = 5000

df_dt_guesses = np.random.uniform(low=1e-7, high=1e-3, size=iterations)
epsilon_guesses = np.random.uniform(low=1e-8, high=1e-4, size=iterations)

# df_dt_guesses = np.random.normal(loc=1e-4, scale=1e-5, size=iterations)
# epsilon_guesses = np.random.uniform(low=1e-8, high=1e-4, size=iterations)

likelihoods = np.zeros(iterations)

for i in range(iterations):
    template = Template(df_dt=df_dt_guesses[i], epsilon=epsilon_guesses[i], f_rot0=data.f_rot0)

    model = template(data.times)
    likelihood = Likelihood(signal=data.signal, model=model)

    likelihoods[i] = likelihood

    # plt.figure()
    # plt.plot(data.signal[:100], label='signal')
    # plt.plot(model[:100], label='model')
    # plt.legend()
    # plt.show()

# plt.scatter(df_dt_guesses, epsilon_guesses, c=likelihoods)
# plt.xlabel('df_dt')
# plt.ylabel('epsilon')
# plt.show()
#
# print(np.amax(likelihoods))



# df_dt_guess = df_dt_guesses[(np.where(likelihoods == np.amax(likelihoods)))[0]]
# epsilon_guess = epsilon_guesses[(np.where(likelihoods == np.amax(likelihoods)))[0]]
#
# print(epsilon_guess, df_dt_guess)
#
# template = Template(df_dt=df_dt_guess, epsilon=epsilon_guess, f_rot0=data.f_rot0)
#
# model = template(data.times)
# likelihood = Likelihood(signal=data.signal, model=model)
#
# plt.figure()
# plt.plot(data.signal[:100], label='signal')
# plt.plot(model[:100], label='model')
# plt.legend()
# plt.show()
#
# print(np.log(likelihood))



plt.figure()
plt.scatter(df_dt_guesses, epsilon_guesses, c=likelihoods, alpha=0.1)
sns.kdeplot(x=df_dt_guesses, y=epsilon_guesses, weights=likelihoods / np.sum(likelihoods), levels=[0.1])
plt.xlabel('df_dt')
plt.ylabel('epsilon')
plt.show()

def effective_samples(likelihoods):
    return np.sum(likelihoods)**2 / np.sum(likelihoods**2)

print(effective_samples(likelihoods))