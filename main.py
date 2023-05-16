import numpy as np
import matplotlib.pyplot as plt

from question_2 import Template, Likelihood
from data_loader import Data


filepath = 'simple-cw-master/'
filename = 'cw_' + 'example' + '.csv'

Likelihood = Likelihood()

data = Data(filepath + filename)

iterations = 10000

df_dt_guesses = np.random.uniform(low=1e-5, high=1e-3, size=iterations)
epsilon_guesses = np.random.uniform(low=1e-7, high=1e-5, size=iterations)

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

plt.scatter(df_dt_guesses, epsilon_guesses, c=likelihoods)
plt.xlabel('df_dt')
plt.ylabel('epsilon')
plt.show()

print(np.amax(likelihoods))

df_dt_guess = df_dt_guesses[(np.where(likelihoods == np.amax(likelihoods)))[0]]
epsilon_guess = epsilon_guesses[(np.where(likelihoods == np.amax(likelihoods)))[0]]

print(epsilon_guess, df_dt_guess)

template = Template(df_dt=df_dt_guess, epsilon=epsilon_guess, f_rot0=data.f_rot0)

model = template(data.times)
likelihood = Likelihood(signal=data.signal, model=model)

plt.figure()
plt.plot(data.signal[:100], label='signal')
plt.plot(model[:100], label='model')
plt.legend()
plt.show()

print(np.log(likelihood))