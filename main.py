import numpy as np
import matplotlib.pyplot as plt

from question_2 import Template, Likelihood
from data_loader import Data


filepath = 'simple-cw-master/'
filename = 'cw_' + 'example' + '.csv'

Likelihood = Likelihood()

data = Data(filepath + filename)

iterations = 1000

df_dt_guesses = np.random.uniform(low=0.5e-4, high=1.5e-4, size=iterations)
epsilon_guesses = np.random.uniform(low=0.5e-6, high=1.5e-6, size=iterations)
likelihoods = np.zeros(iterations)

for i in range(iterations):
    template = Template(df_dt=df_dt_guesses[i], epsilon=epsilon_guesses[i], f_rot0=data.f_rot0)

    model = template(data.times)
    likelihood = Likelihood(signal=data.signal, model=model)

    likelihoods[i] = likelihood

    plt.figure()
    plt.plot(data.signal[:100], label='signal')
    plt.plot(model[:100], label='model')
    plt.legend()
    plt.show()

print(set(likelihoods))