import numpy as np
import emcee as mc

class Template:
    def __init__(self, theta=np.pi/4, phi=0, iota=0, f_rot0=0, df_dt=-1, I_3=(2/5) * (1.4 * 2 * 1e30) * (1.2 * 1e4)**2, epsilon=0, r=3.086e+20):  #spoofed 10 times closer
        self.theta = theta
        self.phi = phi
        self.iota = iota
        self.f_rot0 = f_rot0
        self.df_dt = df_dt
        self.I_3 = I_3
        self.epsilon = epsilon
        self.r = r

        self.G = 6.67 * 1e-11 # m^3 kg^-1 s^-2
        self.c = 3e8 # m s^-1

    def __call__(self, t):
        self.f_gw = self.df_dt * t + 2*self.f_rot0
        self.h_0 = 4 * np.pi ** 2 * self.G / self.c ** 4 * self.I_3 * self.f_rot0 ** 2 / self.r * self.epsilon

        h_plus = (1/2) * (1 + np.cos(self.theta)**2) * np.cos(2*self.phi) * self.h_0 * (1 + np.cos(self.iota)**2) / 2 * np.cos(2 * np.pi * self.f_gw * t)
        h_cross = np.cos(self.theta) * np.sin(2 * self.phi) * self.h_0 * np.cos(self.iota) * np.sin(2 * np.pi * self.f_gw * t)

        h = h_plus + h_cross

        return h

class InnerProduct:
    def __init__(self, frequency=2 * 182, sensitivity=(1e-24)**2):
        self.frequency = frequency
        self.sensitivity = sensitivity

        self.sigma = np.sqrt(sensitivity) / frequency

    def __call__(self, signal, model):
        result = np.sum((signal - model) ** 2 / (len(signal) * self.sigma ** 2))
        return result

class Likelihood:
    def __init__(self):
        pass
    def __call__(self, signal, model):
        inner_product = InnerProduct()
        result = np.exp((-1/2) * inner_product(signal, model))
        return result

class LogProb:
    def __init__(self):
        pass

    def __call__(self, p0, times, signal):
        self.epsilon_guess = p0[0]
        self.df_dt_guess = p0[1]

        template = Template(epsilon=self.epsilon_guess, df_dt=self.df_dt_guess)
        model = template(times)
        likelihood = Likelihood()
        prob = likelihood(signal, model)

        return np.log(prob)
