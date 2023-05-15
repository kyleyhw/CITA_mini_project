import numpy as np

class Template:
    def __init__(self, theta=0, phi=0, iota=0, f_rot0=1, df_dt=-1, I_3=0, epsilon=0, r=1):
        self.theta = theta
        self.phi = phi
        self.iota = iota
        self.f_rot0 = f_rot0
        self.df_dt = df_dt
        self.I_3 = I_3
        self.epsilon = epsilon
        self.r = r

        self.G = 1
        self.c = 1



    def __call__(self, t):
        self.f_gw = self.df_dt * t + self.f_rot0
        self.h_0 = 4 * np.pi ** 2 * self.G / self.c ** 4 * self.I_3 * self.f_gw ** 2 / self.r * self.epsilon


        h_plus = (1/2) * (1 + np.cos(self.theta)**2) * np.cos(2*self.phi) * self.h_0 * 1 + (1 + np.cos(self.iota)**2) / 2 * np.cos(2 * np.pi * self.f_gw * t)
        h_cross = np.cos(self.theta) * np.sin(2 * self.phi) * self.h_0 * np.cos(self.iota) * np.sin(2 * np.pi * self.f_gw * t)

        h = h_plus + h_cross

        return h

class InnerProduct:
    def __init__(self, frequency, sensitivity):
        self.frequency = frequency
        self.sensitivity = sensitivity

