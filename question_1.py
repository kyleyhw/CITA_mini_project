import numpy as np

class ellipsoid:
    def __init__(self, I_x, I_y, I_z, density):
        self.I_x = I_x
        self.I_y = I_y
        self.I_z = I_z
        self.density = density

        self.ellipticity = (self.I_x - self.I_y) / self.I_z

        self.mass_quadrupole = np.zeros(shape=(3, 3), dtype=float)

        for i in range(len(self.mass_quadrupole)):
            for j in range(len(self.mass_quadrupole[i])):
                self.mass_quadrupole[i, j] = 0