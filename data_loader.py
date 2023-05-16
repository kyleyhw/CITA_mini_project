import numpy as np
import csv

class Data:
    def __init__(self, filename):
        f_rot = []
        times = []
        h = []

        with open(filename) as file:
            raw_read = csv.reader(file)
            next(raw_read)
            for row in raw_read:
                f_rot.append(row[0])
                times.append(row[1])
                h.append(row[2])

        self.times = np.array(times)
        self.h = np.array(h)

        if len(set(f_rot)) != 1:
            print('non-unique f_rot')

        self.f_rot = f_rot[0]