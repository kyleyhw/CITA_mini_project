import numpy as np
import csv

class Data:
    def __init__(self, filename):
        f_rot = [182]
        times = []
        signal = []

        with open(filename) as file:
            raw_read = csv.reader(file)
            next(raw_read)
            for row in raw_read:
                # f_rot.append(float(row[0]))
                times.append(float(row[0]))
                signal.append(float(row[1]))

        self.times = np.array(times)
        self.signal = np.array(signal)

        if len(set(f_rot)) != 1:
            print('non-unique f_rot')

        self.f_rot0 = f_rot[0]