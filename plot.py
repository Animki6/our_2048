#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import lfilter, butter, filtfilt, medfilt

headers = ["#", "#moves", "#illegal", "max", "rewards"]
dat = pd.read_csv('test.csv', names=headers)
dif = [0] + [j-i for i, j in zip(dat["rewards"][:-1], dat["rewards"][1:])]

a, b = np.polyfit(dat.index, dat['rewards'], 1)
dif_regr = [a*x+b for x in dat.index]
print(a)
print(b)
#filtr_a, filtr_b = butter(16, 0.2)
data_to_plot = medfilt(dat['rewards'], 49)
plt.plot(data_to_plot)
plt.plot(dat.index, dif_regr)

plt.show()
