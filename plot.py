#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import lfilter, butter, filtfilt, medfilt

headers = ["#", "#moves", "#illegal", "max", "rewards"]
dat = pd.read_csv('test.csv', names=headers)
dif = [0] + [j-i for i, j in zip(dat["rewards"][:-1], dat["rewards"][1:])]
fig = plt.figure()

ax = fig.add_subplot(1, 2, 1)
a, b = np.polyfit(dat.index, dat['rewards'], 1)
dif_regr = [a*x+b for x in dat.index]
med_dif_regr = medfilt(dat['rewards'], 49)
hard_med_dif_regr = medfilt(dat['rewards'], 599)
ax.plot(dat.index, dat['rewards'], color='#b4cfef')
ax.plot(dat.index, med_dif_regr, color='#5b8bc3')
ax.plot(dat.index, hard_med_dif_regr, color='#235899')
ax.plot(dat.index, dif_regr, color='red')

ax = fig.add_subplot(1, 2, 2)
ax.set_yticks([x for x in range(2, 11)])
ax.set_yticklabels([2**x for x in range(2, 11)])
logmax = np.log2(dat["max"])
ax.plot(dat.index, logmax, color='red', marker='o', markersize=0.1, linestyle='')

plt.show()
