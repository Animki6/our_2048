#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import lfilter, butter, filtfilt, medfilt
import argparse
import sys
import colorsys
from matplotlib.patches import Polygon
import matplotlib.cm as cm
from matplotlib.collections import PatchCollection

def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)

### PARSE ARGUMENTS
parser = argparse.ArgumentParser(description=None)
parser.add_argument('-v', '--version', default=1, type=int, help='')
args = parser.parse_args()
if args.version <= 0:
    print("Error: version number must be 1 or higher.")
    sys.exit(1)
v = args.version
print("Using plotter version {}.".format(v))

### READ DATA
headers = ["#", "#moves", "#illegal", "max", "rewards"]
dat = pd.read_csv('test.csv', names=headers)
dif = [0] + [j-i for i, j in zip(dat["rewards"][:-1], dat["rewards"][1:])]
fig = plt.figure()

### ADD SCORE PLOT
ax = fig.add_subplot(1, 2, 1)
a, b = np.polyfit(dat.index, dat['rewards'], 1)
dif_regr = [a*x+b for x in dat.index]
med_dif_regr = medfilt(dat['rewards'], 49)
hard_med_dif_regr = medfilt(dat['rewards'], 599)
ax.plot(dat.index, dat['rewards'], color='#b4cfef')
ax.plot(dat.index, med_dif_regr, color='#5b8bc3')
ax.plot(dat.index, hard_med_dif_regr, color='#235899')
ax.plot(dat.index, dif_regr, color='red')

### ADD TILE SCATTER PLOT
ax = fig.add_subplot(1, 2, 2)
trange = list(range(1, 11))
ax.set_yticks([x for x in trange])
ax.set_yticklabels([2**x for x in trange])
logmax = np.log2(dat["max"])
if v == 1:
    ax.plot(dat.index, logmax, color='red', marker='o', markersize=0.1, linestyle='')
elif v == 2:
    print("Running tile scatter plot in version {}. This may take a while.".format(v))
    sers = dict()
    for x in trange:
        print("Processing tile {}.".format(2**x))
        is_tile = [float(i) for i in (x == logmax)]
        r = 100 # Mean frame size - tweak me!
        filtered = [sum(is_tile[slice(i-r, i+r)])/float(r) for i in range(len(is_tile))] # TODO: check bounduaries?
        colors = [colorsys.hsv_to_rgb(0, 1, clamp(0.1+x/1.6, 0, 1)) for x in filtered]
        s = 500 # Scaling factor - tweak me!
        filtered_scaled = [s*x for x in filtered]
        ax.scatter(dat.index, len(dat.index)*[x], s=filtered_scaled, marker='|', color=colors)
else:
    print("Running tile scatter plot in version {}.".format(v))
    polys = []
    for x in trange:
        print("Processing tile {}.".format(2**x))
        is_tile = [float(i) for i in (x == logmax)]
        r = 100 # Mean frame size - tweak me!
        filtered = [sum(is_tile[slice(i-r, i+r)])/float(r) for i in range(len(is_tile))] # TODO: check bounduaries?
        # colors = [colorsys.hsv_to_rgb(0, 1, clamp(0.1+x/1.6, 0, 1)) for x in filtered]
        s = 20 # Scaling factor - tweak me!
        # filtered_scaled = [s*x for x in filtered]
        # print(filtered + filtered[::-1])
        # print(dat.index + dat.index[::-1])
        # print(list(zip(filtered + filtered[::-1], dat.index + dat.index[::-1])))
        xs = list(range(len(logmax))) + list(range(len(logmax))[::-1])
        # print(xs)
        ys = [s*x for x in (filtered + filtered[::-1])]
        print(ys)
        polys.append(Polygon(list(zip(xs, ys))))
        # print(list(zip(xs, ys)))
        # for x,y in zip(xs, ys):
        #     print("({},{})".format(x,y), end='')
        break
    # ax.scatter(dat.index, len(dat.index)*[x], s=filtered_scaled, marker='|', color=colors)
    polys.append(Polygon([[0,12], [100, 12], [100, 0], [0, 0]]))
    p = PatchCollection(polys, cmap=cm.jet)
    colors = 100*np.random.rand(len(polys))
    p.set_array(np.array(colors))
    # colors = 100*np.random.rand(len(patches))
    # p.set_array(np.array(colors))
    ax.add_collection(p)

### SHOW RESULTS
plt.show()
