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
parser.add_argument('file', type=str, nargs='?', help='input csv file')
args = parser.parse_args()
if args.version <= 0:
    print("Error: version number must be 1 or '?'.")
    sys.exit(1)
v = args.version
print("Using plotter version {}.".format(v))

### READ DATA
headers = ["#", "#moves", "#illegal", "max", "rewards"]
if args.file is None:
    print("Error: provide a file.")
    sys.exit(1)
dat = pd.read_csv(args.file, names=headers)
dif = [0] + [j-i for i, j in zip(dat["rewards"][:-1], dat["rewards"][1:])]
fig = plt.figure()

### ADD SCORE PLOT
ax = fig.add_subplot(1, 2, 1)
if v > 1:
    ax.set_ylim([round(min(dat['rewards'])-500,-3),round(max(dat['rewards'])+500,-3)])
    ax.set_xlim([dat.index[0], dat.index[-1]])
# a, b = np.polyfit(dat.index, dat['rewards']) # TODO: Regression does not work
# dif_regr = [a*x+b for x in dat.index] # TODO
med_dif_regr = medfilt(dat['rewards'], 49)
hard_med_dif_regr = medfilt(dat['rewards'], 599)
ax.plot(dat.index, dat['rewards'], color='#b4cfef')
ax.plot(dat.index, med_dif_regr, color='#5b8bc3')
ax.plot(dat.index, hard_med_dif_regr, color='#235899')
# ax.plot(dat.index, dif_regr, color='red') # TODO

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
        r = 100 # Frame size - tweak me!
        filtered = [sum(is_tile[slice(i-r, i+r)])/float(r) for i in range(len(is_tile))] # TODO: check bounduaries?
        s = 0.5 # Scaling factor - tweak me!
        filtered_scaled = [s*i for i in filtered]
        filtered_top = [x+i for i in filtered_scaled]
        filtered_bottom = [x-i for i in filtered_scaled]
        xs = list(range(len(logmax))) + list(range(len(logmax))[::-1])
        ys = filtered_top + filtered_bottom[::-1]
        polys.append(Polygon(list(zip(xs, ys))))
    p = PatchCollection(polys, cmap=cm.jet)
    ax.set_ylim([trange[0] - 0.5, trange[-1] + 0.5])
    ax.set_xlim([dat.index[0], dat.index[-1]])
    ax.add_collection(p)

### SHOW RESULTS
plt.show()
fig.savefig(args.file + ".png")
