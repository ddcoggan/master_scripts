"""
Makes gamma correction table for psychopy based on
a csv file of photometer readings.
"""
import os
import os.path as op
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np

def full_gamma(V, a, b, k, gamma):
    L = a + (b + k * V) ** gamma
    return L

os.chdir(op.dirname(__file__))
readings = pd.read_csv('readings.csv', header=None)
r, g, b = readings[0], readings[1], readings[2]
l = np.mean(readings, axis=1)

plt.plot(range(len(readings)), r, c='r')
plt.plot(range(len(readings)), g, c='g')
plt.plot(range(len(readings)), b, c='b')
plt.plot(range(len(readings)), l, c='k')
plt.show()

correction_table = pd.DataFrame()
for channel, measurements in zip(['lum', 'R', 'G', 'B'], [l, r, g, b]):
    (a, b, k, gamma), pcov = curve_fit(full_gamma, np.linspace(0,1,33), measurements)
    plt.plot(np.linspace(0,1,33), measurements, c='k')
    plt.plot(np.linspace(0,1,33), full_gamma(np.linspace(0,1,33), *[a,b,k,gamma]), c='r')
    plt.title(channel)
    plt.show()
    correction_table = pd.concat([correction_table, pd.DataFrame({
        'Min': min(measurements),
        'Max': max(measurements),
        'Gamma': gamma,
        'a': a,
        'b': b,
        'k': k}, index=[channel])])
    correction_table.to_csv('correction_table.csv')

