# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 15:45:30 2018

@author: Raja Selvakumar
"""
#%%
import numpy as np
from scipy.interpolate import Rbf
import matplotlib.pyplot as plt

x = np.linspace(0, 4, 13)
y = np.array([0, 0.75, 1.3, 1.9, 2, 2.2, 2.7, 3, 3.5, 3.75, 3.875, 3.9375, 4])
X, Y = np.meshgrid(x, y)
Z = np.sin(np.pi*x/2) * np.exp(y/2)

x2 = np.linspace(0, 4, 65)
y2 = np.linspace(0, 4, 65)
f = Rbf(x, y, Z)
Z1 = f(X,Y)

fig, ax = plt.subplots(nrows=1, ncols=2)
ax[0].pcolormesh(X, Y, Z)

X2, Y2 = np.meshgrid(x2, y2)
Z = np.sin(np.pi*X2/2)*np.exp(Y2/2)
f = Rbf(X2, Y2, Z)
Z2 = f(x2,y2)
ax[1].pcolormesh(X2, Y2, Z2)

plt.show()