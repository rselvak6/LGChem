# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 19:36:08 2018

@author: rselvak6
"""

import numpy as np
import time as tm
import math
import scipy.integrate import as odeint
import matplotlib.pyplot as plt

#hyperparameters
Rc = 1.94 #K/W
Ru = 3.08
Cc = 62.7 #J/K
Cs = 4.5

#initial conditions
SOC_o = 0.5
T_inf = 25 #C
t_0 = 0 #sec

#Values taken from Perez et. al 2015 based on initial conditions
V_oc = 3.3 #V
C_1 = 2500 #F
C_2 = 5.5
R_0 = 0.01 #Ohms
R_1 = 0.01
R_2 = 0.02
I_min = 0 #A
I_max = 46
V_min = 2 #V
V_max = 3.6
SOC_min = 0.25
SOC_max = 0.75

#free parameters
t_max = 1799
dt = 1
V_oc = 1.5 #assume time-invariant for time being
C_batt = 3600 #guess?

#ECM models
def OCV_R(I,t):
    return I/C_batt
    
