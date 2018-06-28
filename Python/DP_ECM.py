# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 11:57:24 2018

@author: rselvak6
"""

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

#free parameters
t_max = 1800
I_min = 0 #A
I_max = 46
V_min = 2 #V
V_max = 3.6
dt = 1

#derivative lists
dV1 = []
dV2 = []

def min_fn(t_f):
    for i in range(t_max,t_0,-dt):
        
