# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 11:57:24 2018

@author: rselvak6
"""
import numpy as np
import time as tm
import math

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

#Voc from file
Voc = np.fromfile('Voc.dat',dtype=float)

#free parameters
t_max = 1800
dt = 1

#grids
SOC_grid = np.arange(SOC_min,SOC_max,0.005)

num_states = len(SOC_grid)
N = t_max-t_0

#control and utility 
I_opt = np.zeros((num_states,N))
V = np.ones((num_states,N+1))*math.inf

#terminal conditions
SOC_f = SOC_max
V[:,N+1] = 0

def DP():
    start = tm.time()
    for i in range(t_max,t_0-1,-dt):
        for j in range(0,num_states):
            lb = max(I_min, C_batt/dt*(SOC_grid[j]-SOC_max))
            ub = min(I_max, C_batt/dt*(SOC_grid[j]-SOC_min))
            if i==t_max:
                ub = min(I_max, C_batt/dt*(SOC_grid[j]-SOC_max))  
            I_opt_grid = np.linspace(lb,ub,100)
            C_k = 1 #coefficient of input term            
            
            I_opt[-i,j] = (ub-lb)/2;   
    stop = tm.time()
    print('DP Time:', '%.2f'%(stop-start),'s')
    return I_opt
    
def sim(I_sim):
    SOC_sim = np.zeros((N,num_states))
    SOC_sim[t_0,:] = [SOC_min]*num_states
    for i in range(t_0+1,t_max,dt):
        for j in range(0,num_states):
            SOC_sim[i,j] = SOC_sim[i-1,j]+I_sim[i,j]*dt/C_batt
    return SOC_sim
    
ret = DP()
SOC_out = sim(ret)
print(['%.2f'%j for j in ret[1000,:]])