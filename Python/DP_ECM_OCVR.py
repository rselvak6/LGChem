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
C_batt = 2.3*3600;

#Voc from file
V_oc = np.loadtxt('/home/rselvak6/Documents/LGChem/Python/Voc.dat',delimiter=',',dtype=float)

#free parameters
t_max = 1800
dt = 1
N = t_max-t_0
#%%Playground
test= [V_oc[len(V_oc[:,0])*x,1] for x in V_oc[:,0] if x==0.25][0]

#%% Preallocate grid space
#state grids
SOC_grid = np.arange(SOC_min,SOC_max,0.005)
num_states = len(SOC_grid)

#control and utility 
V = np.ones((num_states,N+1))*math.inf
I_opt = np.zeros((num_states,N))

#terminal conditions
SOC_f = SOC_max
V[:,N] = 0
#%% DP
def DP():
    start = tm.time()
    for k in range(t_max-1,t_0-1,-dt):
        for idx in range(0,num_states):
            
            #lower/upper control bounds
            v_oc = [V_oc[len(V_oc[:,0])*x,1] for x in V_oc[:,0] if x==0.25][0]
            lb = max(I_min, C_batt/dt*(SOC_grid[idx]-SOC_max), (V_min-v_oc)/R_0)
            ub = min(I_max, C_batt/dt*(SOC_grid[idx]-SOC_min), (V_max-v_oc)/R_0)
            if k==t_max-1:
                ub = min(I_max, C_batt/dt*(SOC_grid[idx]-SOC_f), (V_max-v_oc)/R_0) 
                
            #control initialization 
            I_grid = np.linspace(lb,ub,100)
            
            #value function
            c_k = I_grid 
            
            #iterate next SOC
            SOC_nxt = SOC_grid[idx] + I_grid/C_batt*dt          
            
            #value function interpolation
            V_nxt = np.interp(SOC_grid,V[:,k+1],SOC_nxt)
            
            #Bellman
            print(max(c_k + V_nxt))
            [V[idx,k],ind] = max(c_k+V_nxt)
            
            #save optimal control
            I_opt[idx,k] = I_grid[ind]
    stop = tm.time()
    print('DP Time:', '%.2f'%(stop-start),'s')
    return I_opt
#%% Simulation   
def sim(I_opt):
    SOC_sim = [0]*N
    I_sim = [0]*N
    #initial condition
    SOC_sim[0] = SOC_min
    for i in range(t_0,t_max-1,dt):
        I_sim[i] = np.interp(SOC_grid,I_opt[:,i],SOC_sim[i])
        SOC_sim[i+1] = SOC_sim[i]+I_sim[i]*dt/C_batt
    return SOC_sim

#%% Main 
ret = DP()
SOC_out = sim(ret)