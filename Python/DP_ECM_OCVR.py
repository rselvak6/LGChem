# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 11:57:24 2018

@author: rselvak6
"""
import numpy as np
import time as tm
import math
import matplotlib.pyplot as plt
from scipy import interp

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
C_1 = 2500 #F
C_2 = 5.5
R_0 = 0.01 #Ohms
R_1 = 0.01
R_2 = 0.02
I_min = 0 #A
I_max = 46
V_min = 2 #V
V_max = 3.6
SOC_min = 0.1
SOC_max = 0.75
C_batt = 2.3*3600;

#Voc from file
V_oc = np.loadtxt('/home/rselvak6/Documents/LGChem/Python/Voc.dat',delimiter=',',dtype=float)

#free parameters
t_max = 5*60
dt = 1
N = t_max-t_0
#%%Playground

#%% Preallocate grid space
#state grids
SOC_grid = np.arange(SOC_min,SOC_max,0.005)
num_states = len(SOC_grid)

#control and utility 
V = np.ones((num_states,N+1))*math.inf
I_opt = np.zeros((num_states,N))

#terminal Bellman
V[:,N] = [(SOC_grid[k]-SOC_max)**2 for k in range(0,num_states)]

#%% DP
def DP():
    start = tm.time()
    for k in range(t_max-1,t_0,-dt):
        for idx in range(0,num_states):
            
            #lower/upper control bounds
            v_oc = [V_oc[len(V_oc[:,0])*x,1] for x in V_oc[:,0] if x==SOC_grid[idx]][0]
            lb = max(I_min, C_batt/dt*(SOC_grid[idx]-SOC_max), (V_min-v_oc)/R_0)
            ub = min(I_max, C_batt/dt*(SOC_grid[idx]-SOC_min), (V_max-v_oc)/R_0)
                
            #control initialization 
            I_grid = np.linspace(lb,ub,200)
            
            #value function
            c_k = (SOC_grid[idx]-SOC_max)**2
            
            #iterate next SOC
            SOC_nxt = SOC_grid[idx] + I_grid/C_batt*dt          
            
            #value function interpolation
            V_nxt = interp(SOC_grid,SOC_nxt,V[:,k+1])
            
            #Bellman
            V[idx,k] = min(c_k+V_nxt)
            ind = np.argmin(c_k+V_nxt)
            
            #save optimal control
            I_opt[idx,k] = I_grid[ind]
    stop = tm.time()
    print('DP Time:', '%.2f'%(stop-start),'s')
    return I_opt
#%% Simulation   
def sim(I_opt):
    t_sim = range(0,(N-1))
    SOC_sim = [0]*N
    Vt_sim = [0]*(N-1)
    Voc_sim = [0]*(N-1)
    I_sim = [0]*(N-1)
    
    
    I_ub = [I_max]*N
    I_lb = [I_min]*N
    z_ub = [SOC_max]*(N-1)
    z_lb = [SOC_max]*(N-1)
    v_lb = [V_min]*N
    v_ub = [V_max]*N
    
    #initial condition
    SOC_sim[0] = SOC_min
    
    for i in range(0,(N-1)):
        I_sim[i] = np.interp(SOC_grid,SOC_sim[i],I_opt[:,i])
        SOC_sim[i+1] = SOC_sim[i]+I_sim[i]*dt/C_batt
        Voc_sim[i] = [V_oc[len(V_oc[:,0])*x,1] for x in V_oc[:,0] if x==SOC_sim[i]][0]
        Vt_sim[i] = Voc_sim[i] + I_sim[i]*R_0
    
    ## Plot Simulation Results
    plt.figure(num=1, figsize=(10, 18), dpi=80, facecolor='w', edgecolor='k')
    plt.subplot2grid((3,2), (0,0), colspan = 2)
    # I vs time
    ax1 = plt.plot(t_sim,I_sim,'b','linewidth',1.5,t_sim,I_ub,'r--',t_sim,I_lb,'r--')
    ax1.set_title('Current vs. time')
    ax1.set_ylabel('I [A]')
    
    plt.subplot2grid((3,2), (1,0), colspan = 2)
    # SOC vs time
    ax2 = plt.plot(t_sim,SOC_sim[0:N-1],'b','linewidth',1.5,t_sim,z_ub,'r--',t_sim,z_lb,'r--')
    ax2.set_title('SOC (z) vs. time')
    ax2.set_ylabel('z')
    
    plt.subplot2grid((3,2), (2,0))
    # Vt vs time
    ax3 = plt.plot(t_sim,Vt_sim,'b','linewidth',1.5,t_sim,v_ub,'r--',t_sim,v_lb,'r--')
    ax3.set_title('SOC (z) vs. time')
    ax3.set_ylabel('z')   
    
    plt.subplot2grid((3,2), (2,1))
    # Voc vs time
    ax4 = plt.plot(t_sim,Voc_sim,'b','linewidth',1.5)
    ax4.set_title('Voc vs. time')
    ax4.set_ylabel('V_{oc} [V]')    
    
    plt.show()
    
    return

#%% Main 
ret = DP()
sim(ret)