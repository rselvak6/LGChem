# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 16:53:18 2018

@author: Raja Selvakumar
"""

import numpy as np
from scipy import interpolate as ip
import time
import math
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
V_oc = np.loadtxt('Voc.dat',delimiter=',',dtype=float)

#free parameters
t_max = 5*60
dt = 1
N = int((t_max-t_0)/dt)
#%%Playground
#%% Preallocate grid space
#state grids
SOC_grid = np.arange(SOC_min,SOC_max+0.01,0.01)
V1_grid = np.arange(0,I_max*R_0+0.1,0.1)
n1 = len(SOC_grid)
n2 = len(V1_grid)

#control and utility 
V = np.ones((N+1,n1,n2))*math.inf
I_opt = np.zeros((N,n1,n2))

#terminal Bellman
for i in range(0,n1):
    V[N,i,:] = (SOC_grid[i]-SOC_max)**2
#%% DP
start = time.time()
for k in range(N-1,t_0-1,-dt):
    if(k%10==0):
        print('Computing the Principle of Optimality at %2.2f[s]\n',k*dt)
    for ii in range(0,n1):
        for jj in range(0,n2):
            
            c_soc = SOC_grid[ii]
            c_v1 = V1_grid[jj]
            
            #lower/upper control bounds
            v_oc = [V_oc[x,1] for x in range(0,len(V_oc[:,0])-1) if V_oc[x,0]==round(c_soc,3)][0]
            I_vec = np.linspace(I_min,I_max,25)
            
            z_nxt_test = c_soc + dt/C_batt*I_vec
            V_nxt_test = v_oc + c_v1 + I_vec*R_0
            ind = np.argmin((z_nxt_test-SOC_min >= 0) & (SOC_max-z_nxt_test >= 0) & (V_nxt_test-V_min >= 0) & (V_max-V_nxt_test >= 0))
#            z_lb = C_batt/dt*(c_soc-SOC_max)
#            z_ub = C_batt/dt*(c_soc-SOC_min)
#            V1_lb = (V_min-c_v1-v_oc)/R_0
#            V1_ub = (V_max-c_v1-v_oc)/R_0
#            lb = max(I_min, z_lb, V1_lb)
#            ub = min(I_max, z_ub, V1_ub)
            #control initialization 
#            I_grid = np.linspace(lb,ub,200)
            
            #value function
            c_k = (c_soc-SOC_max)**2
            
            #iterate next SOC
            SOC_nxt = c_soc + I_vec[ind]/C_batt*dt          
            V1_nxt = c_v1*(1-dt/(R_1*C_1))+dt/C_1*I_vec[ind]
            
            #value function interpolation
            z = ip.interp2d(SOC_grid,V1_grid,np.transpose(np.squeeze(V[k+1,:,:])))
            V_nxt = z(SOC_nxt,V1_nxt)
            
            #Bellman
            V[k,ii,jj] = min(c_k+V_nxt)
            ind = np.argmin(c_k+V_nxt)
            
            #save optimal control
            I_opt[k,ii,jj] = I_vec[ind]
end = time.time()
print('DP solver time: %2.2f[s]\n',end-start)
#%% Simulation: Initialize
t_sim = range(0,N)
SOC_sim = np.zeros(N)
Vt_sim = np.zeros(N)
Voc_sim = np.zeros(N)
I_sim = np.zeros(N)
    
I_ub = [I_max]*N
I_lb = [I_min]*N
z_ub = [SOC_max]*N
z_lb = [SOC_min]*N
v_lb = [V_min]*N
v_ub = [V_max]*N
    
#initial condition
SOC_sim[0] = 0.25
 #%% Simulation: Simulate
   
for k in range(0,(N-1)):
    I_sim[k] = np.interp(SOC_sim[k],SOC_grid,ret[:,k])
    SOC_sim[k+1] = SOC_sim[k]+I_sim[k]*dt/C_batt
    Voc_sim[k] = [V_oc[x,1] for x in range(0,len(V_oc[:,0])-1) if V_oc[x,0]==round(SOC_sim[k],3)][0]
    Vt_sim[k] = Voc_sim[k] + I_sim[k]*R_0
    
#%% Plot Simulation Results
plt.figure(num=1, figsize=(10, 18), dpi=80, facecolor='w', edgecolor='k')
plt.subplot2grid((3,2), (0,0), colspan = 2)
# I vs time
plt.plot(t_sim,I_sim,'b',linewidth=1.5)
plt.plot(t_sim,I_ub,'r--',t_sim,I_lb,'r--')
plt.xlim(xmin=0.0)
plt.ylim(ymin=0.0)
plt.title('Current',fontsize=15,fontweight='bold')
plt.ylabel('I [A]',fontsize=12)
    
plt.subplot2grid((3,2), (1,0), colspan = 2)
# SOC vs time
plt.plot(t_sim,SOC_sim,'b',linewidth=1.5)
plt.plot(t_sim,z_ub,'r--',t_sim,z_lb,'r--')
plt.xlim(xmin=0.0)
plt.ylim(ymin=0.0)
plt.title('State of charge',fontsize=15,fontweight='bold')
plt.ylabel('z',fontsize=12)

plt.subplot2grid((3,2), (2,0))
# Vt vs time
plt.plot(t_sim[0:N-1],Vt_sim[0:N-1],'b',linewidth=1.5)
plt.plot(t_sim,v_ub,'r--',t_sim,v_lb,'r--')
plt.xlim(xmin=0.0)
plt.title('Terminal voltage',fontsize=15,fontweight='bold')
plt.xlabel('Time [s]',fontsize=12)
plt.ylabel('$V_{t}$ [V]',fontsize=12)   
    
plt.subplot2grid((3,2), (2,1))
# Voc vs time
plt.plot(t_sim[0:N-1],Voc_sim[0:N-1],'b',linewidth=1.5)
plt.xlim(xmin=0.0)
plt.title('Open Circuit Voltage',fontsize=15,fontweight='bold')
plt.xlabel('Time [s]',fontsize=12)
plt.ylabel('$V_{oc}$ [V]',fontsize=12)    

plt.show()