%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%COPYRIGHT  ? 2013 
%THE REGENTS OF THE UNIVERSITY OF MICHIGAN 
%ALL RIGHTS RESERVED 
%PERMISSION IS GRANTED TO USE, COPY AND REDISTRIBUTE THIS SOFTWARE FOR NONCOMMERCIAL EDUCATION AND RESEARCH PURPOSES, SO LONG AS NO FEE IS CHARGED, 
%AND SO LONG AS THE COPYRIGHT NOTICE ABOVE, THIS GRANT OF PERMISSION, AND THE DISCLAIMER BELOW APPEAR IN ALL COPIES MADE; 
%AND SO LONG AS THE NAME OF THE UNIVERSITY OF MICHIGAN IS NOT USED IN ANY ADVERTISING OR PUBLICITY PERTAINING TO THE USE OR DISTRIBUTION OF THIS 
%SOFTWARE WITHOUT SPECIFIC, WRITTEN PRIOR AUTHORIZATION. PERMISSION TO MODIFY OR OTHERWISE CREATE DERIVATIVE WORKS OF THIS SOFTWARE IS NOT GRANTED. 

%THIS USE OF THIS SOFTWARE REQUIRES MATLAB(TM) AND SIMULINK(TM), A LICENSED PRODUCT OF THE MATHWORKS, INC.  
%THE SOFTWARE USER IS SOLELY RESOPNSIBLE FOR OBTAINTING AN APPROPRIATE LICENSE FOR MATLAB(TM) AND SIMULINK(TM).  
%NO LICENSE TO MATLAB(TM) AND SIMULINK(TM) IS GRANTED, INFERRED OR IMPLIED BY THE DISTRIBUTION OF THIS SOFTWARE. 

%THIS SOFTWARE IS PROVIDED AS IS, WITHOUT REPRESENTATION AS TO ITS FITNESS FOR ANY PURPOSE, AND WITHOUT WARRANTY OF ANY KIND, EITHER EXPRESS 
%OR IMPLIED, INCLUDING WITHOUT LIMITATION THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. 
%THE REGENTS OF THE UNIVERSITY OF MICHIGAN SHALL NOT BE LIABLE FOR ANY DAMAGES, INCLUDING SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, 
%WITH RESPECT TO ANY CLAIM ARISING OUT OF OR IN CONNECTION WITH THE USE OF THE SOFTWARE, EVEN IF IT HAS BEEN OR IS HEREAFTER ADVISED OF THE POSSIBILITY 
%OF SUCH DAMAGES.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all
close all

Cc=62.7; %J/K, Lumped Cell Core Heat Capacity
Cs=4.5; %J/K, Lumped Cell Surface (Casing) Heat Capacity
Rc=1.94; %K/W, conduction resistance; 
Ru = 15; %K/W, convection resistance; for natural convection, this number is above 10

load('V_avgC20.mat'); % open circuit voltage
load('r_rc_rc_scale_cap_A123_26650.mat'); % parameters for the electrical model
%%
load('dUdT'); %dU/dT for entropic heat calculation, from paper of Dr. Forgez
C_bat=2.3;  % battery capacity

Ts_0=25; % initial battery surface temperature,
Tc_0=25; % initial battery core temperature, 
Tf = 25; % ambient temperature
SOC_0 = 1; % initial SOC;
options=[];
t = 0:0.1:1800; % simulation time; don't over-discharge/charge the battery!
t = t';
I = 4.6*ones(length(t),1);  % current input, + for discharge, - for charge

ut=[t I]; % simulation input
[ts,xs,y]=sim('Battery_Electrothermal_Model.mdl', t, options, ut);
Tc= y(:,1); % simulated battery core temperature;
Ts= y(:,2); % simulated battery surface temperature;
SOC= y(:,3); % simulated battery SOC;
V_T = y(:,4); % simulated battery voltages;

plot(t,Ts,t,Tc)
xlabel('t (s)')
ylabel('T (^oC)')
legend('T_s','T_c')
figure;
subplot(2,1,1)
plot(t,SOC);
xlabel('t (s)')
ylabel('SOC')
subplot(2,1,2)
plot(t,V_T)
xlabel('t (s)')
ylabel('Voltage (V)')

