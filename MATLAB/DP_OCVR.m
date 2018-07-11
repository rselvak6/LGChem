%% OCV-R
%   DP formulation for the max charge problem with OCV-R ECM 
%   Raja Selvakumar
%   07/10/2018
%   energy, Controls, and Application Lab (eCAL)

clc; clear;
%% OCV save
Rc = 1.94;
Ru = 3.08;
Cc = 62.7;
Cs = 4.5;
z_0 = 0.5;
T_inf = 25;
t_0 = 0;
C_1 = 2500;
C_2 = 5.5;
R_0 = 0.01;
R_1 = 0.01;
R_2 = 0.02;
I_min = 0;
I_max = 46;
V_min = 2;
V_max = 3.6;
z_min = 0.1;
z_max = 0.9;
C_batt = 2.3*3600;
t_max = 5*60;
dt = 1;
save ECM_params.mat;
%% Voc save
VOC_data = csvread('Voc.dat',1,0);
soc = VOC_data(:,1);
voc = VOC_data(:,2);
save OCV_params.mat;
%% Load data
clc; clear;
load ECM_params.mat;
load OCV_params.mat;
fs = 15;
clear VOC VOC_data;
%% Playground
%% Grid State and Preallocate
SOC_grid = (z_min:0.005:z_max)';
ns = length(SOC_grid);  % #states
N = t_max-t_0; % #iterations
V = inf*ones(ns,N+1); % #value function
u_star = zeros(ns,N);% #control

%% Solve DP
tic;
V(:,N+1) = 0; %Bellman terminal boundary condition

for k = N:-1:1 %time
    for idx = 1:ns %state (SOC)
        c_soc = SOC_grid(idx);
        c_voc = voc(soc==c_soc); %return the voc value when soc = c_soc 
        
        % Bounds
        lb = max([I_min, C_batt/dt*(c_soc-z_max),(V_min-c_voc)/R_0]);
        ub = min([I_max, C_batt/dt*(c_soc-z_min),(V_max-c_voc)/R_0]);
        
        % Control grid
        I_grid = linspace(lb,ub,200)';
        % Cost-per-time-step
        g_k = dt.*I_grid;
        
        % State dynamics
        SOC_nxt = c_soc+ dt/C_batt.*I_grid;
        
        % Linear interpolation for value function
        V_nxt = interp1(SOC_grid,V(:,k+1),SOC_nxt,'linear');   
        % Bellman
        [V(idx, k), ind] = min(-g_k + V_nxt);
        
        % Save Optimal Control
        u_star(idx,k) = I_grid(ind);
    end
end

solveTime = toc;
clc;
fprintf(1,'DP Solver Time %2.2f sec \n',solveTime);

%% Simulate Results

% Preallocate
SOC_sim = zeros(N,1);
I_sim = zeros(N,1);
V_sim = zeros(N,1);

% Initialize
SOC_0 = 0.5;    
SOC_sim(1) = SOC_0;

% Simulate Battery Dynamics
for k = 1:N
    % Calculate optimal control for given state
    I_sim(k) = interp1(SOC_grid,u_star(:,k),SOC_sim(k),'linear');
    
    % Terminal voltage  
    V_sim(k) = voc(soc==round(SOC_sim(k),3)) + I_sim(k).*R_0;
    
    % SOC dynamics
    SOC_sim(k+1) = SOC_sim(k) + dt/C_batt.*I_sim(k); 
end

fprintf(1,'Final SOC %2.3f \n',SOC_sim(N));
fprintf(1,'Terminal voltage %2.3f \n',V_sim(N-1));
%% Plot Results
figure; clf;
t = linspace(t_0,t_max,t_max-t_0);

%current
subplot(3,1,1);
plot(t, I_sim,'b');
hold on
plot(t,I_min.*ones(length(t),1),'r--','LineWidth',0.5);
plot(t,I_max.*ones(length(t),1),'r--','LineWidth',0.5);
hold off
title('Current vs. time');
xlabel('Time [s]');
ylabel('Current [A]');
set(gca,'FontSize',fs)

%terminal voltage
subplot(3,1,2);
plot(t, V_sim,'b');
hold on
plot(t,V_min.*ones(length(t),1),'r--','LineWidth',0.5);
plot(t,V_max.*ones(length(t),1),'r--','LineWidth',0.5);
hold off
title('Terminal voltage vs. time');
xlabel('Time [s]');
ylabel('V_t [V]');
set(gca,'FontSize',fs)

%SOC
subplot(3,1,3);
plot(t, SOC_sim(1:N),'b');
hold on
plot(t,z_min.*ones(length(t),1),'r--','LineWidth',0.5);
plot(t,z_max.*ones(length(t),1),'r--','LineWidth',0.5);
hold off
title('SOC vs. time');
xlabel('Time [s]');
ylabel('SOC');
set(gca,'FontSize',fs)
%% Outdated
%{
%% Write Data
p_nm = ["Rc" "Ru" "Cc" "Cs" "z_0" "T_inf" "t_0" "C_1" "C_2" "R_0" "R_1" ...
    "R_2" "I_min" "I_max" "V_min" "V_max" "z_min" "z_max" "C_batt" "t_max" ...
    "dt"];
p_val = [1.94 3.08 62.7 4.5 0.5 25 0 2500 5.5 0.01 0.01 0.02 0 46 2 3.6 ...
    0.1 0.9 2.3*3600 1800 1];
fid = fopen('ECM_params.dat','w');
fprintf(fid,'%s %6.2f\n',[p_nm; p_val]);
fclose(fid);
%% Parameters and Data
fid = fopen('ECM_params.dat','r');
p = textscan(fid,'%s%8.2f');
fclose(fid);
%}
