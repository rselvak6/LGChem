%% CE 295 - Energy Systems and Control
%   HW 5 : Optimal Energy Management of PHEV via Dynamic Programming
%   Raja Selvakumar, SID 18681868
%   Last updated: April 13, 2018
%   Modified template provided by Professor Scott Moura

clear; close all;
fs = 15;    % Font Size for plots

%% Parameters and Data

% Time step
Delta_t = 1;

% Fuel consumption in grams per unit energy
alph = 1e-4;     % [g/(s-W)]

Qcap = 5*3600;        % [A-s = Coulombs]
V_oc = 330;             % [volts]

% Limits on Batt Power, Eng Power, SOC
P_batt_max = 15e3; % [W]
P_eng_max = 35e3;  % [W]

SOC_min = 0.25;      % [-]
SOC_max = 0.9;      % [-]

% Plot Power Demand Data
M = csvread('UDDS_Pdem.csv',1,0);
t = M(:,1);
P_dem = M(:,2)*1e3;  % convert from kW to W
v_dc = M(:,3);

figure(1); clf;

subplot(2,1,1);
% plot speed
plot(t, v_dc);
title('UDDS speed vs. time');
xlabel('Time [s]');
ylabel('Cycle speed [m/s]');

subplot(2,1,2)
% plot power demand
plot(t, P_dem/1e3);
title('Power demand vs. time');
xlabel('Time [s]');
ylabel('Power demand [kW]');

% Plot Engine efficiency Curve
P_eng = linspace(0,P_eng_max,length(P_dem));
figure(2); clf;
plot(P_eng/1e3, eta_eng(P_eng)) % plot efficiency versus engine power, for total range of engine powers
title('Engine efficiency curve');
xlabel('Engine power [W]');
ylabel('$$\eta$$','Interpreter','latex');

%% Grid State and Preallocate
SOC_grid = (SOC_min:0.005:SOC_max)';

% Grid size
ns = length(SOC_grid);  % No. of states

% Planning horizon (time steps)
N = length(t);

% Preallocate Value Function (rows index state, columns index time)
V = inf*ones(ns,N+1);

% Preallocate Control (rows index state, columns index time)
u_star = zeros(ns,N);

%% Solve DP
tic;

% Boundary Condition of Value Function (Principle of Optimality)
V(:,N+1) = 0;

% Iterate backward in time
for k = N:-1:1

    % Iterate over SOC
    for idx = 1:ns
        
        % Find dominant bounds
        lb = max([Qcap*V_oc/Delta_t*(SOC_min-SOC_grid(idx)),...
            -P_batt_max, P_dem(k)-P_eng_max]);
        ub = min([Qcap*V_oc/Delta_t*(SOC_max-SOC_grid(idx)),...
            P_batt_max, P_dem(k)]);
        
        % Grid Battery Power between dominant bounds
        P_batt_grid = linspace(lb,ub,200)';
        
        % Compute engine power (vectorized for all P_batt_grid)
        P_eng = P_dem(k)-P_batt_grid;
        
        % Cost-per-time-step (vectorized for all P_batt_grid)
        g_k = alph*Delta_t.*P_eng./eta_eng(P_eng);
        
        % Calculate next SOC (vectorized for all P_batt_grid)
        SOC_nxt = SOC_grid(idx)+ Delta_t/(Qcap*V_oc).*P_batt_grid;
        
        % Compute value function at nxt time step (need to interpolate)
        V_nxt = interp1(SOC_grid,V(:,k+1),SOC_nxt,'linear');
        
        % Value Function (Principle of Optimality)
        [V(idx, k), ind] = min(g_k + V_nxt);
        
        % Save Optimal Control
        u_star(idx,k) = P_batt_grid(ind);

    end
end

solveTime = toc;
fprintf(1,'DP Solver Time %2.2f sec \n',solveTime);

%% Simulate Results

% Preallocate
SOC_sim = zeros(N,1);
P_batt_sim = zeros(N,1);
P_eng_sim = zeros(N,1);
J_sim = zeros(N,1);

% Initialize
SOC_0 = 0.3;    
SOC_sim(1) = SOC_0;

% Simulate PHEV Dynamics
for k = 1:(N-1)
    
    % Use optimal battery power, for given SOC (need to interpolate)
    P_batt_sim(k) = interp1(SOC_grid,u_star(:,k),SOC_sim(k),'linear');
    
    % Compute engine power
    P_eng_sim(k) = P_dem(k)-P_batt_sim(k);
    
    % Fuel Consumption
    J_sim(k) = alph*P_eng_sim(k)./eta_eng(P_eng_sim(k));
    
    % Time-step SOC dynamics
    SOC_sim(k+1) = SOC_sim(k) + Delta_t/(Qcap*V_oc).*P_batt_sim(k);
    
end

fprintf(1,'Final SOC %2.4f \n',SOC_sim(N));
fprintf(1,'Total Fuel Consumption %2.2f kg \n',sum(J_sim)/1e3);

%% Plot Results
figure(3); clf;

subplot(2,2,1);
% UDDS speed versus time 
plot(t, v_dc);
title('UDDS speed vs. time');
xlabel('Time [s]');
ylabel('Cycle speed [m/s]');
set(gca,'FontSize',fs)

subplot(2,2,2);
% SOC versus time
plot(t, SOC_sim);
title('SOC vs. time');
xlabel('Time [s]');
ylabel('State of charge');
set(gca,'FontSize',fs)

subplot(2,2,3);
% Accumulated fuel consumption [g] versus time
J = zeros(length(J_sim),1);
for i=1:length(J_sim)
    for j=1:i
        J(i) = J(i) + J_sim(j);
    end
end
plot(t, J);
title('Accumulated fuel consumption vs. time');
xlabel('Time [s]');
ylabel('Fuel consumption [g]');
set(gca,'FontSize',fs)

subplot(2,2,4);
% Battery and engine power [kW] versus time
plot(t, P_batt_sim/1e3,t, P_eng_sim/1e3);
title('Engine and battery power vs. time');
xlabel('Time [s]');
ylabel('Power [kW]');
legend('Battery power','Engine power','Location','northwest');
set(gca,'FontSize',fs)