%% OCV-R-RC
%   ADP formulation for a OCV-R-RC equivalent circuit model of a 2300 mAh
%   Li-ion A123 battery
%   Raja Selvakumar
%   08/04/2018
%   All parameter values taken from Perez, H.E; Hu, X; Dey, S; Moura, S.J. 2015 

%% Load data
clc; clear;
load ECM_params.mat;
load OCVRRC.mat;
fs = 15;
%% Initialize ADP variables
N = (t_max-t_0)/dt; % total time scale

% ADP parameters
iter = 500;
eta_a = 7e-5;
eta_c = 9e-5;
N_ah = 3; % 1 hidden layer, 3 neurons
N_ch = 3; 
m = 2; % 2 states [SOC, V1]
n = 1; % 1 output [I]
%% Solve ADP
clc;
tic;
x = [0.25; 0]; % initial values for states
target = [z_target 0 0]'; % reward: get to SOC target
for k = 1:(N-1) %time
    if mod(k,10)==0
        fprintf(1,'Computing actor-critic algorithm at %3.0f sec\n',k*dt);
    end
    
    w_a1 = zeros(N_ah,m);
    w_a2 = zeros(n,N_ah);
    w_c1 = zeros(N_ch,m+n);
    w_c2 = zeros(1,N_ch);
    J_prev = 0;
    
    while e_c < tol
        for i=1:iter
            a = Actor(iter,w_a1,w_a2,eta_a);
            a.q_az = applywa1(a,x);
            u = applywa2(a);
            c = Critic(iter,w_c1,w_c2,eta_c);
            c.q_cs = applywc1(c,x,u);
            J_hat = predictJ(c);
            e_c = cerror(J_hat,J_prev,x,u,target);
        end
    end
end

solveTime = toc;
fprintf(1,'ADP Solver Time %2.0f min %2.0f sec \n',floor(solveTime/60),mod(solveTime,60));

%% Simulate Results

% Preallocate
SOC_sim = nan*ones(N,1);
I_sim = nan*ones(N-1,1);
V_sim = nan*ones(N-1,1);
V1_sim = nan*ones(N,1);
Voc_sim = nan*ones(N-1,1);

% Initialize
SOC_sim(1) = z_0;
V1_sim(1) = 0;

% Simulate Battery Dynamics
tic;
for k = 1:N-1
    if mod(k,10)==0
        fprintf(1,'Simulating at %3.0f sec\n',k*dt);
    end
    % Calculate optimal control for given state
    I_sim(k) = interp2(SOC_grid,V1_grid,squeeze(u_star(k,:,:))',SOC_sim(k),...
        V1_sim(k),'linear');
    
    % Voc, terminal voltage 
    Voc_sim(k) = voc(soc==round(SOC_sim(k),3));
    V_sim(k) = Voc_sim(k) + V1_sim(k) + I_sim(k).*R_0;
    
    % Dynamics
    SOC_sim(k+1) = SOC_sim(k) + dt/C_batt.*I_sim(k);
    V1_sim(k+1) = V1_sim(k)*(1-dt/(R_1*C_1)) + dt/C_1.*I_sim(k);
end
solveTime = toc;
fprintf(1,'Final SOC %2.3f \n',SOC_sim(N));
fprintf(1,'Terminal voltage %2.3f \n',V_sim(N-1));
fprintf(1,'Simulation Time %2.0f min %2.0f sec \n',floor(solveTime/60),mod(solveTime,60));
%% Plot Results
clc;
figure; clf;
t_1 = linspace(t_0,t_max,length(b));
t = linspace(t_0,t_max,N);

%current
subplot(3,2,[1 2]);
plot(t(1:N-1), I_sim,'k','LineWidth',0.5,'DisplayName','z_{max}=0.8');
hold on
% plot(t_2(1:length(a1)), a1,'r','LineWidth',0.5,'DisplayName','280s');
plot(t_1(1:length(a)), a,'b','LineWidth',0.5,'DisplayName','z_{max}=0.75');
% plot(t,I_min.*ones(length(t),1),'r--','LineWidth',0.5,...
%     'DisplayName','lb');
% plot(t,I_max.*ones(length(t),1),'r--','LineWidth',0.5,...
%     'DisplayName','ub');
hold off
title('Current vs. time','FontSize',fs);
ylabel('\it{I^{*}} \rm{[A]}','FontSize',13);
lgd = legend('show');
lgd.FontSize = 10;
lgd.Location = 'East';

%SOC
subplot(3,2,[3 4]);
plot(t, SOC_sim,'k','LineWidth',0.5,'DisplayName','z_0=0.5');
hold on
plot(t_1, b,'b','LineWidth',0.5,'DisplayName','z_0=0.25');
% plot(t,z_min.*ones(length(t),1),'r--','LineWidth',0.5,...
%     'DisplayName','z_{min}');
% plot(t,z_max.*ones(length(t),1),'r--','LineWidth',0.5,...
%     'DisplayName','z_{min}');
hold off
title('SOC vs. time','FontSize',fs);
xlabel('\it{t} \rm{[s]}','FontSize',13);
ylabel('\it{z}','FontSize',13);

%terminal voltage
subplot(3,2,5);
plot(t(1:N-1), V_sim,'k','LineWidth',0.5,'DisplayName','z_0=0.5');
hold on
plot(t_1(1:length(a)), d,'b','LineWidth',0.5,'DisplayName','z_0=0.25');
% plot(t,V_min.*ones(length(t),1),'r--','LineWidth',0.5,...
%     'DisplayName','V_{min}');
% plot(t,V_max.*ones(length(t),1),'r--','LineWidth',0.5,...
%     'DisplayName','V_{max}');
hold off
title('Terminal voltage vs. time','FontSize',fs);
xlabel('\it{t} \rm{[s]}','FontSize',13);
ylabel('\it{V_T} \rm{[V]}','FontSize',13);

subplot(3,2,6);
plot(t, V1_sim,'k','LineWidth',0.5);
hold on
plot(t_1, c,'b','LineWidth',0.5);
hold off
title('Capacitor voltage vs. time','FontSize',fs);
xlabel('\it{t} \rm{[s]}','FontSize',13);
ylabel('\it{V_{1}} \rm{[V]}','FontSize',13);
%% Save data
a = I_sim;
b = SOC_sim;
c = V1_sim;
d = V_sim;
e = Voc_sim;
save OCVRRC.mat a b c d e