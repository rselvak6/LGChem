%% OCV-R-RC
%   DP formulation for the min OCV-R-RC SOC error problem
%   Raja Selvakumar
%   07/12/2018
%   energy, Controls, and Application Lab (eCAL)

clc; clear;
%% OCV save
Rc = 1.94;
Ru = 3.08;
Cc = 62.7;
Cs = 4.5;
z_0 = 0.25;
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
z_max = 0.95;
z_target = 0.75;
C_batt = 2.3*3600;
t_max = 300;
dt = 1;
beta = 0.5;
save ECM_params.mat;
%% Voc save
clear;
VOC_data = csvread('Voc.dat',1,0);
soc = VOC_data(:,1);
voc = VOC_data(:,2);
save OCV_params.mat;
%% Load data
clc; clear;
load ECM_params.mat;
load OCV_params.mat;
load OCVRRC.mat;
fs = 15;
clear VOC VOC_data;
%% Playground
%% Grid State and Preallocate
SOC_grid = (z_min:0.01:z_max)';
V1_min = 0;
V1_max = I_max*R_0;
V1_grid = (V1_min:0.01:V1_max)';

ns = [length(SOC_grid) length(V1_grid)];  % #states
N = (t_max-t_0)/dt; % #iterations

V = inf*ones(N,ns(1),ns(2)); % #value function
u_star = nan*ones(N-1,ns(1),ns(2));% #control
%% Solve DP
clc;
tic;
for i=1:ns(1)
    V(end,i,:) = (1-beta)*(SOC_grid(i)-z_target)^2; %terminal boundary condition
end

for k = (N-1):-1:1 %time
    if mod(k,10)==0
        fprintf(1,'Computing Principle of Optimality at %3.0f sec\n',k*dt);
    end
    for ii = 1:ns(1) %SOC
        for jj=1:ns(2) %V_1
            c_soc = SOC_grid(ii);
            c_v1 = V1_grid(jj);
            c_voc = voc(soc==round(c_soc,3)); %return voc when soc = c_soc 
                   
            % Bounds
            z_lb = C_batt/dt*(c_soc-z_max);
            z_ub = C_batt/dt*(c_soc-z_min);
            V_lb = (V_min-c_voc-c_v1)/R_0;
            V_ub = (V_max-c_voc-c_v1)/R_0;
            lb = max([I_min,z_lb,V_lb]);
            ub = min([I_max,z_ub,V_ub]);

            % Control grid
            I_grid = linspace(lb,ub,200)';
            
            % Cost-per-time-step
            %cv = ones(length(I_grid),1).*abs(c_voc-V_max) + I_grid.*R_0;
            %g_k = dt.*I_grid-cv;
            g_k = -beta*(c_soc-z_target)^2;

            % State dynamics
            SOC_nxt = c_soc+ dt/C_batt.*I_grid;
            V1_nxt = c_v1*(1-dt/(R_1*C_1))+dt/C_1.*I_grid;
            
            % Linear interpolation for value function
            V_nxt = interp2(SOC_grid,V1_grid,squeeze(V(k+1,:,:))',SOC_nxt,...
                V1_nxt,'linear');   
            [V(k,ii,jj), ind] = min(-g_k + V_nxt); %Bellman

            % Save Optimal Control
            u_star(k,ii,jj) = I_grid(ind);
        end
    end
end

solveTime = toc;
fprintf(1,'DP Solver Time %2.0f min %2.0f sec \n',floor(solveTime/60),mod(solveTime,60));

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
%% Outdated
a = I_sim;
b = SOC_sim;
c = V1_sim;
d = V_sim;
e = Voc_sim;
save OCVRRC.mat a b c d e
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
