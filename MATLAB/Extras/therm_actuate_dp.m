%% Optimal Charging with Thermal Actuation
%   Created May 29, 2018

% therm_actuate_dp.m
clear;
fs = 16;

%% Problem Data

% Electrical parameters
Q = 3600;       % [Ah * 3600 sec/hr]
R = 8.314;      % [J/mol*K]

R0_ref = 0.050; % [Ohms]
E_R0 = 30e3;    % [J/mol]

Rc = 0.005;     % [Ohms]
Cc = 500;       % [Farads]

z_des = 0.95;   % [-]

% OCV polynomial coefficients
p_0 = 3.4707;
p_1 = 1.6112;
p_2 = -2.6287;
p_3 = 1.7175;

% Thermal parameters
h = 1/50;       % [W/K]
Tinf = 25;      % [deg C]
T_ref = 25+273.15;     % [Kelvin]

% Aging params
p3a =      -47.84;
p2a =        1215;
p1a =       -9419;
p0a =   3.604e+04;

% Time Step
dt = 5;         % [sec]

% State/control limits
z_low = 0.1;
z_high = 1.0;

V_c_low = -0.1;
V_c_high = 0.1;

V_T_low = 3.6;
V_T_high = 4.2;

T_low = 0;
T_high = 45;

I_low = -10;
I_high = 10;

u_low = -1;
u_high = 1;

% Utility function weights
alpha1 = 1e-4;
alpha2 = 1e3;
alpha3 = 1;

% Time Horizon
N = 100;


%% Pre-allocate

% Vector of States
z_vec = z_low:0.01:z_high;
T_vec = T_low:1:T_high;

nz = length(z_vec);
nT = length(T_vec);

% Vector of Controls
nu = 25;
I_vec = linspace(I_low,I_high,nu)';
u_vec = linspace(u_low,u_high,nu)';

% Generate matrix of control actions
control_mat = nan*ones(nu^2,2);

cnt = 1;
for idx1 = 1:nu
    for idx2 = 1:nu
            
        control_mat(cnt,:) = [I_vec(idx1), u_vec(idx2)];
        cnt = cnt+1;
            
    end
end

% Preallocate Value Function to INFINITY 
% "inf" initial label is flag to say "not-computed yet")
W = inf*ones(N,nz,nT);

% Preallocate control policy to NaN 
% "nan" initial label is flag to say "not-computed yet")
I_star = nan*ones(N-1,nz,nT);
u_star = nan*ones(N-1,nz,nT);

% Boundary Condition
for idx_z = 1:nz
    W(end,idx_z,:) = alpha3*(z_vec(idx_z) - z_des)^2;
end

%% Solve DP Equations
tic;

% Iterate through time backwards
for k = (N-1):-1:1;
    
    fprintf(1,'Computing Principle of Optimality at %3.0f sec\n',k*dt);
    
    % Iterate through states
    for iz = 1:nz
        for iT = 1:nT
            
            % parse out vectors of control actions
            I_test = control_mat(:,1);
            u_test = control_mat(:,2);

            % test all the next states
            z_nxt_test = z_vec(iz) + dt/Q*I_test;
            
            OCV = p_0 + p_1*z_vec(iz) + p_2*z_vec(iz)^2 + p_3*z_vec(iz)^3;
            R0 = R0_ref*exp(E_R0/R*(1/(T_vec(iT)+273.15) - 1/T_ref));
            V_T_test = OCV + R0*I_test;
            
            Q_dot_test = abs(I_test.*(V_T_test - OCV));
            T_nxt_test = T_vec(iT) + dt*( h*(Tinf - T_vec(iT)) + Q_dot_test + u_test );

            % keep only the actions which maintain the state in feasible domain
            ind = find( (z_nxt_test >= z_low) & (z_nxt_test <= z_high) ...
                        & (V_T_test >= V_T_low) & (V_T_test <= V_T_high) ...
                        & (T_nxt_test >= T_low) & (T_nxt_test <= T_high));

            % Select admissible actions
            I_admis = control_mat(ind,1);
            u_admis = control_mat(ind,2);
            
            % Cost terms
            J1 = alpha1*u_admis.^2;
            
            M = p0a + p1a*I_admis + p2a*I_admis.^2 + p3a*I_admis.^3;
            Ea = 31700 - 370.3*I_admis;
            A_tol = (20./(M.*exp(-Ea./(R*(T_vec(iT)+273.15))))).^(1/0.55);
            J2 = alpha2*abs(I_admis)*dt ./ (7200*A_tol);

            % Next State, z_{k+1}, T_{k+1}
            z_nxt = z_vec(iz) + dt/Q*I_admis;
            V_T = OCV + R0*I_admis;
            
            Q_dot = abs(I_admis.*(V_T - OCV));
            T_nxt = T_vec(iT) + dt*( h*(Tinf - T_vec(iT)) + Q_dot + u_admis );

            % Compute value function at nxt time step (need to interpolate)
            W_nxt = interp2(z_vec,T_vec,squeeze(W(k+1,:,:))',...
                            z_nxt,T_nxt,'linear');

            % Principle of Optimality (aka Bellman's equation)
            [W(k,iz,iT),ind] = min(J1 + J2 + W_nxt);

            % Save minimizing control action
            I_star(k,iz,iT) = I_admis(ind);
            u_star(k,iz,iT) = u_admis(ind);
            
        end
    end
end

solveTime = toc;
fprintf(1,'DP Solver Time %2.0f min %2.0f sec \n',floor(solveTime/60),mod(solveTime,60));

%% Plot Optimal Control Law & Value Function



%% Simulate Optimal Behavior

% Set Initial State
z0 = 0.1;
T0 = 3;
V_c0 = 0;

% Preallocate state trajectory
% nan is flag to indicate "not-calculated"
z_sim = nan*ones(N,1);
z_sim(1) = z0;
V_c_sim = nan*ones(N,1);
V_c_sim(1) = V_c0;
V_T_sim = nan*ones(N-1,1);

T_sim = nan*ones(N,1);
T_sim(1) = T0;
Q_dot_sim = nan*ones(N-1,1);

% Preallocate control trajectory
% nan is flag to indicate "not-calculated"
I_sim = nan*ones(N-1,1);
u_sim = nan*ones(N-1,1);

% Preallocate cumulative cost
J1_sim = inf*ones(N-1,1);
J1_sim(1) = 0;
J2_sim = J1_sim;

% Iterate through time
tic;
for k = 1:(N-1)
    
    fprintf(1,'Simulating at %3.0f sec\n',k*dt);
    
    % Get optimal actions, given time step and state
    I_sim(k) = interp2(z_vec,T_vec,squeeze(I_star(k,:,:))',...
                z_sim(k),T_sim(k),'linear');
    u_sim(k) = interp2(z_vec,T_vec,squeeze(u_star(k,:,:))',...
                z_sim(k),T_sim(k),'linear');
    
    % Dynamics
    z_sim(k+1) = z_sim(k) + dt/Q*I_sim(k);
    V_c_sim(k+1) = (1 - dt/(Rc*Cc))*V_c_sim(k) + dt/Cc*I_sim(k);
    OCV = p_0 + p_1*z_sim(k) + p_2*z_sim(k)^2 + p_3*z_sim(k)^3;
    R0 = R0_ref*exp(E_R0/R*(1/(T_sim(k)+273.15) - 1/T_ref));
    V_T_sim(k) = OCV + R0*I_sim(k);
    
    Q_dot_sim(k) = abs(I_sim(k).*(V_T_sim(k) - OCV));
    T_sim(k+1) = T_sim(k) + dt*(h*(Tinf - T_sim(k)) + Q_dot_sim(k) + u_sim(k));
    
    % Cumulative cost
    J1 = alpha1*u_sim(k)^2;
    J1_sim(k+1) = J1_sim(k) + J1;
    
    
    M = p0a + p1a*I_sim(k) + p2a*I_sim(k)^2 + p3a*I_sim(k)^3;
    Ea = 31700 - 370.3*I_sim(k);
    A_tol = (20./(M.*exp(-Ea./(R*(T_sim(k)+273.15))))).^(1/0.55);
    J2 = alpha2*abs(I_sim(k))*dt ./ (7200*A_tol);
    J2_sim(k+1) = J2_sim(k) + J2;
    
end
solveTime = toc;
fprintf(1,'Final SOC : %0.2f \n',z_sim(N));
fprintf(1,'Simulation Time %2.0f min %2.0f sec \n',floor(solveTime/60),mod(solveTime,60));

%% Plot Optimal Sequence
figure(2); clf; cla;

subplot(4,1,1)
plot(1:N,z_sim,'LineWidth',2); hold on;
plot([1, N],[z_high, z_high],'k--','LineWidth',2);
set(gca,'FontSize',fs);
legend({'$$z_k$$'},'interpreter','latex','fontsize',fs);

subplot(4,1,2)
plot(1:N,T_sim,'LineWidth',2); hold on;
plot([1, N],[T_high, T_high],'k--','LineWidth',2);
set(gca,'FontSize',fs);
legend({'$$T_k$$'},'interpreter','latex','fontsize',fs);

subplot(4,1,3)
plot(1:(N-1),V_T_sim,'LineWidth',2); hold on;
plot([1, N],[V_T_high, V_T_high],'k--','LineWidth',2);
set(gca,'FontSize',fs);
legend({'$$V_{T,k}$$'},'interpreter','latex','fontsize',fs);

subplot(4,1,4)
plot(1:(N-1),I_sim,'b-',1:(N-1),u_sim*10,'r--','LineWidth',2); hold on;
set(gca,'FontSize',fs);
legend({'$$I_k$$';'$$u_k$$'},'interpreter','latex','fontsize',fs);
xlabel('Time Period','FontSize',fs)


