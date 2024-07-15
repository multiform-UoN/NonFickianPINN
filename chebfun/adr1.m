%% multirate1.m
% This script solves a simple advection-diffusion-reaction problem
% Matteo Icardi, 2023

set(0,'DefaultFigureWindowStyle','docked')
writeData = 1;
col = 'r'

datafolder = "../data/adr/";
beta = 1;
D = 1e-1;
V = 1;

%% time dependent bc constants
omega = 0; % BC coeff
gamma = 0; % BC coeff

%% reaction constants
sigma = 1;
forward_rate = 3;
backward_rate = 3;
sigma2 = 1e-2;
reaction_model = "mm";

if reaction_model == "mm" %michaelis-menten
    reaction = @(u) sigma*u.^forward_rate./(sigma2 + u.^backward_rate);
elseif reaction_model == "linear"
    reaction = @(u) sigma*u;
elseif reaction_model == "quadratic"
    reaction = @(u) sigma*u.^2;
elseif reaction_model == "polynomial"
    reaction = @(u) sigma*u.^forward_rate.*(1-u).^backward_rate;
end


% sampling points
nt = 50;
nx = 100;

%% Problem set-up
L = 1;
T = 1;

%% Construct a chebfun of the space variable on the domain and time vector
dom = [0 L];
xx = 0:L/nx:L;
t = 0:T/nt:T;
x = chebfun(@(x) x, dom);

%% initial conditions.
u0 = 1-tanh(x*10);
sol0 = [u0];

%% Time dependent initial conditions
bc.left = @(t,u) [u-abs(cos(omega*2*pi*t))*exp(-gamma*t)];
bc.right = @(t,u) [diff(u)];

%% Make the right-hand side of the PDE.
pdefun = @(t,x,u) [ ...
                    (-diff(V.*u - D.*diff(u)) - reaction(u))/beta; ...
                    ];
pdeflag = [1]; % Zero when a variable is indep of time.


%% Call pde15s to solve the problem.
opts = pdeset('Eps', 1e-6, 'PDEflag', pdeflag, 'Ylim', [0.5,1]);
[t, u] = pde15s(pdefun, t, sol0, bc, opts);

%% Plot the solution components.
figure(1)
hold on
loglog(t,1-u(L),col,'DisplayName',num2str(sigma))
legend
hold on
figure(2)
hold on
plot(u(:),col)
xlabel('x'), ylabel('t'), title('u')
hold on
% figure
% waterfall(v, t)
% plot(v(:,1:10),'r')
% xlabel('x'), ylabel('t'), title('v')

%% extract data points
uu = u(xx');
%vv = v(xx');

%% Write parameters to file
if writeData
    parameters = [beta; D; V; sigma; sigma2; forward_rate; backward_rate];
    filename = fullfile(datafolder, 'p.csv');
    % Create the folder if it does not exist
    if ~exist(datafolder, 'dir')
        mkdir(datafolder);
    end
    dlmwrite(filename, parameters, 'delimiter', ',', 'precision', '%.6f');
    
    %% Write solution vectors to CSV
    x_file_name = fullfile(datafolder, 'x.csv');
    t_file_name = fullfile(datafolder, 't.csv');
    uu_file_name = fullfile(datafolder, 'c.csv');
    %vv_file_name = fullfile(datafolder, 'c1.csv');
    dlmwrite(x_file_name, xx', 'delimiter', ',', 'precision', '%.6f');
    dlmwrite(t_file_name, t, 'delimiter', ',', 'precision', '%.6f');
    dlmwrite(uu_file_name, uu(:), 'delimiter', ',', 'precision', '%.6f');
    %dlmwrite(vv_file_name, vv(:), 'delimiter', ',', 'precision', '%.6f');
end