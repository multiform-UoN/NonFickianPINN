%% multirate1.m
% This script solves the multirate model
% Matteo Icardi, 2023

writeData = 1

% %% Testcase 1
% datafolder = "../data/testcase1/";
% beta = 0.3;
% D = .1;
% V = 1;
% betak = 0.1;
% lambdak = -10;
% nt = 100;
% nx = 100;
% omega = 0; % BC coeff
% gamma = 0; % BC coeff


% %% Testcase 0 % Pure Diffusion
% datafolder = "../data/testcase0/";
% beta = 1;
% D = 1;
% V = 0;
% betak = 0;
% lambdak = 0;
% nt = 50;
% nx = 100;
% omega = 0; % BC coeff
% gamma = 0; % BC coeff

%% Testcase 4 % Advection Diffusion
% datafolder = "../data/testcase5/";
% beta = 1;
% D = 0.1;
% V = 1;
% betak = 0;
% lambdak = 0;
% nt = 50;
% nx = 50;
% omega = 0; % BC coeff
% gamma = 0; % BC coeff

%% Testcase 5 % Advection Diffusion (high Peclet)
% datafolder = "../data/testcase5/";
% beta = 1;
% D = 0.01;
% V = 1;
% betak = 0;
% lambdak = 0;
% nt = 50;
% nx = 100;
% omega = 0; % BC coeff
% gamma = 10; % BC coeff

%% Testcase 6 % Diffusion (time dependent BC)
datafolder = "../data/testcase6/";
beta = 1;
D = 1;
V = 0;
betak = 0;
lambdak = 0;
nt = 50;
nx = 100;
omega = 0; % BC coeff
gamma = 10; % BC coeff

% %% Testcase 2
% datafolder = "../data/testcase2/";
% beta = .5;
% D = 1;
% V = 1;
% betak = .5;
% lambdak = -100;
% nt = 10;
% nx = 10;
% omega = 0; % BC coeff
% gamma = 0; % BC coeff

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
v0 = 0.*x;
sol0 = [u0, v0]

%% Time dependent initial conditions
bc.left = @(t,u,v) [u-abs(cos(omega*2*pi*t))*exp(-gamma*t)];
bc.right = @(t,u,v) [diff(u)];

%% Make the right-hand side of the PDE.
pdefun = @(t,x,u,v) [ ...
                    (-diff(V.*u - D.*diff(u)) - betak.*lambdak.*(v-u))/beta; ...
                    lambdak.*(v-u) ...
                    ];
pdeflag = [1  1]; % Zero when a variable is indep of time.


%% Call pde15s to solve the problem.
opts = pdeset('Eps', 1e-6, 'PDEflag', pdeflag, 'Ylim', [0.5,1]);
[t, u, v] = pde15s(pdefun, t, sol0, bc, opts);

%% Plot the solution components.
figure(1)
colormap;
loglog(t,1-u(L),'DisplayName',num2str(lambdak))
legend
hold on
figure(2)
% waterfall(u, t)
plot(u(:),'k')
xlabel('x'), ylabel('t'), title('u,v')
hold on
% figure
% waterfall(v, t)
plot(v(:,1:10),'r')
% xlabel('x'), ylabel('t'), title('v')

%% extract data points
uu = u(xx');
vv = v(xx');

%% Write parameters to file
if writeData
    parameters = [beta; D; V; betak; lambdak];
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
    vv_file_name = fullfile(datafolder, 'c1.csv');
    dlmwrite(x_file_name, xx', 'delimiter', ',', 'precision', '%.6f');
    dlmwrite(t_file_name, t, 'delimiter', ',', 'precision', '%.6f');
    dlmwrite(uu_file_name, uu(:), 'delimiter', ',', 'precision', '%.6f');
    dlmwrite(vv_file_name, vv(:), 'delimiter', ',', 'precision', '%.6f');
end