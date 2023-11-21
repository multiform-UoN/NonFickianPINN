%% multirate.m -- an executable m-file for solving a partial differential equation
% Automatically created in CHEBGUI by user pmzmi1.
% Created on October 31, 2023 at 10:53.

%% Problem description.
% Solving
%   u_t = u" - v,
%   v" - u = 0,
% for x in [-1,1] and t in [0,2], subject to
%   u = 1, v = 1 at x = -1
% and
%   u = 1, v = 1 at x = 1

%% Problem set-up
L=1;
T = 1;
nt = 50;
nx = 100;

%% Construct a chebfun of the space variable on the domain and time vector
dom = [0 L];
xx = 0:L/nx:L;
t = 0:T/nt:T;
x = chebfun(@(x) x, dom);

%% Assign boundary conditions.
bc.left = @(t,u,v) [u-abs(cos(omega*2*pi*t))];
bc.right = @(t,u,v) [diff(u)];

% and of the initial conditions.
u0 = 1-tanh(x*10);
v0 = 0.*x;
sol0 = [u0, v0];

% %% Testcase 1
% datafolder = "../data/testcase1/";
% beta = 0.3;
% D = .1;
% V = 1;
% betak = 0.1;
% lambdak = -10;
% omega = 0;


%% Testcase 0
datafolder = "../data/testcase0/";
beta = 1;
D = 1;
V = 0;
betak = 0;
lambdak = 0;
omega = 0;



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
figure
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