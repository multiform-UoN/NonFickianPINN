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
% Create an interval of the space domain...
L=1;
dom = [0 L];
%...and specify a sampling of the time domain:
T = 1;
t = 0:T/100:T;

K = 10;

%% Testcase 1
datafolder = "../data/testcase1/";
beta = 0.3;
D = .1;
V = 1;
betak = 0.1;
lambdak = -10;

omega = 0;


%% Write parameters to file
parameters = [beta; D; V; betak; lambdak];
filename = fullfile(datafolder, 'p.csv');
dlmwrite(filename, parameters, 'delimiter', ',', 'precision', '%.6f');

%% Make the right-hand side of the PDE.
pdefun = @(t,x,u,v) [ ...
                    (-diff(V.*u - D.*diff(u)) - betak.*lambdak.*(v-u))/beta; ...
                    lambdak.*(v-u) ...
                    ];
pdeflag = [1  1]; % Zero when a variable is indep of time.

%% Assign boundary conditions.
bc.left = @(t,u,v) [u-abs(cos(omega*2*pi*t))];
bc.right = @(t,u,v) [diff(u)];

%% Construct a chebfun of the space variable on the domain,
x = chebfun(@(x) x, dom);
% and of the initial conditions.
u0 = 1-tanh(x*10);
v0 = 0.*x;
sol0 = [u0, v0];

%% Setup preferences for solving the problem.
opts = pdeset('Eps', 1e-6, 'PDEflag', pdeflag, 'Ylim', [0.5,1]);

%% Call pde15s to solve the problem.
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

%% extract data
xx = 0:L/100:L;
uu = u(xx');
vv = v(xx');
% Specify file names
x_file_name = 'x.csv';
t_file_name = 't.csv';
uu_file_name = 'c.csv';
vv_file_name = 'c1.csv';

% Write vector x to CSV
dlmwrite(x_file_name, xx', 'delimiter', ',', 'precision', '%.6f');

% Write vector t to CSV
dlmwrite(t_file_name, t, 'delimiter', ',', 'precision', '%.6f');

% Reshape matrices uu and vv into single columns
uu_column = uu(:);
vv_column = vv(:);

% Write matrix uu to CSV
dlmwrite(uu_file_name, uu_column, 'delimiter', ',', 'precision', '%.6f');

% Write matrix vv to CSV
dlmwrite(vv_file_name, vv_column, 'delimiter', ',', 'precision', '%.6f');