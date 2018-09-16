% MM1 + GPML package with SK and GP
% Last Update: 11/13/2017 by Wenjing Wang

% Purpose: Use stochastic kriging to fit the response surface of expected 
%          waiting time in M/M/1 queue 
% Variable Definition:
%       X - Design points
%       Y - Simulation outputs at design points
%       Vhat - Intrinsic variance at design points
%       k - number of design points
%       K - number of prediction points

 clc; clear all; close all;

 s = RandStream('mt19937ar','Seed',0);
 RandStream.setGlobalStream(s);


%*************************************************************************
maxx = 2; minx = 1.1;   % range of utilization
arrival_rate = 1;       % fixed arrival rate
K = 1000;               % number of prediction points 
k = 500;                 % number of design points
runlength = 1000;       % runlength at each design point
C = 500;                % total computation budget
X = (minx:((maxx-minx)/(k-1)):maxx)';               % design points
truek = arrival_rate./(X.*(X-arrival_rate));        % analytic values at design points
Xpred = (minx:((maxx-minx)/(K-1)):maxx)';              % prediction points
y_pred_true = arrival_rate./(Xpred .* (Xpred - arrival_rate));   % analytic values at prediction points
 
% === >>> Obtain outputs at design points:
% Effort allocation is proportional to standard deviation
n_init = 1*ones(k,1);
rho = 1./X;
ratio = sqrt(4*rho./(1-rho).^4);
%n = ceil(C*ratio/sum(ratio));   % replications at each design point

q = 0;                          % degree of polynomial to fit in regression part(default)
B = repmat(X,[1 q+1]).^repmat(0:q,[k 1]);       % basis function matrix at design points
BK = repmat(Xpred,[1 q+1]).^repmat(0:q,[K 1]);     % basis function matrix at prediction points

Tot_macroreps = 1;

 
tic;
for Mreps  = 1:Tot_macroreps
 [y_init, Y_init(:,Mreps), Vhat_trash(:,Mreps)]  =  MM1sim_w(X,arrival_rate,n_init,runlength,'stationary');  
    A_init = getRepMatrix(n_init);
     ytemp = Y_init(:,Mreps);
    [NMSE_init, Ey_init, Vmean_init, mutst_init, diagSigmatst_init, atst_init, diagCtst_init, LambdaTheta_init, convergence_init] = ... 
        vhgpr_ui_w(X, y_init, X, ytemp, A_init, 10);
 

  %  vbvar_init(:,Mreps) = Vy_init;
        vbvar_init(:,Mreps) = Vmean_init;
        ratio = sqrt(vbvar_init(:,Mreps));
        n_2nd = ceil((C-sum(n_init))*ratio/sum(ratio));
        [y_2nd, Y_2nd(:,Mreps), Vhat_2nd(:,Mreps)]  =  MM1sim_w(X,arrival_rate,n_2nd,runlength,'stationary');  
        n = n_init + n_2nd;
 
for i = 1:k
                y(i).n = [y_init(i).n, y_2nd(i).n];
                Vhat_all(i,Mreps) = var(y(i).n,0,2);
                Y(i,Mreps) = mean(y(i).n);
end
        
%---------------------VB predict---------------------%
replication = n;
A = getRepMatrix(replication);

[NMSE, Ey, Vy, mutst, diagSigmatst, atst, diagCtst, LambdaTheta, convergence] = ... 
    vhgpr_ui_w(X, y, Xpred, y_pred_true, A, 10);
vby(:,Mreps) = Ey;
MSE_GPML_VB(:,Mreps) = 1/length(y_pred_true)*norm(vby(:,Mreps)-y_pred_true)^2; 

%---------------------VB predict---------------------%


 end
toc;
%%
save('MM1_VB_SK_GP_2stage_ninit1_C500_k500.mat');
