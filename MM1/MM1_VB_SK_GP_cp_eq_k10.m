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
k = 10;                 % number of design points
runlength = 1000;       % runlength at each design point
C = 500;                % total computation budget
X = (minx:((maxx-minx)/(k-1)):maxx)';               % design points
truek = arrival_rate./(X.*(X-arrival_rate));        % analytic values at design points
Xpred = (minx:((maxx-minx)/(K-1)):maxx)';              % prediction points
y_pred_true = arrival_rate./(Xpred .* (Xpred - arrival_rate));   % analytic values at prediction points
 
% === >>> Obtain outputs at design points:
% Effort allocation is proportional to standard deviation
n = ones(k,1)*ceil(C*1/k); 
rho = 1./X;
ratio = sqrt(4*rho./(1-rho).^4);
%n = ceil(C*ratio/sum(ratio));   % replications at each design point

q = 0;                          % degree of polynomial to fit in regression part(default)
B = repmat(X,[1 q+1]).^repmat(0:q,[k 1]);       % basis function matrix at design points
BK = repmat(Xpred,[1 q+1]).^repmat(0:q,[K 1]);     % basis function matrix at prediction points

Tot_macroreps = 1;

Mreps = 1;
tic;
while Mreps < Tot_macroreps +1
try
[y Y(:,Mreps)  Vhat(:,Mreps)] = MM1sim_w(X,arrival_rate,n,runlength,'stationary');  
% simulate M/M/1 queue %Vhat(m) = var(waits(m).n,0,2);
meanfunc = {'meanConst'};      hypgp.mean = [0]; hypsk.mean = [0];
covfuncsk = {'covSEard'}; hypsk.cov = log([1.2; 2]);% isotropic Gaussian % one hyperparameter   % ell,sf,sn
% for SK with white noise and sample variance
%covfuncsk = {'covSEiso'};
%hypsk.cov =log([4; 4]);
%covfunc = {'covSEiso'}; hypgp.cov = log([1.2; 2]);  % for SK with sample variance
covfunc = {'covSEard'}; hypgp.cov = log([1.2; 2]); % isotropic Gaussian % one hyperparameter   % ell,sf,sn
likfunc = {'likGauss'};  sn = 0.1; hypgp.lik = log(sn);            

%---------------------GP predict---------------------%
hyp2 = minimize(hypgp, @gp, -100, @infGaussLik, meanfunc, covfunc, [], X, Y(:,Mreps)); 
%using infGaussLik original to estimate nlZ and so on
[m ns2] = gp(hyp2, @infGaussLik, meanfunc, covfunc, [], X, Y(:,Mreps), Xpred);
gpy(:,Mreps) = m;
%---------------------GP predict---------------------%
MSE_GPML_GP(Mreps) = 1/length(y_pred_true)*norm(gpy(:,Mreps)-y_pred_true)^2; %fs2s

%---------------------SK predict---------------------%
hyp3(:,Mreps) = minimize(hypsk, @gp_ww, -100, @infGaussLik_ww, meanfunc, covfuncsk, [], X, Y(:,Mreps),Vhat(:,Mreps)./n); 
%hyp3 = minimize(hypsk, @gp, -100, @infGaussLik, meanfunc, covfuncsk, likfunc, X, Y(:,Mreps)); 
[ymu, ys2, fmu, fs2, lp, post] = gp_ww(hyp3(:,Mreps), @infGaussLik_ww, meanfunc, covfuncsk, [], X, Y(:,Mreps),Vhat(:,Mreps)./n,Xpred);
sky(:,Mreps) = fmu;
%---------------------SK predict---------------------%
MSE_GPML_SK(Mreps) = 1/length(y_pred_true)*norm(sky(:,Mreps)-y_pred_true)^2; %fs2s

%---------------------VB predict---------------------%

replication = n;

%V = Vhat'./replication;
A = getRepMatrix(replication);

[NMSE, Ey, Vy, mutst, diagSigmatst, atst, diagCtst, LambdaTheta, convergence] = ... 
    vhgpr_ui_w(X, y, Xpred, y_pred_true, A, 10);
% 
% [NMSE, Ey2, Vy, mutst, diagSigmatst, atst, diagCtst, LambdaTheta, convergence] = ... 
%     vhgpr_ui_w_test(X, y, Xpred, y_pred_true, A, 10);
LambdaThetas(:,Mreps) = LambdaTheta;
vby(:,Mreps) = Ey;
%vby2(:,Mreps) = Ey2;

MSE_GPML_VB(Mreps) = 1/length(y_pred_true)*norm(vby(:,Mreps)-y_pred_true)^2; 
%MSE_GPML_VB2(:,Mreps) = 1/length(y_pred_true)*norm(vby2(:,Mreps)-y_pred_true)^2; 

%---------------------VB predict---------------------%

% if (MSE_GPML_GP(Mreps)>0.3) || (MSE_GPML_SK(Mreps)>0.3) || (MSE_GPML_VB(Mreps)>0.3)
% error('Error occurred.')
%  end
%  
 catch
end
Mreps = Mreps + 1;


 end
toc;
%%
save('MM1_VB_SK_GP_cp_cov_eq_C2kk50_skip_update.mat');

%--------------------------------------------------------------------------
