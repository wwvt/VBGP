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
k = 7;                 % number of design points
runlength = 1000;       % runlength at each design point
C = 2000;                % total computation budget
X = (minx:((maxx-minx)/(k-1)):maxx)';               % design points
truek = arrival_rate./(X.*(X-arrival_rate));        % analytic values at design points
Xpred = (minx:((maxx-minx)/(K-1)):maxx)';              % prediction points
y_pred_true = arrival_rate./(Xpred .* (Xpred - arrival_rate));   % analytic values at prediction points
 
% === >>> Obtain outputs at design points:
% Effort allocation is proportional to standard deviation
n_init = 4*ones(k,1);
rho = 1./X;
ratio = sqrt(4*rho./(1-rho).^4);
%n = ceil(C*ratio/sum(ratio));   % replications at each design point

q = 0;                          % degree of polynomial to fit in regression part(default)
B = repmat(X,[1 q+1]).^repmat(0:q,[k 1]);       % basis function matrix at design points
BK = repmat(Xpred,[1 q+1]).^repmat(0:q,[K 1]);     % basis function matrix at prediction points

%Tot_macroreps = 101;

%Mreps = 1;
tic;
%while Mreps  < Tot_macroreps 
%try
for Mreps = 1: 100
[y_init(:,Mreps) Y_init(:,Mreps)  Vhat_init(:,Mreps)] = MM1sim_w(X,arrival_rate,n_init,runlength,'stationary');  
         ratio = sqrt(Vhat_init(:,Mreps));
         n_2nd(:,Mreps) = ceil((C-sum(n_init))*ratio/sum(ratio));
[y_2nd(:,Mreps) Y_2nd(:,Mreps)  Vhat_2nd(:,Mreps)] = MM1sim_w(X,arrival_rate,n_2nd(:,Mreps),runlength,'stationary');  
      n(:,Mreps) = n_init + n_2nd(:,Mreps);

        for i = 1:k
                y(i,Mreps).n = [y_init(i,Mreps).n, y_2nd(i,Mreps).n];
                Vhat_all(i,Mreps) = var(y(i,Mreps).n,0,2);
                Y(i,Mreps) = mean(y(i,Mreps).n);
        end
        sel_indx = Vhat_all(:,Mreps)>0;
        Vhat(:,Mreps) = (Vhat_all(:,Mreps)).*sel_indx + Vhat_init(:,Mreps).*(1-sel_indx);

end

save('MM1_data_twostage_C2000k7.mat');
%%
clear;clc;
load('MM1_data_twostage_C500k100.mat');
for Mreps = 1:100
meanfunc = {'meanConst'};      hypgp.mean = [0]; hypsk.mean = [0];
covfuncsk = {'covSEard'}; hypsk.cov = log([1.2; 2]);% isotropic Gaussian % one hyperparameter   % ell,sf,sn
% for SK with white noise and sample variance
%covfuncsk = {'covSEiso'};
%hypsk.cov =log([4; 4]);
%covfunc = {'covSEiso'}; hypgp.cov = log([1.2; 2]);  % for SK with sample variance
covfunc = {'covSEard'}; hypgp.cov = log([1.2; 2]); % isotropic Gaussian % one hyperparameter   % ell,sf,sn
likfunc = {'likGauss'};  sn = 0.1; hypgp.lik = log(sn);            

%---------------------GP predict---------------------%
hyp2(:,Mreps) = minimize(hypgp, @gp, -100, @infGaussLik, meanfunc, covfunc, [], X, Y(:,Mreps)); 
%hyp2(:,Mreps) = hypgp;
%using infGaussLik original to estimate nlZ and so on
[m ns2] = gp(hyp2(:,Mreps), @infGaussLik, meanfunc, covfunc, [], X, Y(:,Mreps), Xpred);
gpy(:,Mreps) = m;
%---------------------GP predict---------------------%
MSE_GPML_GP(Mreps) = 1/length(y_pred_true)*norm(gpy(:,Mreps)-y_pred_true)^2; %fs2s

%---------------------SK predict---------------------%
%hyp3(:,Mreps) = minimize(hypsk, @gp_ww, -100, @infGaussLik_ww, meanfunc, covfuncsk, [], X, Y(:,Mreps),Vhat(:,Mreps)./n); 
%hyp3(:,Mreps) = minimize(hyp2(:,Mreps), @gp_ww, -100, @infGaussLik_ww, meanfunc, covfuncsk, [], X, Y(:,Mreps),Vhat(:,Mreps)./n); 
%hyp3 = minimize(hypsk, @gp, -100, @infGaussLik, meanfunc, covfuncsk, likfunc, X, Y(:,Mreps)); 
[ymu, ys2, fmu, fs2, lp, post] = gp_ww(hyp2(:,Mreps), @infGaussLik_ww, meanfunc, covfuncsk, [], X, Y(:,Mreps),Vhat(:,Mreps)./n(:,Mreps),Xpred);
sky(:,Mreps) = fmu;
%---------------------SK predict---------------------%
MSE_GPML_SK(Mreps) = 1/length(y_pred_true)*norm(sky(:,Mreps)-y_pred_true)^2; %fs2s

%---------------------VB predict---------------------%

replication(:,Mreps) = n(:,Mreps);
A = getRepMatrix(replication(:,Mreps));
[NMSE, Ey, Vy, mutst, diagSigmatst, atst, diagCtst, LambdaTheta, convergence] = ... 
    vhgpr_ui_w(X, y(:,Mreps), Xpred, y_pred_true, A, 10);
LambdaThetas(:,Mreps) = LambdaTheta;
vby(:,Mreps) = Ey;
MSE_GPML_VB(Mreps) = 1/length(y_pred_true)*norm(vby(:,Mreps)-y_pred_true)^2; 

%---------------------VB predict---------------------%

% if (MSE_GPML_GP(Mreps)>0.3) || (MSE_GPML_SK(Mreps)>0.3) || (MSE_GPML_VB(Mreps)>0.3)
% error('Error occurred.')
%  end
%  
%    Mreps = Mreps + 1;
% catch
% end

end

save('MM1_VB_SKfixgppar_GP_2stage_ninit4_C500k100_update.mat');
