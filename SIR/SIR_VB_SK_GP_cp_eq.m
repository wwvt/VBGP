%% (S,s)Inventory Problem + GPML package with SK and GP
% (S,s) Inventory Model
% Created: 9/20/2017 by ww
% Last Update: 7/20/2018 by ww

clear;
clc;
%rng('shuffle');

 s = RandStream('mt19937ar','Seed',0);
 RandStream.setGlobalStream(s);

%*************************************************************************
Macroreps = 1;  % Number of macroreplications 
k = 50;    % Number of design points
T = 1000;  % Number of design points of the finest grid
C = 50;                % total computation budget
n = ones(k,1)*ceil(C*1/k); 

tic;
Nx = 50; %total number of prediction points
Xpred = zeros(Nx^2,2);  % 2500 prediction points in grid
%Xpred(:,1) = lhsdesign(2500,1)*30+10;              % prediction points
%Xpred(:,2) = lhsdesign(2500,1)*40+10;              % prediction points

 for Mreps=1:1
        %--------------------Latin-hyper-cube sampling design points--------------       
        X = lhsdesign(k,2,'criterion','maximin');
        y  =  sirSimulate_w(X);
 end
 %%

meanfunc = {'meanConst'};      hypgp.mean = [10]; hypsk.mean = [10]; %hypmcmc.mean = [0];
covfuncsk = {'covSEard'}; hypsk.cov = log([1.2; 1.2;2]);%hypmcmc.cov = [0; 0; -Inf];  % isotropic Gaussian % one hyperparameter   % ell,sf,sn
covfunc = {'covSEard'}; hypgp.cov = log([1.2; 1.2;2]);%hypmcmc.cov = [0; 0; -Inf];  % isotropic Gaussian % one hyperparameter   % ell,sf,sn
likfunc = {'likGauss'};  sn = 0.1; hypgp.lik = log(sn);%hypsk.lik = log(sn); %hyp.lik  = Vhat_init(:,Mreps)./n;% 0.049859457950193;%                 
%---------------------GP predict---------------------%
hyp2(:,Mreps) = minimize(hypgp, @gp, -100, @infGaussLik, meanfunc, covfunc, [], X(:,:,Mreps), Y(:,Mreps)); 
[m ns2] = gp(hyp2(:,Mreps), @infGaussLik, meanfunc, covfunc, [], X(:,:,Mreps), Y(:,Mreps), Xpred);
gpy(:,Mreps) = m;
%---------------------VB predict---------------------%
y_pred_true = ones(length(Xpred),1);
[NMSE, NLPD, Ey, Vy, mutst, diagSigmatst, atst, diagCtst, LambdaTheta, convergence] = ... 
    vhgpr_ui(X(:,:,Mreps), Y(:,Mreps), Xpred, y_pred_true,  10);

%%
save('SIR_VB_SK_GP_eq_k20_update_M115.mat');
%%
plot(X,Y)


