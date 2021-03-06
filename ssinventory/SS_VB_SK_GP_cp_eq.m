%% (S,s)Inventory Problem + GPML package with SK and GP
% (S,s) Inventory Model
% Last Update: 9/20/2017 by ww
clear;
clc;
%rng('shuffle');

 s = RandStream('mt19937ar','Seed',0);
 RandStream.setGlobalStream(s);

%*************************************************************************
Macroreps = 100;  % Number of macroreplications 
k = 20;    % Number of design points
T = 1000;  % Number of design points of the finest grid
C = 2000;                % total computation budget
n = ones(k,1)*ceil(C*1/k); 
%Xw = lhsdesign(K,2);  % design points
%rho = Xw(:,1).*Xw(:,1) + Xw(:,2).*Xw(:,2) + 1;
%n = ceil(C*rho/sum(rho)); 
%n = 100;  % Sample size of intrinsic random error for each macro-replication
CRN = 0; % choose to use CRN
%flagLHS  = 1; %whether LHS design or grid design is used

%%Parameters % Can be adjusted to see the surface and variability
LBSs = [10,10];      % lower bounds for (S-s) and s
UBSs  = [40,50];     % upper bounds for (S-s) and s
% LBSs = [100,100];      % lower bounds for (S-s) and s
% UBSs  = [300,350];     % upper bounds for (S-s) and s


% Korder = 5;  % set-up cast for placing an order
% pbklog = 0.5;   % unit backlog cost
% meanD = 20;   % mean of demand
% c = 0.05;         % per-unit ordering cost
% h = 0.05;         % holding costs per unit inventory
Scount =zeros(Macroreps,1); %count the number of matrix singularity problems occurred

%Parameters
Korder = 100;  % set-up cast for placing an order
pbklog = 10;   % unit backlog cost
meanD = 20;   % mean of demand
c = 5;         % per-unit ordering cost
h = 1;         % holding costs per unit inventory


tic;
%>>>>>>>>>>>>>>>>>>>Full Deisgn Space of [LBS,UBS]*[LBs,UBs] for Prediction<<<<<<<<<<<<<<<<<<<<<<
Nx = 50; %total number of prediction points
Xtemp = [LBSs(1):(UBSs(1)-LBSs(1))/(Nx-1):UBSs(1);...
         LBSs(2):(UBSs(2)-LBSs(2))/(Nx-1):UBSs(2)]'; 
Xpred = zeros(Nx^2,2);  % 2500 prediction points in grid
%Xpred(:,1) = lhsdesign(2500,1)*30+10;              % prediction points
%Xpred(:,2) = lhsdesign(2500,1)*40+10;              % prediction points


for i = 1:Nx
    Xpred((i-1)*Nx+1:i*Nx,1)= repmat(Xtemp(i,1),Nx,1);
end
    Xpred(:,2) = repmat(Xtemp(:,2),Nx,1);

y_pred_true  = TrueYSs(Xpred,Korder,pbklog,meanD,c,h);
%Update the number of check points in the two-dimensional design space

 %Get the B and X that are used for prediction
Ntotal = length(Xpred); %total number of prediction points
Bpred = ones(Ntotal,1);


 %>>>>>>>>>>>>>>>>>>>>>>>Begin the design points loop<<<<<<<<<<<<<<<<<<<<<<
   
 % for i=1:size(K,2)

 for Mreps=1:115
        %--------------------Latin-hyper-cube sampling design points--------------       
        XX = lhsdesign(k-4,2,'criterion','maximin');
        delta = XX(:,1)*(UBSs(1)-LBSs(1))+LBSs(1); % Order up to point
        s = XX(:,2)*(UBSs(2)-LBSs(2))+LBSs(2); % Replenish point
         
        XX = [LBSs; UBSs; [LBSs(1),UBSs(2)];[UBSs(1),LBSs(2)];delta,s];
        X(:,:,Mreps) = XX;


        Ndesign =  size(X(:,:,Mreps),1);  % Number of design points in the deisgn Space of 
                    % [LBS,UBS]*[LBs,UBs] used in LHS;
                    
            %Get(K x b) matrix of basis functions at K chosen design points
            %that are used in parameter estimation
            B = [ones(length(X(:,:,Mreps)),1)];
    [y(:,Mreps), Y(:,Mreps), Vhat(:,Mreps)]  =  SsInventoryData_w(Korder,pbklog,meanD,c,h,T,n,X(:,:,Mreps),CRN);
   
 end
 
 save('SSinv_data_eq_C2000k20.mat');
 
%  clear;clc;
%  load('SSinv_data_eq_C2000k200.mat');

 for Mreps=1:115
%Mreps=1;
meanfunc = {'meanConst'};      hypgp.mean = [10]; hypsk.mean = [10]; %hypmcmc.mean = [0];
covfuncsk = {'covSEard'}; hypsk.cov = log([1.2; 1.2;2]);%hypmcmc.cov = [0; 0; -Inf];  % isotropic Gaussian % one hyperparameter   % ell,sf,sn
% for SK with white noise and sample variance
%covfuncsk = {'covSEiso'};
%hypsk.cov =log([4; 4]);
%covfunc = {'covSEiso'}; hypgp.cov = log([1.2; 2]);  % for SK with sample variance
covfunc = {'covSEard'}; hypgp.cov = log([1.2; 1.2;2]);%hypmcmc.cov = [0; 0; -Inf];  % isotropic Gaussian % one hyperparameter   % ell,sf,sn
likfunc = {'likGauss'};  sn = 0.1; hypgp.lik = log(sn);%hypsk.lik = log(sn); %hyp.lik  = Vhat_init(:,Mreps)./n;% 0.049859457950193;%                 
%likfuncMCMC = {'likT'};  hypmcmc.lik = [log(4-1); log(0.01)];%log(0.01); %hyp.lik  = Vhat_init(:,Mreps)./n;% 0.049859457950193;%                 
%likfuncMCMC = {'likLaplace'}; hypmcmc.lik = log(0.1);
%lr6 = {'likGumbel','+'}; hypr6 = log(sn);
% Ncg = 50;                                   % number of conjugate gradient steps

%---------------------GP predict---------------------%
hyp2(:,Mreps) = minimize(hypgp, @gp, -100, @infGaussLik, meanfunc, covfunc, [], X(:,:,Mreps), Y(:,Mreps)); 
%using infGaussLik original to estimate nlZ and so on
[m ns2] = gp(hyp2(:,Mreps), @infGaussLik, meanfunc, covfunc, [], X(:,:,Mreps), Y(:,Mreps), Xpred);
gpy(:,Mreps) = m;
%---------------------GP predict---------------------%
MSE_GPML_GP(:,Mreps) = 1/length(y_pred_true)*norm(gpy(:,Mreps)-y_pred_true)^2; %fs2s

%---------------------SK predict---------------------%
hyp3(:,Mreps) = minimize(hypsk, @gp_ww, -100, @infGaussLik_ww, meanfunc, covfuncsk, [], X(:,:,Mreps), Y(:,Mreps),Vhat(:,Mreps)./n); 
%hyp3 = minimize(hypsk, @gp, -100, @infGaussLik, meanfunc, covfuncsk, likfunc, X, Y(:,Mreps)); 
[ymu, ys2, fmu, fs2, lp, post] = gp_ww(hyp3(:,Mreps), @infGaussLik_ww, meanfunc, covfuncsk, [], X(:,:,Mreps), Y(:,Mreps),Vhat(:,Mreps)./n,Xpred);
sky(:,Mreps) = fmu;
%---------------------SK predict---------------------%
MSE_GPML_SK(:,Mreps) = 1/length(y_pred_true)*norm(sky(:,Mreps)-y_pred_true)^2; %fs2s

%---------------------VB predict---------------------%
%replication = ones(K,1)*ceil(C*1/K);
%replication = [99, 92 69 80 69];
replication = n;

%V = Vhat'./replication;
A = getRepMatrix(replication);

[NMSE, Ey, Vy, mutst, diagSigmatst, atst, diagCtst, LambdaTheta, convergence] = ... 
    vhgpr_ui_w(X(:,:,Mreps), y(:,Mreps), Xpred, y_pred_true, A, 10);
% 
% [NMSE, Ey2, Vy, mutst, diagSigmatst, atst, diagCtst, LambdaTheta, convergence] = ... 
%     vhgpr_ui_w_test(X, y, Xpred, y_pred_true, A, 10);
LambdaThetas(:,Mreps) = LambdaTheta;
vby(:,Mreps) = Ey;
%vby2(:,Mreps) = Ey2;

MSE_GPML_VB(:,Mreps) = 1/length(y_pred_true)*norm(vby(:,Mreps)-y_pred_true)^2; 
%MSE_GPML_VB2(:,Mreps) = 1/length(y_pred_true)*norm(vby2(:,Mreps)-y_pred_true)^2; 

%---------------------VB predict---------------------%


 end

save('SS_VB_SK_GP_eq_k20_update_M115.mat');

