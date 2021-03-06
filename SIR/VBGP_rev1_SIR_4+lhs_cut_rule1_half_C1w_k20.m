% VBGP for ATO 
% equal allocation

clear;clc;
k=20; %k = 20, 50, 100, 200
C=10000; % C = 5000
K=2025;

s = RandStream('mt19937ar','Seed',100);
RandStream.setGlobalStream(s);

load('SIR_cut_k2k5_n1w_train.mat','y','S0','I0');
yyy=y.*800;
XK = [S0,I0];
S0scaled = (S0-1200)/500;
I0scaled = (I0-20)/180;
XKscaled = [S0scaled,I0scaled];
Ytrue=mean(yyy,2);
Ytruescaled = Ytrue/800;
Ytruescaled_scale = Ytrue;

clear y S0 I0 yyy S0scaled I0scaled;

n_init = floor(0.5*C/k)*ones(k,1);

LBSs = [1200,20];      % lower bounds for (S-s) and s
UBSs  = [1700,200];     % upper bounds for (S-s) and s

Mreps=1;
tic;
for Mreps = 1:100
%    try
        XX = lhsdesign(k-4,2,'criterion','maximin');
 
        delta = XX(:,1)*(UBSs(1)-LBSs(1))+LBSs(1); % Order up to point
        s = XX(:,2)*(UBSs(2)-LBSs(2))+LBSs(2); % Replenish point
         
        XX = [LBSs; UBSs; [LBSs(1),UBSs(2)];[UBSs(1),LBSs(2)];delta,s];

S0(:,Mreps) = XX(:,1); %[1200,1700]
I0(:,Mreps) = XX(:,2); % [20,200];

  [y_init(:,Mreps), Y_init(:,Mreps), Vhat_init(:,Mreps)]  = sirSimulate_M2k_w(S0(:,Mreps), I0(:,Mreps), n_init);
  
  ratio = sqrt(Vhat_init(:,Mreps));
  n_2nd(:,Mreps) = ceil((C-sum(n_init))*ratio/sum(ratio));
  
  [y_2nd(:,Mreps), Y_2nd(:,Mreps), Vhat_2nd(:,Mreps)]  =  sirSimulate_M2k_w(S0(:,Mreps), I0(:,Mreps), n_2nd(:,Mreps));
  n(:,Mreps) = n_init + n_2nd(:,Mreps);
       for i = 1:k
                y(i,Mreps).n = [y_init(i,Mreps).n, y_2nd(i,Mreps).n];
                Vhat_all(i,Mreps) = var(y(i,Mreps).n,0,2);
                Y(i,Mreps) = mean(y(i,Mreps).n);
        end
        sel_indx = Vhat_all(:,Mreps)>0;
        Vhat(:,Mreps) = (Vhat_all(:,Mreps)).*sel_indx + Vhat_init(:,Mreps).*(1-sel_indx);

%end

%save('VBGP_rev1_SIR_4+lhs_M2k_cut_data_rule1_C1wk20.mat');
% %%
% clear;clc;
%load('VBGP_rev1_SIR_lhs_cut_data_eq_C5000k20.mat');

%for Mreps=1:100
    
S0scaled(:,Mreps) = (S0(:,Mreps)-1200)/500;
I0scaled(:,Mreps) = (I0(:,Mreps)-20)/180;

Xdesign=  [S0scaled(:,Mreps),I0scaled(:,Mreps)];
ydesign(:,Mreps) = y(:,Mreps);
Ydesign(:,Mreps)= Y(:,Mreps);

meanfunc = {'meanConst'};      
hyp.mean = [2]; 
covfunc = {'covSEard'}; 
hyp.cov = [log(0.1);log(0.1);log(5)];
likfunc = {'likGauss'};  
sn = 0.1; hyp.lik = -2.132935872659505; 
%---------------------GP predict---------------------%
hyp2(:,Mreps) = minimize(hyp, @gp, -100, @infGaussLik, meanfunc, covfunc, likfunc, Xdesign, Ydesign(:,Mreps)); 
[m ns2] = gp(hyp2(:,Mreps), @infGaussLik, meanfunc, covfunc, [], Xdesign, Ydesign(:,Mreps), XKscaled);
gpy(:,Mreps) = m;
%---------------------SK predict---------------------%
hyp3(:,Mreps).mean = hyp2(:,Mreps).mean;
hyp3(:,Mreps).cov = hyp2(:,Mreps).cov;
[ymu, ys2, fmu, fs2, lp, post] = gp_ww(hyp3(:,Mreps), @infGaussLik_ww, meanfunc, covfunc, [], Xdesign, Ydesign(:,Mreps),Vhat(:,Mreps)./n(:,Mreps),XKscaled);
sky(:,Mreps) = fmu;
%---------------------VB predict---------------------%
replication(:,Mreps) = n(:,Mreps);
A = getRepMatrix(replication(:,Mreps));
[NMSE, Ey, Vy, mutst, diagSigmatst, atstdiagCtst, LambdaTheta, convergence] = ... 
   vhgpr_ui_w(Xdesign, ydesign(:,Mreps), XKscaled, Ytruescaled, A, 10);
LambdaThetas(:,Mreps) = LambdaTheta;
vby(:,Mreps) = Ey;
%---------------------VB orig predict---------------------%
[NMSE2, NLPD, Ey2, Vy2, mutst2, diagSigmatst2, atstdiagCtst2, LambdaTheta2, convergence] = ... 
   vhgpr_ui(Xdesign, Ydesign(:,Mreps), XKscaled, Ytruescaled, 100);
LambdaThetas(:,Mreps) = LambdaTheta2;
vb_orig(:,Mreps) = Ey2;


MSE_GP(Mreps) = 1/K*norm(gpy(:,Mreps)-Ytruescaled)^2;  %ns2
MSE_SK(Mreps) = 1/K*norm(sky(:,Mreps)-Ytruescaled)^2;  
MSE_VB(Mreps) = 1/K*norm(vby(:,Mreps)-Ytruescaled)^2;  
MSE_VBorig(Mreps) = 1/K*norm(vb_orig(:,Mreps)-Ytruescaled)^2;  

gpy2(:,Mreps) = gpy(:,Mreps)*800;
MSE_GP2(Mreps) = 1/K*norm(gpy2(:,Mreps)-Ytruescaled_scale)^2;  %ns2
sky2(:,Mreps) = sky(:,Mreps)*800;
MSE_SK2(Mreps) = 1/K*norm(sky2(:,Mreps)-Ytruescaled_scale)^2;  
vby2(:,Mreps) = vby(:,Mreps)*800;
MSE_VB2(Mreps) = 1/K*norm(vby2(:,Mreps)-Ytruescaled_scale)^2;  
vb_orig2(:,Mreps) = vb_orig(:,Mreps)*800;
MSE_VBorig2(Mreps) = 1/K*norm(vb_orig2(:,Mreps)-Ytruescaled_scale)^2;  
%catch
 %   continue
%end
end
    toc;


save('VBGP_rev1_SIR_4+lhs_M2k_cut_rule1_half_C1w_k20_SK.mat');
