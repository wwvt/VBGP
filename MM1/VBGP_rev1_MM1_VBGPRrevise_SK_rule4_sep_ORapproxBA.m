% SKOR approx BA: use 5000 points for calculating n. 
% yes, for 7 points to allocate, second last highest. 
% but for more points, the trend is not the same.
% SKOR paper did not provide a good budget allocation rule. WW edited.
% 7/26/2018 by WW

% Purpose: Use stochastic kriging to fit the response surface of expected 
%          waiting time in M/M/1 queue 
% Variable Definition:
%       X - Design points
%       Y - Simulation outputs at design points
%       Vhat - Intrinsic variance at design points
%       k - number of design points
%       K - number of prediction points
·
 clc; clear all; close all;

 s = RandStream('mt19937ar','Seed',0);
 RandStream.setGlobalStream(s);


%*************************************************************************
maxx = 2; minx = 1.1;   % range of utilization
arrival_rate = 1;       % fixed arrival rate
K = 1000;               % number of prediction points 
k = 7;      % 7, 20, 50, 100           % number of design points
runlength = 500;       % runlength at each design point
C = 2000;                % total computation budget
X = (minx:((maxx-minx)/(k-1)):maxx)';               % design points
truek = arrival_rate./(X.*(X-arrival_rate));        % analytic values at design points
Xpred = (minx:((maxx-minx)/(K-1)):maxx)';              % prediction points
y_pred_true = arrival_rate./(Xpred .* (Xpred - arrival_rate));   % analytic values at prediction points
 
% === >>> Obtain outputs at design points:
% Effort allocation is proportional to standard deviation
n_init = 1*ones(k,1);

q = 0;                          % degree of polynomial to fit in regression part(default)
B = repmat(X,[1 q+1]).^repmat(0:q,[k 1]);       % basis function matrix at design points
BK = repmat(Xpred,[1 q+1]).^repmat(0:q,[K 1]);     % basis function matrix at prediction points

tic;
for Mreps =1:115

[y_init Y_init(:,Mreps)  Vhat_init(:,Mreps)] = MM1sim_w(X,arrival_rate,n_init,runlength,'stationary');  
       %  ratio = sqrt(Vhat_init(:,Mreps));
Xpred5k = (minx:((maxx-minx)/(5000-1)):maxx)';              % for calculating W
y_pred_true_5k = arrival_rate./(Xpred5k .* (Xpred5k - arrival_rate));   % analytic values at prediction points
 
%---------------------VB parameter---------------------%

replication1(:,Mreps) = n_init;
A1 = getRepMatrix(replication1(:,Mreps));
[NMSE0, Ey0, Vy0, mutst0, diagSigmatst0, atst0, diagCtst0, LambdaTheta0, convergence0] = ... 
    vhgpr_ui_w(X, y_init, Xpred5k, y_pred_true_5k, A1, 10);
[n0,D] = size(X); 
ntheta1 = eval(feval('covSEardj'));
theta1 = LambdaTheta0(n0+1:n0+ntheta1);
SigmaM = feval('covSEardj',theta1, X);
[K1ss, Ks] = feval('covSEardj', theta1, X, Xpred5k);     % test covariance f

%---------------------VB parameter---------------------%
   
r = Ks;
m=k;
W=zeros(m,m);
for i=1:m
    for j=1:m
        W(i,j)=mean(r(i,:).*r(j,:));
    end
end

CC=diag(inv(SigmaM)*W*inv(SigmaM));
[NMSEtemp, Eytemp, Vhat_init0, mutsttemp, diagSigmatsttemp, atsttemp, diagCtsttemp, LambdaThetatemp, convergencetemp] = ... 
    vhgpr_ui_w(X, y_init, X, truek, A1, 10);
VC=sqrt(abs(Vhat_init0.*CC));
VC=VC/sum(VC);%function (29)
n_2nd(:,Mreps) = max(1,floor((C-sum(n_init))*VC));
 
[y_2nd Y_2nd(:,Mreps)  Vhat_2nd(:,Mreps)] = MM1sim_w(X,arrival_rate,n_2nd(:,Mreps),runlength,'stationary');  
      n(:,Mreps) = n_init + n_2nd(:,Mreps);
    %********************* Beginning of fitting ***********************        
      %Get MLEs of parameters based on K chosen design points   
       %input includes both response values and gradient estimates at
       %design points
      %    y0design = Y(:,Mrep);
        for i = 1:k
                y(i).n = [y_init(i).n, y_2nd(i).n];
                Vhat_all(i,Mreps) = var(y(i).n,0,2);
                Y(i,Mreps) = mean(y(i).n);
        end
        sel_indx = Vhat_all(:,Mreps)>0;
        Vhat(:,Mreps) = (Vhat_all(:,Mreps)).*sel_indx + Vhat_init(:,Mreps).*(1-sel_indx);

%---------------------VB predict---------------------%

replication(:,Mreps) = n(:,Mreps);
A = getRepMatrix(replication(:,Mreps));
[NMSE, Ey, Vy, mutst, diagSigmatst, atst, diagCtst, LambdaTheta, convergence] = ... 
    vhgpr_ui_w(X, y, Xpred, y_pred_true, A, 10);
LambdaThetas(:,Mreps) = LambdaTheta;
vby(:,Mreps) = Ey;
MSE_GPML_VB(Mreps) = 1/length(y_pred_true)*norm(vby(:,Mreps)-y_pred_true)^2; 

%---------------------VB predict---------------------%

end

toc;

save('VBGP_rev1_MM1_SK_SKORapproxBA_rule4_sep_C2000k7.mat');
