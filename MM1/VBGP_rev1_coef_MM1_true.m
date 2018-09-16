%% for MM1 coef of variation
% MM1 true
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
% === >>> Generate evenly distributed design and prediction points:
%maxx = 0.9; minx = 0.3;   % range of utilization
maxx = 2; minx = 1.1;   % range of utilization
%service_rate = 1;       % fixed service rate
arrival_rate = 1;

K = 1000;               % number of prediction points 
%k = 4;                 % number of design points
runlength = 1000;       % runlength at each design point
%X = (minx:((maxx-minx)/(k-1)):maxx)';               % design points of arrive rate
%truek = X./(1-X);        % analytic values at design points
%truek = arrival_rate./(X.*(X-arrival_rate));  
XK = (minx:((maxx-minx)/(K-1)):maxx)';              % prediction points
%C = 100;                % total computation budget
%y_pred_true = X./(Xpred .* (Xpred - X));   % analytic values at prediction points

%n = ones(k,1)*ceil(C*1/k); 
trueY =1./(XK.*(XK-1));  % analytic values at prediction points
%trueVhat = 2.*XK.*(1+XK)./runlength./(1-XK).^4;
trueVhat =   4./(XK.*(1-1./XK).^4*runlength);
temp1 = trueVhat;
temp2 = trueY;
Z = sqrt(temp1)./abs(temp2); 
figure
plot(XK,trueY);
figure
plot(XK,sqrt(trueVhat))
figure
plot(XK,Z);
%xlim([1.1 2]);


%[y Y  Vhat] = MM1sim_waiting_w(service_rate,X,n,runlength,'stationary');  
