% data generation
clear;clc;

s = RandStream('mt19937ar','Seed',42);
RandStream.setGlobalStream(s);

maxx = 2; minx = -2;    % range 
K = 1000;               % number of prediction points 
k = 50;                 % number of design points
C = 100;               % total computation budget
X = (minx:((maxx-minx)/(k-1)):maxx)';               % design points
trueY = sin(X);        % analytic values at design points
XK = (minx:((maxx-minx)/(K-1)):maxx)';              % prediction points
trueYK = sin(XK);   % analytic values at prediction points
trueVar = 0.05*XK.*XK +0.01;

% equal allocation example
n = ones(k,1)*ceil(C*1/k); 

[y, Y, Vhat]  =  demo_kris(X,n);  

save('demo_VBGP_kris.mat')

clear;clc;

load demo_VBGP_kris

%%
% VBGP

replication = n;
A = getRepMatrix(replication);
x_tr = X;
y_tr = y;
x_tst = XK;
y_tst = trueYK;
iter = 100;
[k, D] = size(x_tr);

meanp=mean(Y); 

covfuncSignal = {'covSum',{'covSEisoj','covConst'}};
covfuncNoise  = {'covSum',{'covSEisoj','covNoise'}};

  SignalPower = var(Y,1);
   NoisePower = SignalPower/4;
   lengthscales=log((max(x_tr)-min(x_tr))'/2);
    
    display('Running standard, homoscedastic GP...')
    loghyperGP = [lengthscales; 0.5*log(SignalPower); 0.5*log(NoisePower);-0.5*log(max(SignalPower/20,meanp^2))];
    loghyperGP = minimize(loghyperGP, 'gpr', 40, {'covSum', {'covSEardj','covNoise','covConst'}}, x_tr, Y);

    lengthscales=loghyperGP(1:D); % lengthscale ell in covSEardj
    SignalPower = exp(2*loghyperGP(D+1));
    NoisePower =  exp(2*loghyperGP(D+2));
    ConstPower =  exp(-2*loghyperGP(D+3));
    
    sn2 = 1;
    mu0 = log(NoisePower)-sn2/2-2;
    loghyperSignal = [0; 0.5*log(SignalPower);-0.5*log(ConstPower)];
    loghyperNoise =  [0; 0.5*log(sn2); 0.5*log(sn2*0.25)];

     
x_tr = x_tr./(ones(k,1)*exp(lengthscales(:)'));
x_tst = x_tst./(ones( size(x_tst,1) ,1)*exp(lengthscales(:)'));

display('Initializing VBGP...')
LambdaTheta = [log(0.5)*ones(k,1);loghyperSignal;loghyperNoise;mu0]; % D + 2 + 2 + 1

% ------------------------------------------%
% without replication: comment out; with replication: use this to help initialization
[LambdaTheta, convergence] = minimize(LambdaTheta, 'vhgpr', 100, covfuncSignal, covfuncNoise, 0, X, Y);
% ------------------------------------------%

display('Running VBGP...')
[LambdaTheta, convergence0] = minimize(LambdaTheta, 'vhgpr_w_1101', 30, covfuncSignal, covfuncNoise, 2, x_tr, y_tr, A);
[LambdaTheta, convergence] = minimize(LambdaTheta, 'vhgpr_w_1101', 100, covfuncSignal, covfuncNoise, 0, x_tr, y_tr, A); 
 % prediction
[Ey, Vy, mutst, diagSigmatst, atst, diagCtst]= vhgpr_w_1101(LambdaTheta, covfuncSignal, covfuncNoise, 0, x_tr, y_tr, A, x_tst); 
%%
figure
plot(XK, Ey,'r', XK, trueYK,'k') %predicted mean v.s. true mean
figure
plot(XK, Vy,'r', XK, trueVar,'k') %predicted var v.s. true var
