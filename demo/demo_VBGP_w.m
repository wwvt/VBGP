% data generation
clear;clc;

s = RandStream('mt19937ar','Seed',42);
RandStream.setGlobalStream(s);

maxx = 2; minx = -2;    % range 
K = 100;               % number of prediction points 
k = 10;                 % number of design points
C = 10;               % total computation budget
X = (minx:((maxx-minx)/(k-1)):maxx)';               % design points
trueY = sin(X);        % analytic values at design points
XK = (minx:((maxx-minx)/(K-1)):maxx)';              % prediction points
trueYK = sin(XK);   % analytic values at prediction points
trueVar = 0.05*XK.*XK +0.01;

n = ones(k,1)*ceil(C*1/k); 

[y, Y, Vhat]  =  demo_kris(X,n);  

save('demo_VBGP_kris.mat')

clear;clc;

% equal allocation example
load demo_VBGP_kris
%%
% VBGP

replication = n;
A = getRepMatrix(replication);


%[NMSE, Ey, Vy, mutst, diagSigmatst, atst, diagCtst, LambdaTheta, convergence] = vbgp_ui_w_demo(X, y, XK, trueYK, A, 100);


x_tr = X;
y_tr = y;
x_tst = XK;
y_tst = trueYK;
iter = 100;

%[NMSE, Ey, Vmean, mutst, diagSigmatst, atst, diagCtst, ... 
%    LambdaTheta, loghyperGP, convergence] = vbgp_ui_w_demo(x_tr, y_tr, x_tst, y_tst, A, iter, loghyperGP, loghyper)

[k, D] = size(x_tr);

meanp=mean(Y); % WW edited

% Covariance functions
% covfuncSignal = {'covSum',{'covSEisoj','covConst'}};
% covfuncNoise  = {'covSum',{'covSEisoj','covNoise'}};
 covfuncSignal = {'covSEardj'}; % D+1 hyp
 covfuncNoise  = {'covSEardj'}; % D+1 hyp

   SignalPower = var(Y,1);
   NoisePower = SignalPower/4;
   lengthscales=log((max(x_tr)-min(x_tr))'/2);
    
    display('Running standard, homoscedastic GP...')
%    loghyperGP = [lengthscales; 0.5*log(SignalPower); 0.5*log(NoisePower);-0.5*log(max(SignalPower/20,meanp^2))];
      loghyperGP = [lengthscales; 0.5*log(SignalPower); 0.5*log(NoisePower)];
 %  loghyperGP = minimize(loghyperGP, 'gpr', 40, {'covSum', {'covSEardj','covNoise','covConst'}}, x_tr, Y');
     loghyperGP = minimize(loghyperGP, 'gpr', 40, {'covSum', {'covSEardj','covNoise'}}, x_tr, Y);
    % 3 par: lengthscale ell, sf2, white noise s2

lengthscales=loghyperGP(1:D); % lengthscale ell in covSEardj
x_tr = x_tr./(ones(k,1)*exp(lengthscales(:)'));
x_tst = x_tst./(ones( size(x_tst,1) ,1)*exp(lengthscales(:)'));

SignalPower = exp(2*loghyperGP(D+1)); % sf2 in covSEardj
NoisePower =  exp(2*loghyperGP(D+2)); % s2 in covNoise

    sn2 = 1;
    mu0 = log(NoisePower)-sn2/2-2; % 2*s2 - sn2/2 - 2
    
%    loghyperSignal = [0; 0.5*log(SignalPower);-0.5*log(ConstPower)];
%    loghyperNoise =  [0; 0.5*log(sn2); 0.5*log(sn2*0.25)];
%     loghyperSignal = [zeros([D,1]);0.5*log(SignalPower)]; % [0;sf2 in covSEardj] 
%     loghyperNoise =  [zeros([D,1]);0.5*log(sn2)]; % [0;0.5*log(sn2)]
     loghyperSignal = [lengthscales;0.5*log(SignalPower)]; % [0;sf2 in covSEardj] % for evaluating covfuncSignal, need ell and sf2
     loghyperNoise =  [0.5*log(1);0.5*log(NoisePower)]; % [0;0.5*log(sn2)] % % for evaluating covfuncNoise, need ell and sf2

     



    display('Initializing VHGPR (keeping hyperparameters fixed)...')
    LambdaTheta = [log(0.5)*ones(k,1);loghyperSignal;loghyperNoise;mu0]; % D + 2 + 2 + 1
[LambdaTheta, convergence0] = minimize(LambdaTheta, 'vhgpr_w_1101', 30, covfuncSignal, covfuncNoise, 2, x_tr, y_tr, A); % A added % error here % 2=fixhyp
 % LambdaTheta = fminsearch('vhgpr_w_1101_derivfree',LambdaTheta, 30, covfuncSignal, covfuncNoise, 2, x_tr, y_tr, A ) ;
 % fminsearch changes LabdaTheta but leads to bad MSE
  display('Running VHGPR...')
     [LambdaTheta, convergence] = minimize(LambdaTheta, 'vhgpr_w_1101', iter, covfuncSignal, covfuncNoise, 0, x_tr, y_tr, A); % A added
     convergence = [convergence0; convergence];

% ntheta1 = eval(feval(covfuncSignal{:})) % 2
% ntheta2 = eval(feval(covfuncNoise{:})) % 2
%      
% Lambda = exp(LambdaTheta(1:k));
% theta1 = LambdaTheta(k+1:k+ntheta1); % 2 par
% theta2 = LambdaTheta(k+ntheta1+1:k+ntheta1+ntheta2); % 2 par
% delta0 = LambdaTheta(k+ntheta1+ntheta2+1); % 1 par
% 
% 
% Kf = feval(covfuncSignal{:},theta1, X);

    
 % prediction
[Ey, Vy, mutst, diagSigmatst, atst, diagCtst]= vhgpr_w_1101(LambdaTheta, covfuncSignal, covfuncNoise, 0, x_tr, y_tr, A, x_tst); % A added

    
    
figure
%plotvarianza(XK, Ey, Vy)
%hold on
%plot(X, Y,'xb', XK, Ey,'r')
plot(XK, Ey,'r', XK, trueYK,'k') %predicted mean v.s. true mean


figure
plot(XK, Vy,'r', XK, trueVar,'k') %predicted var v.s. true var



%%
% VBGP using VHGPR parameter setting

replication = n;
A = getRepMatrix(replication);


%[NMSE, Ey, Vy, mutst, diagSigmatst, atst, diagCtst, LambdaTheta, convergence] = vbgp_ui_w_demo(X, y, XK, trueYK, A, 100);


x_tr = X;
y_tr = y;
x_tst = XK;
y_tst = trueYK;
iter = 100;

%[NMSE, Ey, Vmean, mutst, diagSigmatst, atst, diagCtst, ... 
%    LambdaTheta, loghyperGP, convergence] = vbgp_ui_w_demo(x_tr, y_tr, x_tst, y_tst, A, iter, loghyperGP, loghyper)

[k, D] = size(x_tr);

meanp=mean(Y); % WW edited

% Covariance functions
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


    display('Initializing VBGP (keeping hyperparameters fixed)...')
    LambdaTheta = [log(0.5)*ones(k,1);loghyperSignal;loghyperNoise;mu0]; % D + 2 + 2 + 1
[LambdaTheta, convergence0] = minimize(LambdaTheta, 'vhgpr_w_1101', 30, covfuncSignal, covfuncNoise, 2, x_tr, y_tr, A); % A added % error here % 2=fixhyp
 % LambdaTheta = fminsearch('vhgpr_w_1101_derivfree',LambdaTheta, 30, covfuncSignal, covfuncNoise, 2, x_tr, y_tr, A ) ;
 % fminsearch changes LabdaTheta but leads to bad MSE
  display('Running VBGP...')
     [LambdaTheta, convergence] = minimize(LambdaTheta, 'vhgpr_w_1101', iter, covfuncSignal, covfuncNoise, 0, x_tr, y_tr, A); % A added
     convergence = [convergence0; convergence];
%%
% ntheta1 = eval(feval(covfuncSignal{:})) % 2
% ntheta2 = eval(feval(covfuncNoise{:})) % 2
%      
% Lambda = exp(LambdaTheta(1:k));
% theta1 = LambdaTheta(k+1:k+ntheta1); % 2 par
% theta2 = LambdaTheta(k+ntheta1+1:k+ntheta1+ntheta2); % 2 par
% delta0 = LambdaTheta(k+ntheta1+ntheta2+1); % 1 par
% 
% 
% Kf = feval(covfuncSignal{:},theta1, X);

    
 % prediction
[Ey, Vy, mutst, diagSigmatst, atst, diagCtst]= vhgpr_w_1101(LambdaTheta, covfuncSignal, covfuncNoise, 0, x_tr, y_tr, A, x_tst); % A added

    
    
figure
%plotvarianza(XK, Ey, Vy)
%hold on
%plot(X, Y,'xb', XK, Ey,'r')
plot(XK, Ey,'r', XK, trueYK,'k') %predicted mean v.s. true mean


figure
plot(XK, Vy,'r', XK, trueVar,'k') %predicted var v.s. true var


%%
% VB orig
[NMSE2, NLPD2, Ey2, Vy2, mutst2, diagSigmatst2, atst2, diagCtst2, LambdaTheta2, convergence2] = ... 
    vhgpr_ui(X, Y, XK, trueYK, 100);

%%
figure
%plotvarianza(XK, Ey, Vy)
%hold on
%plot(X, Y,'xb', XK, Ey,'r')
plot(XK, Ey2,'r', XK, trueYK,'k') %predicted mean v.s. true mean


figure
plot(XK, Vy2,'r', XK, trueVar,'k') %predicted var v.s. true var


%%
figure
plotvarianza(XK, trueYK, trueVar)
hold on
plot(X, Y,'xb')

figure
plotvarianza(XK, Ey, Vy)
hold on
%plot(X, Y,'xb', XK, Ey,'r')
plot(XK, Ey,'r', XK, trueYK,'k') %predicted mean v.s. true mean

figure
plot(XK, Vy,'r', XK, trueVar,'k') %predicted var v.s. true var

figure
plot(XK, Vy,'r')


%%
%---------------------GP predict---------------------%

meanfunc = {'meanConst'};      
hyp.mean = [2]; 
covfunc = {'covSEard'}; 
hyp.cov = [log(0.1);log(5)];
likfunc = {'likGauss'};  
sn = 0.1; hyp.lik = -2.132935872659505; 

hyp2 = minimize(hyp, @gp, -100, @infGaussLik, meanfunc, covfunc, likfunc, X, Y); 
[m ns2] = gp(hyp2, @infGaussLik, meanfunc, covfunc, [], X, Y, XK);
gpy = m;

plot(XK, gpy,'r', XK, trueYK,'k') %predicted mean v.s. true mean
