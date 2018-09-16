function [NMSE, Ey, Vmean, mutst, diagSigmatst, atst, diagCtst, ... 
    LambdaTheta, loghyperGP, convergence] = vhgpr_ui_w(x_tr, y_tr, x_tst, y_tst, A, iter, loghyperGP, loghyper)
% WW edited.  9/22/2017
% Changed all vhgpr to vhgpr_w, nlogprob_vhgpr to nlogprob_vhgpr_w.
% (X, y, Xpred, y_pred_true, A, 100);
% x_tr=X; y_tr=y; x_tst=Xpred; y_tst =  y_pred_true; A=A; iter=100;
% x_tr=X; y_tr=y_init; x_tst=X; y_tst =  ytemp; A=A_init; iter=10;
% VHGPR_UI implements a convenient user interface for VHGPR
%
% Input:    - x_tr: Training input data. One vector per row.
%           - y_tr: Training output value. One scalar per row.
%           - x_tst: Test input data. One vector per row.
%           - y_tst: Tetst output value. One scalar per row. Used to
%           compute error measures, predictions below do not use it.
%           - iter: Number of iterations for the optimizer. (Optional).
%           - loghyperGP: Use this full GP hyperparameters to initialize
%           VHGPR's hyperparameters. (Optional).
%           - loghyper: Use these initial VHGPR's hyperparameters.
%           (Optional).
%
% Output:   - NMSE: Normalized Mean Square Error on test set.
%           - NLPD: Negative Log-Probability Density on test set.
%           - Ey: Expectation of the approximate posterior for test data.
%           - Vy: Variance of the approximate posterior for test data.
%           - mutst: Expectation of the approximate posterior for g.
%           - diagSigmatst: Variance of the approximate posterior for g.
%           - atst: Expectation of the approximate posterior for f.
%           - diagCtst: Variance of the approximate posterior for f.
%           - LambdaTheta: Obtained values for [Lambda hyperf hyperg mu0]
%           - convergence: Evolution of the MV bound during optimization.
%       
% Note that the input dimensions are scaled according to the lengthscales 
% in loghyperGP (either supplied or determined by this function) in 
% addition to the single lengthscale in hyperf and hyperg.
%
% The algorithm in this file is based on the following paper:
% M. Lazaro Gredilla and M. Titsias, 
% "Variational Heteroscedastic Gaussian Process Regression"
% Published in ICML 2011
%
% See also: vhgpr
%
% - NOTE: This is just a helper function providing a default initialization,
% other initializations, or other optimization techniques, may be better
% suited to achieve a better bound on other problems.
%
% Copyright (c) 2011 by Miguel Lazaro Gredila


[n, D] = size(x_tr);

% for dataset with the y_tr(l).n structure % WW added
for l = 1:n
   Y(l)  = mean(y_tr(l).n); 
 %  Vhat(l) = var(y_tr(l).n,0,2);
end

meanp=mean(Y); % WW edited

% Covariance functions
% covfuncSignal = {'covSum',{'covSEisoj','covConst'}};
% covfuncNoise  = {'covSum',{'covSEisoj','covNoise'}};
 covfuncSignal = {'covSEardj'}; % D+1 hyp
 covfuncNoise  = {'covSEardj'}; % D+1 hyp

if nargin < 6  % 5->6 because of A added
    iter = 40;
end
if nargin < 7 % 6->7 because of A added
    % Hyperparameter initialization
    SignalPower = var(Y,1);
    NoisePower = SignalPower/4;
    lengthscales=log((max(x_tr)-min(x_tr))'/2);
    
    display('Running standard, homoscedastic GP...')
%    loghyperGP = [lengthscales; 0.5*log(SignalPower); 0.5*log(NoisePower);-0.5*log(max(SignalPower/20,meanp^2))];
      loghyperGP = [lengthscales; 0.5*log(SignalPower); 0.5*log(NoisePower)];
 %  loghyperGP = minimize(loghyperGP, 'gpr', 40, {'covSum', {'covSEardj','covNoise','covConst'}}, x_tr, Y');
     loghyperGP = minimize(loghyperGP, 'gpr', 40, {'covSum', {'covSEardj','covNoise'}}, x_tr, Y');
end
lengthscales=loghyperGP(1:D);
x_tr = x_tr./(ones(n,1)*exp(lengthscales(:)'));
x_tst = x_tst./(ones( size(x_tst,1) ,1)*exp(lengthscales(:)'));

if nargin < 8 % Learn hyperparameters  % 7-> 8 because A added
    SignalPower = exp(2*loghyperGP(D+1));
    NoisePower =  exp(2*loghyperGP(D+2));
%    ConstPower =  exp(-2*loghyperGP(D+3));
    
 %   sn2 = 1; %9/9/2018 original
     sn2 = 1;
    mu0 = log(NoisePower)-sn2/2-2;
%     mu0 = 0;

%    loghyperSignal = [0; 0.5*log(SignalPower);-0.5*log(ConstPower)];
%    loghyperNoise =  [0; 0.5*log(sn2); 0.5*log(sn2*0.25)];
     loghyperSignal = [zeros([D,1]);0.5*log(SignalPower)];
     loghyperNoise =  [zeros([D,1]); 0.5*log(sn2)]; % 9/9/2018 VBGP var init par
       
    display('Initializing VHGPR (keeping hyperparameters fixed)...')
  %  LambdaTheta = [-1.01835192176657;-0.554857279270721;-0.853618611145473;-0.777632850640241;-1.11365825665709;-1.37656450959410;-1.30756635282257;-1.29245611778058;-1.48130629961746;-0.860402209415784;-0.308626125685028;-1.14286964414799;-1.06357253332555;-0.782930509769712;-1.20491787654869;0.130968925755486;-1.13845178485962;0.444264935954312;0.195336350438129;-0.448438473953057;-0.102813540687512;3.60502041637454;-5.43543609810159;0.165417741594779;0.493698579174624;-1.02014722462961;1.14651166731648];
    LambdaTheta = [log(0.5)*ones(n,1);loghyperSignal;loghyperNoise;mu0];
 %XX=X;
%X=LambdaTheta; f = 'vhgpr_w'; length=30;varargin = [covfuncSignal, covfuncNoise, 2, x_tr, y_tr, A];
%(X, f, length, varargin)
%vhgpr_w(X=LambdaTheta, covfunc1, covfunc2, fixhyp, X, y, A, Xtst)
   [LambdaTheta, convergence0] = minimize(LambdaTheta, 'vhgpr_w_1101', 30, covfuncSignal, covfuncNoise, 2, x_tr, y_tr, A); % A added % error here % 2=fixhyp
  
  % LambdaTheta = fminsearch('vhgpr_w_1101_derivfree',LambdaTheta, 30, covfuncSignal, covfuncNoise, 2, x_tr, y_tr, A ) ;
  % fminsearch changes LabdaTheta but leads to bad MSE
  display('Running VHGPR...')
 %   [LambdaTheta, convergence] = minimize(LambdaTheta, 'vhgpr_w_1101', iter, covfuncSignal, covfuncNoise, 0, x_tr, y_tr, A); % A added
     [LambdaTheta, convergence] = minimize(LambdaTheta, 'vhgpr_w_1101', iter, covfuncSignal, covfuncNoise, 0, x_tr, y_tr, A); % A added
     convergence = [convergence0; convergence];
else % Use given, fixed hyperparameters
    display('Running VHGPR...')
    LambdaTheta = [log(0.5)*ones(n,1);loghyper];
    [LambdaTheta, convergence] = minimize(LambdaTheta, 'vhgpr_w_1101', iter, covfuncSignal, covfuncNoise, 2, x_tr, y_tr, A); % A added
end

%Prediction
if nargout > 1
    [Ey, Vmean, mutst, diagSigmatst, atst, diagCtst]= vhgpr_w_1101(LambdaTheta, covfuncSignal, covfuncNoise, 0, x_tr, y_tr, A, x_tst); % A added
  %  [Ey, Vy, mutst, diagSigmatst, atst, diagCtst]= vhgpr_w_trS(LambdaTheta, covfuncSignal, covfuncNoise, 0, x_tr, y_tr, A, x_tst); % A added
else
    [Ey, Vmean, mutst, diagSigmatst]= vhgpr_x_1027(LambdaTheta, covfuncSignal, covfuncNoise, 0, x_tr, y_tr, A, x_tst);  % A added
end

NMSE=mean((Ey-y_tst).^2)/mean((meanp-y_tst).^2);

% if nargout > 1
%     [NLPDapprox, NLPD] = nlogprob_vhgpr_w(y_tst, mutst, diagSigmatst, atst, diagCtst);
%     NLPD = mean(NLPD);
% end

