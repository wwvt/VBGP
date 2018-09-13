function [out1]= vhgpr_w_ga_derivfree(LambdaTheta, covfunc1, covfunc2, X, y, A)
% WW edited: A should be an input
% WW edited 10/27/2017
% Last update: 10/15/2017 minimize problems
%covfunc1 = covfuncSignal; covfunc2 = covfuncNoise;
% VHGPR - MV bound for heteroscedastic GP regression
% Input:    - LambdaTheta: Selected values for [Lambda hyperf hyperg mu0].
%           - covfunc1: Covariance function for the GP f (signal).
%           - covfunc2: Covariance function for the GP g (noise).
%           - fixhyp: 
%                        0 - Nothing is fixed (all variational parameters
%                        and hyperparameters are learned).
%                        1 - Hyperparameters for f are fixed.
%                        2 - Hyperparameters for f and g are fixed. (Only
%                        variational parameters are learned).
%           - X: Training input data. One vector per row.
%           - y: Training output value. One scalar per row.
%           - Xtst: Test input data. One vector per row.
%
% Output:   - out1: 
%                   Training mode: MV bound.
%                   Testing mode: Expectation of the approximate posterior
%                   for test data.
%           - out2: 
%                   Training mode: MV bound derivatives wrt LambdaTheta.
%                   Testing mode: Variance of the approximate posterior
%                   for test data.
%           - mutst: Expectation of the approximate posterior for g.
%           - diagSigmatst: Variance of the approximate posterior for g.
%           - atst: Expectation of the approximate posterior for f.
%           - diagCtst: Variance of the approximate posterior for f.
%
% Modes:    Testing (all inputs are given) / Training (last input is omitted)
%
% The algorithm in this file is based on the following paper:
% M. Lazaro Gredilla and M. Titsias, 
% "Variational Heteroscedastic Gaussian Process Regression"
% Published in ICML 2011
%
% See also: vhgpr_ui
%
% Copyright (c) 2011 by Miguel Lazaro Gredila

[n,D] = size(X); 

% for dataset with the y(l).n structure
 Y = zeros(n,1);

for l = 1:n
   Y(l)  = mean(y(l).n); 
   bd(l) = sum((y(l).n).^2);
   Vhat(l) = var(y(l).n,0,2);
end
 
% Parameter initialization
ntheta1 = eval(feval(covfunc1{:}));
ntheta2 = eval(feval(covfunc2{:}));

% if n+ntheta1+ntheta2+1 ~= size(LambdaTheta,1),error('Incorrect number of parameters');end
% Lambda = exp(LambdaTheta(1:n));
% theta1 = LambdaTheta(n+1:n+ntheta1);
% theta2 = LambdaTheta(n+ntheta1+1:n+ntheta1+ntheta2);
% delta0 = LambdaTheta(n+ntheta1+ntheta2+1);
% mu0 = 0;
% Kf = feval(covfunc1{:},theta1, X);
% Kg = feval(covfunc2{:},theta2, X);


% ------------WW edited------------%
% 
nV = sum(A,2); % # replications need to be calculated within the function
withoutrep = any(nV==1);  %**************************

  nn = sum(nV); % total budget, i.e. total number of replications
if n+ntheta1+ntheta2+1 ~= size(LambdaTheta,1),error('Incorrect number of parameters');end
Lambda = exp(LambdaTheta(1:n));
theta1 = LambdaTheta(n+1:n+ntheta1);
theta2 = LambdaTheta(n+ntheta1+1:n+ntheta1+ntheta2);
delta0 = LambdaTheta(n+ntheta1+ntheta2+1);

mu0 = 0;
Kf = feval(covfunc1{:},theta1, X);
Kg = feval(covfunc2{:},theta2, X);

   DD = diag(Lambda);
   IKDinv = (eye(n) + Kg * DD)\eye(n);
   KDinv = ((Kg\eye(n)) + DD)\eye(n);
  
% sLambda = sqrt(Lambda);
% cinvB = chol(eye(n) + Kg.*(sLambda*sLambda'))'\eye(n);   % O(n^3)
% cinvBs = cinvB.*(ones(n,1)*sLambda');
% hBLK2 = cinvBs*Kg;                                         % O(n^3)
% Sigma = Kg - hBLK2'*hBLK2;                                 % O(n^3) (will need the full matrix for derivatives)
 
 Sigma = (Kg\eye(n) + DD)\eye(n);  % same thing    %*********************
 mu =((ones(1,n) * diag(Lambda)-0.5*nV')* Kg + delta0.*ones(1,n))';
 %ones(n,1) * 0; % in this right case, the resulting MSE is too big
 %mu = Kg * (Lambda-0.5) + delta0; % make a difference %wrong in our case
 %  mucheck = Kg * DD * ones(n,1) - 1/2. * Kg * nV; %same when delta0=0 
  
  Rdiag = exp(mu-diag(Sigma)/2);
  R = diag(repelem(Rdiag,nV));
  Rdiaginv = diag(1./(Rdiag));
  Rinv = diag(1./repelem(Rdiag,nV));  
 % M = A'* Kf *A + R;
  Phi = diag(Rdiag./nV);
  G = Kf+Phi;
  Ginv = (Kf+Phi)\eye(n);
  MlogDet = sum((mu-diag(Sigma)./2).*(nV-1) + log(nV)) + log(det(G));  
    % MlogDet = log(det(M)); % same
  yy=[];
    
%     BB = A * Rinv * A';
%         BBinv = inv(BB);

        %BB = Rdiaginv*exp(-theta2(end));   %*************
        
        BB = inv(Phi);
        BBinv = Phi;

        
        
    IBKinv = (eye(n) + BB * Kf)\eye(n); %singular
  for i=1:length(y)
    yy = [yy,(y(i).n)];
  end

%      goal = (yy-mu0) * Rinv * (yy-mu0)' 
%      goalchecked = sum((yy-mu0).^2'./repelem(Rdiag,nV))
%      goalcheck =  bd * inv(Phi)/diag(nV) * ones(n,1)
%   goalchecks = bd * Rdiaginv * ones(n,1)

% if  withoutrep == 1  %************************************
%     F1 = -0.5 *(nn*log(2*pi)+MlogDet+ sum((yy).^2'./repelem(Rdiag,nV)) - Y' * diag(nV./Rdiag) * Kf * Ginv * Y);% O(m3)
    Minv = Rinv - Rinv * A' * Kf * Ginv * Phi * A * Rinv;
   % -0.5*nn*log(2*pi) %normal
   %  -0.5*MlogDet %normal


    F1 = -0.5*(nn*log(2*pi)+MlogDet+(yy-mu0) * Minv * (yy-mu0)'); % O(n3) same.
%  else
%         Minv = Rinv - Rinv * A' * Kf * Ginv * Phi * A * Rinv;
%         F1 = -0.5*(nn*log(2*pi)+MlogDet+(yy) * Minv * (yy)');
%   end
  
  F2 = -0.25 .* nV' * diag(Sigma);
%  F3 = -KLdiv(mu,repelem(delta0,n)',Sigma,Kg);  %***************************
  F3 = -KLdiv(mu,repelem(delta0,n)',Sigma,Kg);  %*************************** KLdiv definition corrected

  F = F1+F2+F3;
  out1 = -F;
end
