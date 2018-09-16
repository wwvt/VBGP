function [out1, out2, mutst, diagSigmatst, atst, diagCtst]= vhgpr_w_1101(LambdaTheta, covfunc1, covfunc2, fixhyp, X, y, A, Xtst)
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
%   Vhat(l) = var(y(l).n,0,2);
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

sLambda = sqrt(Lambda);
cinvB = chol(eye(n) + Kg.*(sLambda*sLambda'))'\eye(n);   % O(n^3)
cinvBs = cinvB.*(ones(n,1)*sLambda');
hBLK2 = cinvBs*Kg;                                         % O(n^3)
Sigma = Kg - hBLK2'*hBLK2;                                 % O(n^3) (will need the full matrix for derivatives)

%IKDinv = (eye(n) + Kg * DD)\eye(n);
IKDinv = cinvB' * cinvB; % same thing
%KDinv = ((Kg\eye(n)) + DD)\eye(n);
KDinv = Kg * IKDinv; % same thing
   
 %Sigma = (Kg\eye(n) + DD)\eye(n);  % same thing    %*********************
  mu =((ones(1,n) * diag(Lambda)-0.5*nV')* Kg + delta0.*ones(1,n))'; %ones(n,1) * 0; 
% mu = Kg * (Lambda-0.5) + delta0; % make a difference %wrong in our case
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

p = 1e-3; % This value doesn't affect the result, it is a scaling factor
scale = 1./sqrt(p+diag(R)); 
Rscale = 1./(1+p./diag(R));
% Ls = chol((A'*Kf*A).*(scale*scale')+diag(Rscale));     % O(n^3)
% Lys = (Ls'\(yy'.*scale));
% alphascale = Ls\Lys;
% alpha = alphascale.*scale;       
        
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
   %------------scaling test---------------%

 
%   previous = -0.5*(yy-mu0) * Minv * (yy-mu0)';
%  scalingtest = -0.5*(yy-mu0) * alpha ;% same. not helpful.
    %------------scaling test---------------%

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

% --- Derivatives   % Note: VBGP Rcode uses optim() method ' SANN, which does not consider the derivatives. 
if nargin == 7 && nargout == 2  % training, getting [LambdaTheta, convergence0] = minimize() %6->7 because of A as input 
                                % in minimize.m function, [f0 df0] = feval(f, X, varargin{:});          % get function value and gradient
                                % need to find the derivatives. The old one
                                % does not work
out2=zeros(n+1+ntheta1+ntheta2,1);
%<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    
    % wrt Lambda

%   for i = 1:n    
%             dDD = zeros(n);
%             dDD(i,i) = 1;
%             temp = Kg * dDD * ones(n,1);
%             dmuD = temp(i);
%             temp2  = -KDinv * dDD * KDinv;
%             dSigmaD = temp2(i,i);
%             temp3 = Rdiag(i);
%             dRDi = temp3 * dmuD - 1/2 * temp3 * dSigmaD; 
%             dRD = zeros(nn);
% 
%           for j=  (sum(nV(1:i-1))+1):(sum(nV(1:i)))
%             dRD(j,j) = dRDi;
%           end
%             dBD  = exp(-theta2(end))* A * inv(R) *  dRD * inv(R)  * A';  %**********************
%             dFLambda(i)= trace(-dBD*BBinv)+ trace(IBKinv*dBD* Kf) + bd * dBD * ones(n,1) + ...
%                        (-1)*(Y' * (eye(n) - BB * Kf * IBKinv) * dBD * Kf * IBKinv * BB' * Y ) +...
%                        (-1)* Y'* BB * Kf*  IBKinv * dBD* Y + ...
%                        trace(IKDinv * Kg * dDD) -  trace(IKDinv * Kg * dDD * IKDinv ) + ...
%                        2*  ones(n,1)' * dDD * Kg * DD * ones(n,1)  + ...
%                        (-1)* (ones(n,1)'* Kg  * dDD * ones(n,1)) - 1/2 *trace(KDinv * dDD *  KDinv);
%   end
%     out2(1:n) = dFLambda'/2;
 
    
      for i = 1:n    
            dDD = zeros(n);
            dDD(i,i) = 1;
            temp = Kg * dDD * ones(n,1);
            dmuD = temp(i);
            temp2  = -KDinv * dDD * KDinv;
            dSigmaD = temp2(i,i);
            temp3 = Rdiag(i);
            dRDi = temp3 * dmuD - 1/2 * temp3 * dSigmaD; 
            
            dRD = zeros(n);
            dRD(i,i) = dRDi ;
 
        %    dBD  = -diag(1./nV) * Rdiaginv *  dRD * Rdiaginv;  %**********************
            dBD  = -diag(nV) * Rdiaginv *  dRD * Rdiaginv;  %**********************
            dFLambda(i)= trace(-dBD*BBinv)+ trace(IBKinv*dBD* Kf) + bd * dBD * (1./nV) + ... %% * ones(n,1) 
                       (-1)*(Y' * (eye(n) - BB * Kf * IBKinv) * dBD * Kf * IBKinv * BB' * Y ) +...
                       (-1)* Y'* BB * Kf*  IBKinv * dBD* Y + ...
                       trace(IKDinv * Kg * dDD) -  trace(IKDinv * Kg * dDD * IKDinv ) + ...
                       2*  ones(n,1)' * dDD * Kg * DD * ones(n,1)  + ...
                       (-1)* (ones(n,1)'* Kg  * dDD * ones(n,1)) - 1/2 *trace(KDinv * dDD *  KDinv);
       end
    out2(1:n) = Lambda.*dFLambda'/2;   %**************************
    
    
 %<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    if fixhyp < 1       % fixhyp = 0, do this. 
    % wrt Kf hyperparameters
            for k = 1:ntheta1
             dKthetaf  = feval(covfunc1{:}, theta1, X, k);
           %  dFthetaf = sum(sum(IBKinv * BB  - BB' * Y *  Y' * BB' * IBKinv' + IBKinv' * Kf'* BB' *Y*Y'  * BB' *   IBKinv'  * BB ).*dKthetaf);
          dFthetaf = trace(IBKinv * BB*dKthetaf) - (Y' * BB * (eye(n) - Kf * IBKinv *BB) * dKthetaf * IBKinv * BB * Y );
           out2(n+k) = dFthetaf/2;
            end
    end
    
 
    if fixhyp < 2         %  fixhyp = 0, do this
%<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
%         % wrt Kg hyperparameters
%     for k = 1:ntheta2    
%     dKthetag = feval(covfunc2{:}, theta2, X, k); 
%    % temp = dKthetag * DD * ones(n,1);
%     temp = dKthetag * (DD * ones(n,1)-0.5*nV);  %*********************
%     temp2  = KDinv * inv(Kg) * dKthetag * inv(Kg) * KDinv;
%        
%     dRthetag = zeros(nn);
%     for i = 1:n
%     dmuthetag = temp(i);
%     dSigmathetag = temp2(i,i);
%     temp3 = Rdiag(i);
%     dRthetagi = temp3 * dmuthetag - 1/2 * temp3 * dSigmathetag; 
%     for j=  (sum(nV(1:i-1))+1):(sum(nV(1:i)))
%     dRthetag(j,j) = dRthetagi;
%     end
%     end
%     dBdthetag = exp(-theta2(end))* A * inv(R)  *  dRthetag * inv(R)  * A'; %*************************
%     dFthetag= trace(-dBdthetag*BBinv)+ trace(IBKinv*dBdthetag* Kf) + bd * dBdthetag * ones(n,1) + ...
%                (-1)*(Y' * (eye(n) - BB * Kf * IBKinv) * dBdthetag * Kf * IBKinv * BB' * Y ) +...
%                (-1)* Y'* BB * Kf*  IBKinv * dBdthetag* Y + ...
%                trace(IKDinv * dKthetag * DD) -  trace(IKDinv * dKthetag * DD * IKDinv ) + ...
%                ones(n,1)' * (DD - eye(n)) * dKthetag * DD * ones(n,1)  + ...
%                1/2 *trace(KDinv * inv(Kg) *  dKthetag * inv(Kg) *  KDinv) + ...
%                -1/4* nV' * dKthetag *  nV + ones(n,1)' * dKthetag *(nV/2);
%         out2(n+ntheta1+k) = dFthetag/2;
%     end


    % wrt Kg hyperparameters
    for k = 1:ntheta2    
    dKthetag = feval(covfunc2{:}, theta2, X, k); 
   % temp = dKthetag * DD * ones(n,1);
    temp = dKthetag * (DD * ones(n,1)-0.5*nV);  %*********************
   % temp2  = KDinv * inv(Kg) * dKthetag * inv(Kg) * KDinv;
    temp2  = IKDinv * dKthetag * IKDinv; %theoretically same, numerically different
       
   
            for i = 1:n
            dmuthetag = temp(i);
            dSigmathetag = temp2(i,i);
            temp3 = Rdiag(i);
            dRthetagi = temp3 * dmuthetag - 1/2 * temp3 * dSigmathetag; 

            dRthetag = zeros(n);
            dRthetag(i,i) = dRthetagi;
            end
%    dBdthetag = -diag(1./nV) * Rdiaginv* dRthetag * Rdiaginv; %*************************
    dBdthetag = -diag(nV) * Rdiaginv* dRthetag * Rdiaginv; %*************************
    dFthetag= trace(-dBdthetag*BBinv)+ trace(IBKinv*dBdthetag* Kf) + bd * dBdthetag * (1./nV) + ...
               (-1)*(Y' * (eye(n) - BB * Kf * IBKinv) * dBdthetag * Kf * IBKinv * BB' * Y ) +...
               (-1)* Y'* BB * Kf*  IBKinv * dBdthetag* Y + ...
               trace(IKDinv * dKthetag * DD) -  trace(IKDinv * dKthetag * DD * IKDinv ) + ...
               ones(n,1)' * (DD - eye(n)) * dKthetag * DD * ones(n,1)  + ...
               1/2 *trace(KDinv * inv(Kg) *  dKthetag * inv(Kg) *  KDinv) + ...
               -1/4* nV' * dKthetag *  nV + ones(n,1)' * dKthetag *(nV/2);
        out2(n+ntheta1+k) = dFthetag/2;
    end
    
    
%<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    % wrt mu0

%            dRmu0 = zeros(nn);
%  for i = 1:n
%     temp4 = Rdiag(i);
%     dRmu0i = temp4; 
%     for j=  (sum(nV(1:i-1))+1):(sum(nV(1:i)))
%     dRmu0(j,j) = dRmu0i;
%     end
%  end
%     
%     dBmu0 = exp(-theta2(end))* A * inv(R)  *  dRmu0 * inv(R)   * A'; %**********************************
%     
%    dFmu0= trace(-BBinv* dBmu0)+ trace(IBKinv*dBmu0* Kf) + bd * dBmu0 * ones(n,1) + ...
%                (-1)*(Y' * (eye(n) - BB * Kf * IBKinv) * dBmu0 * Kf * IBKinv * BB' * Y ) +...
%                (-1)* Y'* BB * Kf*  IBKinv * dBmu0* Y + (nV - ones(n,1))'*ones(n,1);
%     out2(n+ntheta1+ntheta2+1) = dFmu0/2;
%     end


    dRmu0 = zeros(n);
 for i = 1:n
    temp4 = Rdiag(i);
    dRmu0i = temp4; 
    dRmu0(i,i) = dRmu0i;
 end
    
  %  dBmu0 = -diag(1./nV)  *Rdiaginv  *  dRmu0 * Rdiaginv; %**********************************
    dBmu0 = -diag(nV)  *Rdiaginv  *  dRmu0 * Rdiaginv; %**********************************
    
   dFmu0= trace(-BBinv* dBmu0)+ trace(IBKinv*dBmu0* Kf) + bd * dBmu0 * (1./nV) + ...
               (-1)*(Y' * (eye(n) - BB * Kf * IBKinv) * dBmu0 * Kf * IBKinv * BB' * Y ) +...
               (-1)* Y'* BB * Kf*  IBKinv * dBmu0* Y + (nV - ones(n,1))'*ones(n,1);
    out2(n+ntheta1+ntheta2+1) = dFmu0/2;
    end
    
    
% --- Predictions
  %-----------------WW edited ---------------%
 elseif nargin == 8  % add A as input
    [K1ss, K1star] = feval(covfunc1{:}, theta1, X, Xtst);     % test covariance f
    [K2ss, K2star] = feval(covfunc2{:}, theta2, X, Xtst);     % test covariance g
  rf = K1star;
  rdelta = K2star;
  pa = mu0 + (rf)' * Ginv * (Y-repelem(mu0,n)');
  pb = delta0 + (rdelta)'*(diag(Lambda)-0.5.*diag(nV))*repelem(1,n)';  % with replication VBGPpredict 3
  atst = pa; 
  mutst= pb;  
  out1 = atst;                                              % predicted mean  y
 if nargout > 1
  pc = theta1(end)-diag((rf)' * Ginv *rf);
  pd = theta2(end) - diag((rdelta)' * ((Kg+diag(1./Lambda))\eye(n)) * rdelta);
  diagCtst=pc;
  diagSigmatst = pd;
%  out2 = diagCtst + exp(mutst+diagSigmatst/2);              % predicted variance y   = pc + exp(pb+pd/2);; % predicted variance y
   out2 = exp(mutst+diagSigmatst/2);              %  mean of variance log normal   = exp(pb+pd/2)
end
  %-----------------WW edited ---------------%
 end