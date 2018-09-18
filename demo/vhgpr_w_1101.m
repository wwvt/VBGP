function [out1, out2, mutst, diagSigmatst, atst, diagCtst]= vhgpr_w_1101(LambdaTheta, covfunc1, covfunc2, fixhyp, X, y, A, Xtst)
% Last update: 9/18/2018 
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
end
 
% Parameter initialization
ntheta1 = eval(feval(covfunc1{:}));
ntheta2 = eval(feval(covfunc2{:}));

nV = sum(A,2); % # replications need to be calculated within the function
nn = sum(nV); % total budget, i.e. total number of replications
if n+ntheta1+ntheta2+1 ~= size(LambdaTheta,1),error('Incorrect number of parameters');end
Lambda = exp(LambdaTheta(1:n));
theta1 = LambdaTheta(n+1:n+ntheta1);
theta2 = LambdaTheta(n+ntheta1+1:n+ntheta1+ntheta2);
delta0 = LambdaTheta(n+ntheta1+ntheta2+1);

mu0 = 0; % beta0
Kf = feval(covfunc1{:},theta1, X);
Kg = feval(covfunc2{:},theta2, X);

   DD = diag(Lambda);

sLambda = sqrt(Lambda);
cinvB = chol(eye(n) + Kg.*(sLambda*sLambda'))'\eye(n);   % O(n^3)
cinvBs = cinvB.*(ones(n,1)*sLambda');
hBLK2 = cinvBs*Kg;                                         % O(n^3)
Omega = Kg - hBLK2'*hBLK2;           % same as Omega=inv(inv(Kg)+DD); 
IKDinv = cinvB' * cinvB; % same as IKDinv = (eye(n) + Kg * DD)\eye(n);
%Omega = Kg * IKDinv; % same as KDinv = ((Kg\eye(n)) + DD)\eye(n) = Omega;
   
mu = Kg*(  DD*ones(n,1)-0.5*nV)+ ones(n,1).*delta0; %ones(n,1) * 0; 
  
  Rdiag = exp(mu-diag(Omega)/2); 
  R = diag(repelem(Rdiag,nV));
  Rdiaginv = diag(1./(Rdiag));
  Rinv = diag(1./repelem(Rdiag,nV));  
 % M = A'* Kf *A + R;
  Phi = diag(Rdiag./nV); %Sigma_varepsilon'
  G = Kf+Phi;
  Ginv = (Kf+Phi)\eye(n);
  MlogDet = sum((mu-diag(Omega)./2).*(nV-1) + log(nV)) + log(det(G));  
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

%     F1 = -0.5 *(nn*log(2*pi)+MlogDet+ sum((yy).^2'./repelem(Rdiag,nV)) - Y' * diag(nV./Rdiag) * Kf * Ginv * Y);% O(m3)
  Minv = Rinv - Rinv * A' * Kf * Ginv * Phi * A * Rinv;
  F1 = -0.5*(nn*log(2*pi)+MlogDet+(yy-mu0) * Minv * (yy-mu0)'); % O(n3) same.
  F2 = -0.25 .* nV' * diag(Omega);
  F3 = -KLdiv(mu,repelem(delta0,n)',Omega,Kg);  %*************************** KLdiv definition corrected

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


     dRD = zeros(n);

      for i = 1:n
            dDD = zeros(n);
            dDD(i,i) = 1;
            temp = Kg * dDD * ones(n,1); % d mu/d D_{ii}
            temp2  = -Omega * dDD * Omega;
            
            for j =1:n
            dmuD = temp(j); % d mu_j / d D_{ii}
            dSigmaD = temp2(j,j);
            temp3 = Rdiag(j);
            dRDi = temp3 * dmuD - 1/2 * temp3 * dSigmaD; 
            dRD(j,j) = dRDi ;
            end
                   
             dBDtemp=(-diag(nV) * Rdiaginv * dRD * Rdiaginv);
             dBD  = dBDtemp;  %**********************
%           dBD  = -diag(1./nV) * Rdiaginv *  dRD * Rdiaginv;  %**********************

             dFLambda(i)=  trace(-dBD*BBinv)+ trace(IBKinv*dBD* Kf) +  ... 
                bd * dBD * (1./nV) +(-1)*(Y' * (eye(n) - BB * Kf * IBKinv) * dBD * Kf * IBKinv * BB' * Y ) +...
                (-1)* Y'* BB * Kf*  IBKinv * dBD* Y   + ...
                           trace(Omega * dDD) -  trace(Omega * dDD * Omega * inv(Kg) ) + ...
                           trace((Kg * DD * ones(n,1)*ones(n,1)'+ ones(n,1)*ones(n,1)'*DD * Kg -ones(n,1)*ones(n,1)'*Kg)*dDD)- 1/2 *trace(Omega * dDD *  Omega);
      end
    out2(1:n) = Lambda.*dFLambda'/2;   %**************************
                    %  

 
 %<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    if fixhyp < 1       % fixhyp = 0, do this. 
    % wrt Kf hyperparameters
            for k = 1:ntheta1
             dKthetaf  = feval(covfunc1{:}, theta1, X, k);
             dFthetaf = trace(IBKinv * BB*dKthetaf) - (Y' * inv((BBinv+Kf)) * dKthetaf * inv((BBinv+Kf)) * Y ); %same
             out2(n+k) = dFthetaf/2;
            end
    end
    
 
    if fixhyp < 2         %  fixhyp = 0, do this
    % wrt Kg hyperparameters
    for k = 1:ntheta2    
    dKthetag = feval(covfunc2{:}, theta2, X, k); 
    temp = dKthetag * (DD * ones(n,1)-0.5*nV);  %*********************
    temp2  = IKDinv * dKthetag * IKDinv; %theoretically same, numerically different
       
   
       dRthetag = zeros(n);
            for i = 1:n
            dmuthetag = temp(i);
            dSigmathetag = temp2(i,i);
            temp3 = Rdiag(i);
            dRthetagi = temp3 * dmuthetag - 1/2 * temp3 * dSigmathetag; 

            dRthetag(i,i) = dRthetagi;
            end
    dBdthetag = -diag(nV) * Rdiaginv * dRthetag * Rdiaginv; %*************************
    dFthetag= trace(-dBdthetag*BBinv)+ trace(IBKinv*dBdthetag* Kf) + bd * dBdthetag * (1./nV) + ...
               (-1)*(Y' * (eye(n) - BB * Kf * IBKinv) * dBdthetag * Kf * IBKinv * BB' * Y ) +...
               (-1)* Y'* BB * Kf*  IBKinv * dBdthetag* Y + ...
               trace(IKDinv * dKthetag * DD) -  trace(IKDinv * dKthetag * DD * IKDinv ) + ...
               ones(n,1)' * (DD - eye(n)) * dKthetag * DD * ones(n,1)  + ...
               1/2 *trace(Omega * inv(Kg) *  dKthetag * inv(Kg) *  Omega) + ...
               -1/4* nV' * dKthetag *  nV + ones(n,1)' * dKthetag *(nV/2);
        out2(n+ntheta1+k) = dFthetag/2;
    end
    
    
%<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    % wrt mu0


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
  out2 = diagCtst + exp(mutst+diagSigmatst/2);              % predicted variance y   = pc + exp(pb+pd/2);; % predicted variance y
  % out2 = exp(mutst+diagSigmatst/2);              %  mean of variance log normal   = exp(pb+pd/2)
end
  %-----------------WW edited ---------------%
 end
