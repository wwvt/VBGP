function model = SKfit_xi(X, Y, B, Vhat, gammaP)
% fit a stochastic kriging model to simulation output
% k - number of design points
% d - dimension of design variables
% X - design points (k x d) 
% Y - (k x 1) vector of simulation output at each design point
% B - (k x b) matrix of basis functions at each design point
%     The first column must be a column of ones!
% Vhat - (k x 1) vector of simulation output variances
% Types of correlation function used to fit surface:
%    gammaP = 1: exponential
%    gammaP = 2: gauss
%    gammaP = 3: cubic
% 
% Examples
%       skriging_model = SKfit(X,Y,ones(k,1),Vhat,2);
% Use SK model with gauss correlation function to fit data, (X,Y,Vhat)
% X is design points, Y and Vhat are outputs at design points 
% (Y is mean, Vhat is intrinsic variance), non-trend estimate B = ones(k,1)

[k d] = size(X);

if not(all(size(Y)==[k 1]))
    error('Output vector and design point matrix must have the same number of rows.')
end
if (size(B,1)~=k)
    error('Basis function and design point matrices must have the same number of rows.')
end
if not(all(B(:,1)==1))
    error('The first column of the basis function matrix must be ones.')
end
% if (not(all(size(Vhat)==[k 1])))
%     error('The variance vector and design point matrix must have the same number of rows.')
% end
if ((gammaP ~= 1)&(gammaP ~= 2)&(gammaP ~= 3))
    error('please type of correlation function: 1 = exponential, 2 = gauss, 3 = cubic.')
end

% Normalize data by scaling each dimension from 0 to 1
minX = min(X);  
maxX = max(X);
% X = (X - repmat(minX,k,1)) ./ repmat(maxX-minX,k,1);

% Normalize data
mX = mean(X);   sX = std(X);
mY = mean(Y);   sY = std(Y);
% 02.08.27: Check for 'missing dimension'
% j = find(sX == 0);
% if  ~isempty(j),  sS(j) = 1; end
% j = find(sY == 0);
% if  ~isempty(j),  sY(j) = 1; end
X = (X - repmat(mX,k,1)) ./ repmat(sX,k,1);
% Y = (Y - repmat(mY,k,1)) ./ repmat(sY,k,1);





% calculate the distance matrix between points (copied from DACE)
% distances are recorded for each dimension separately
ndistX = k*(k-1) / 2;        % max number of non-zero distances
ijdistX = zeros(ndistX, 2);  % initialize matrix with indices
distX = zeros(ndistX, d);    % initialize matrix with distances
temp = 0;
for i = 1 : k-1
    temp = temp(end) + (1 : k-i);
    ijdistX(temp,:) = [repmat(i, k-i, 1) (i+1 : k)']; 
    distX(temp,:) = repmat(X(i,:), k-i, 1) - X(i+1:k,:); 
end
IdistX = sub2ind([k k],ijdistX(:,1),ijdistX(:,2));

% distance matrix raised to the power of gamma
D = zeros(k,k,d);
for p=1:d
    temp = zeros(k);
    if (gammaP == 3)
        temp(IdistX) = abs(distX(:,p));
    else
        temp(IdistX) = -abs(distX(:,p)).^gammaP;
    end
    D(:,:,p) = temp+temp';
end

% diagonal intrinsic variance matrix
      V = diag(Vhat);



% initialize parameters theta, tau2 and beta
% inital extrinsic variance = variance of ordinary regression residuals
betahat = (B'*B)\(B'*Y);
   tau2_0 = var(Y-B*betahat);
%  tau2_0 = var(Y);

% see stochastic kriging tutorial for explanation
% make correlation = 1/2 at average distance
average_distance = mean(abs(distX));
theta_0 = ones(d,1);
% if gammaP == 3
%     theta_0(1:d) = 0.5;
% else
%     theta_0(1:d) = (log(2)/d)*(average_distance.^(-gammaP));
% end

% lower bounds for parameters tau2, theta, and beta included 
% only to satisfy fmincon  
lbtau2 = 0.001*tau2_0;     % naturally 0 
lbtheta = 0.001*ones(d,1); % naturally 0; increase to avoid numerical trouble
lb = [lbtau2;lbtheta];


ubtau2 = 500;     % naturally 0 
ubtheta = 100; % naturally 0; increase to avoid numerical trouble
ub = [ubtau2;ubtheta];
% maximize profile log-likelihood function (-"logPL")
% subject to lower bounds on tau2 and theta
%  myopt = optimset('Display','iter','MaxFunEvals',1000000,'MaxIter',5000);
%    myopt = optimset('Display','iter','MaxFunEvals',1000000,'MaxIter',5000,'Algorithm','sqp');

 myopt = optimset('Display','off','MaxFunEvals',1000000,'MaxIter',5000,...
     'Algorithm','interior-point');
parms = fmincon(@(x) logPL(x,k,d,D,B,V,Y,gammaP),...
        [tau2_0;theta_0],[],[],[],[],lb,ub,[],myopt); 

%     myopt = psoptimset('CompletePoll','off',...
%         'CompleteSearch','off','Display','off',...
%         'PollMethod','GPSPositiveBasis2N','UseParallel',true,...
%         'TolMesh',1e-6, 'TolX', 1e-6);
%      %solve objective function by the fmpincon directly
% 
%     parms = patternsearch(@(x) logPL(x,k,d,D,B,V,Y,gammaP), ...
%          [tau2_0;theta_0],[],[],[],[],lb,ub,[],myopt);
%      
tau2hat = parms(1);
thetahat = parms(2:length(parms));
% MLE of beta is known in closed form given values for tau2, theta
% and is computed below
% calculate estimates of the correlation and covariance matrices
if gammaP == 3
    Rhat = corrcubR(thetahat,D);
else
    Rhat = correxpR(thetahat,D);
end
SigmaM = tau2hat*Rhat;
Sigmahat  = SigmaM + V;
Lhat = chol(Sigmahat)';
Lhatinv = inv(Lhat);
Sigmahatinv = Lhatinv'*Lhatinv;
betahat = (B'*Sigmahatinv*B)\(B'*(Sigmahatinv*Y));
vec_c = B'*Sigmahatinv*B;


% issue warnings related to constraints 
warningtolerance = 0.001;
if min(abs(lbtheta - thetahat)) < warningtolerance
    warning('thetahat was very close to artificial lower bound');
end
if abs(lbtau2 - tau2hat) < warningtolerance
    warning('tau2hat was very close to artificial lower bound');
end

% output MLEs and other things useful in prediction
model.tausquared =  tau2hat;
model.beta = betahat;
model.B =B;
model.theta = thetahat;
model.X = X;
model.minX = minX;
model.maxX = maxX;
model.Xsc =[mX; sX];
model.Ysc =[mY; sY];
model.gamma = gammaP;
model.L = Lhat;
model.Z = Lhat\(Y-B*betahat);
model.bSigma = Sigmahat;
model.bSigmainv = Sigmahatinv;
model.Rhat = Rhat;
model.vec_c = vec_c;
model.LSSVR_L = SigmaM*(Sigmahatinv - Sigmahatinv*ones(k,k)/vec_c*Sigmahatinv)...
                 + ones(k,k)/vec_c*Sigmahatinv;
model.alpha = tau2hat * Sigmahatinv * (Y - B* betahat);           

