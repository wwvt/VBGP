function [post nlZ dnlZ] = infGaussLik_ww(hyp, mean, cov, x, y, Vhat,opt)

% Exact inference for a GP with Gaussian likelihood. Compute a parametrization
% of the posterior, the negative log marginal likelihood and its derivatives
% w.r.t. the hyperparameters. See also "help infMethods".
%
% Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch, 2016-10-28.
%                                      File automatically generated using noweb.
%
% See also INFMETHODS.M, APX.M.

if nargin<9, opt = []; end                          % make sure parameter exists
% if iscell(lik), likstr = lik{1}; else likstr = lik; end
% if ~ischar(likstr), likstr = func2str(likstr); end
% if ~strcmp(likstr,'likGauss')               % NOTE: no explicit call to likGauss
%   error('Exact inference only possible with Gaussian likelihood');
% end
 
[n, D] = size(x);
[m,dm] = feval(mean{:}, hyp.mean, x);           % evaluate mean vector and deriv
%sn2 = Vhat; 
W = Vhat;%ones(n,1)./sn2;            % noise variance of likGauss   %WW edited
K = apx_w(hyp,cov,x,opt);                        % set up covariance approximation
[ldB2,solveKiW,dhyp,post.L] = K.fun(W); % obtain functionality depending on W

alpha = solveKiW(y-m);
post.alpha = K.P(alpha);                       % return the posterior parameters
    %[K,dK] = feval(cov{:},hyp.cov,x);     % covariance matrix and dir derivative
%post.sW = sqrt(diag(W)+K);                              % sqrt of noise precision vector
post.sW = sqrt(W);                              % sqrt of noise precision vector

if nargout>1                               % do we want the marginal likelihood?
  nlZ = (y-m)'*alpha/2 + ldB2 + n*log(2*pi)/2;    % -log marginal likelihood
 %    
 %WW edited. Originally, they optimize over hyp, where nlZ and dnlZ.lik have the information about hyp, since sn2 =
 % exp(2*hyp.lik). However, here sn2 = Vhat does not change over hyp. We
 % can only use infGaussLik to optimize the hyp and use infGaussLik_ww to
 % do the prediction. 
  if nargout>2                                         % do we want derivatives?
    dnlZ = dhyp(alpha); dnlZ.mean = -dm(alpha);
  %  dnlZ.lik = -sn2*(alpha'*alpha) - 2*sum(dW)/sn2 + n;   %WW commented
  end
end
