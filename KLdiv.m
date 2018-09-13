function [KLd] = KLdiv(Mtest, Mref, Stest, Sref)
% WW coded 
%  F3 = -KLdiv(mu,repelem(delta0,n)',Omega,Kdelta); % false

% modified from the R code from rags2ridges package
% KLd <- (sum(diag(solve(Stest) %*% Sref)) +
%               t(Mtest - Mref) %*% solve(Stest) %*% (Mtest - Mref) -
%               nrow(Sref) - log(det(Sref)) + log(det(Stest)))/2
%   ##############################################################################
%   # - Function that calculates the Kullback-Leibler divergence between two
%   #   normal distributions
%   # - Mtest     > mean vector approximating m.v. normal distribution
%   # - Mref      > mean vector 'true'/reference m.v. normal distribution
%   # - Stest     > covariance matrix approximating m.v. normal distribution
%   # - Sref      > covariance matrix 'true'/reference m.v. normal distribution
%   # - symmetric > logical indicating if original symmetric version of KL div.
%   #               should be calculated
%   ##############################################################################

 % # Dependencies
 % # require("base")

 % Evaluate KL divergence
 [n D] = size(Sref);
 KLd = (sum(diag(((Sref)\eye(n)) * Stest)) +  (Mtest - Mref)' * ((Sref)\eye(n)) * (Mtest - Mref) - length(Sref) - log(det(Stest)) + log(det(Sref)))/2;

