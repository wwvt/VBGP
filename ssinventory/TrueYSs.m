function y_true  = TrueYSs(X,Korder,pbklog,meanD,c,h)

%Created 7/5/2011
%Last update 7/5/2011

% This function is to calculate the true long-run avg. cost per period
% for given (S,s) 
% S = X(:,1);
% s = X(:,2);
% deltaSs = S-s
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
s = X(:,2);
deltaSs= X(:,1); 
y_true = c*meanD +...
              (Korder+h*(s-meanD+...
              1/meanD*deltaSs.*(s+deltaSs/2))+...
              (h+pbklog)*meanD.*exp(-1/meanD*s))./...
              (1+1/meanD*deltaSs);
 

