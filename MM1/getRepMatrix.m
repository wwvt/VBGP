function [A] = getRepMatrix(replication)

% WW translate from R to MATLAB
% WW edited on 9/22/2017
% Make a matrix for replications 
  n = sum(replication);
  m = length(replication);
  A = zeros(m,n);
  startloc = 0;
  for i = 1:m
    A(i,(startloc+1):(startloc+replication(i))) = 1;
    startloc = startloc + replication(i);
  end