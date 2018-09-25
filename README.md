# VBGP Code

# Introduction
This Github repository provides the code for the paper "A Variational Inference-based Heteroscedastic Gaussian Process Approach for Simulation Metamodeling". 

Before starting, please see `VBGP/demo/demo_VBGP_w.m ` for a demo example using VBGP.

# Acknowlegement

The algorithm in this file is based on the following paper: 
W. Wang, N. Chen, X. Chen, and L. Yang, "A Variational Inference-based Heteroscedastic Gaussian Process Approach for Simulation Metamodeling".

- The code in this file is based on [VHGP MATLAB package](http://www.tsc.uc3m.es/~miguel/downloads.php) provided by M. Lazaro Gredilla and M. Titsias, and [GPML Matlab Code version 4.2](http://www.gaussianprocess.org/gpml/code/matlab/doc/) provided by Rasmussen and Williams: Gaussian Processes for Machine Learning.

# How To Use

  - A sample parameter setting can be found in `VBGP/demo/demo_VBGP_w.m `
  - Training and prediction is done within `VBGP/demo/vbgp_w_1101.m `  
  - **Necessary steps and comments can be found there.**

You can also:
  - Use your own parameter setting and fix them at the given initial values.
  - Use `fmincon` for derivative-free optimization in finding parameters.

About data structure:
  - `y(l).n` MATLAB data structure is recommended for the use of VBGP.
  - If your data output is a matrix, use the following to convert into MATLAB data structure:
  ```MATLAB
  Mreps = 1 % if there is macro-replications for your Y output
  for m = 1:k
    y(m,Mreps).n = zeros([1 n(m)]); % n is a vector for number of replications at each design pt
  end

for i = 1:k
    y(i,Mreps).n = Y(i,:);      % convert into MATLAB data structure
end
  ```

# Help
If you have questions, you can contact me [wenjing at vt dot edu]
