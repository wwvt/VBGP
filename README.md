# VBGP Code

# Introduction
This Github repository provides the code for the paper "A Variational Inference-based Heteroscedastic Gaussian Process Approach for Simulation Metamodeling". 

Before starting, please see `VBGP/demo/demo_VBGP_w.m ` for a demo example using VBGP.

# Acknowlegement

The algorithm in this file is based on the following paper: 
W. Wang, X. Chen, N. Chen and L. Yang, "A Variational Inference-based Heteroscedastic Gaussian Process Approach for Simulation Metamodeling".

- The code in this file is based on the MATLAB package given by 
http://www.tsc.uc3m.es/~miguel/downloads.php, see code for "Variational Heteroscedastic Gaussian Process Regression", Code and demo for VHGP, including a volatility example. Many thanks to M. Lazaro Gredilla and M. Titsias for their work  "Variational Heteroscedastic Gaussian Process Regression" published in ICML 2011.

- Some of the covariance functions are based on GPML Matlab Code version 4.2 http://www.gaussianprocess.org/gpml/code/matlab/doc/ that originally demonstrated the main algorithms from Rasmussen and Williams: Gaussian Processes for Machine Learning.

# How To Use

  - A sample parameter setting can be found in `VBGP/demo/demo_VBGP_w.m `
  - Training and prediction is done within `VBGP/demo/vbgp_w_1101.m `  
  
Necessary steps and comments can be found there.

You can also:
  - Use your own parameter setting and fix them at the given initial values.
  - Use `fmincon` for derivative-free optimization in finding parameters.
  - 
About data structure:
  - `y(l).n` MATLAB data structure is recommended for the use of VBGP.
  - If your data output is a matrix, 

# Help
If you have questions, you can contact me
