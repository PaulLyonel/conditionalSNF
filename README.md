# Stochastic Normalizing Flows for Inverse Problems: a Markov Chains Viewpoint

This is the code belonging to the research paper "Stochastic Normalizing Flows for Inverse Problems: a Markov Chains Viewpoint" [1] available at https://arxiv.org/abs/2109.11375.  If you find this code useful for your research, please cite the paper.  

The repository implements conditional stochastic normalizing flows (SNFs) as introduced in [1]. Conditional SNFs extend the model of SNFs [4] for inverse problems.  

The forward model for the scatterometry example (Section 7.2 in [1] and script `Main_scatterometry.py`) was kindly provided by Sebastian Heidenreich from PTB (Physikalisch-Technische Bundesanstalt). 
For more information on this particular inverse problem, we refer the reader to the papers [2] and [3].
Furthermore, if you use the forward model of the scatterometry inverse problem, please cite the papers [2] and [3]. 

If you have any questions or bug reports, feel free to contact Paul Hagemann (hagemann(at)math.tu-berlin.de) or Johannes Hertrich (j.hertrich(at)math.tu-berlin.de).

## REQUIREMENTS

The code was written in Python using PyTorch and the FrEIA framework (https://github.com/VLL-HD/FrEIA), which implements several invertible neural network architectures. We tested the code using the following Python packages and versions:

1. PyTorch 1.9.0  
2. Numpy 1.20.3
3. Scipy 1.6.2
4. tqdm 4.62.1
5. Matplotlib 3.4.2
6. pot 0.7.0
7. FrEIA 0.2

Usually the code is also compatible with some other versions of the corresponding Python packages.

## USAGE

The code is structured as follows:  
The script `Main_mixture.py` reproduces the mixture model example from Section 7.1 of the paper, `Main_scatterometry.py` reproduces the scatterometry example from Section 7.2.  

The directory `core` contains the INN and SNF model. The directory `utils` contains different functions needed for evaluating the densities or sampling for the particular inverse problem. 


## REFERNCES

[1] P. Hagemann, J. Hertrich and G. Steidl.  
Stochastic Normalizing Flows for Inverse Problems: a Markov Chains Viewpoint.  
ArXiv preprint arXiv:2109.11375, 2021.  

[2] S. Heidenreich, H. Gross, and M. Bär.  
Bayesian approach to the statistical inverse problem of scatterometry: Comparison of three surrogate models.  
International Journal for Uncertainty Quantification, 5(6), 2015.  

[3] S. Heidenreich, H. Gross, and M. Bär.  
Bayesian approach to determine critical dimensions from scatterometric measurements.  
Metrologia, 55(6):S201, Dec. 2018.

[4] H. Wu, J. Köhler, and F. Noé.  
Stochastic normalizing flows.  
In Advances in Neural Information Processing Systems, 2020.
