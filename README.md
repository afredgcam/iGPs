# iGPs
Code implementing integrated Gaussian Processes (iGPs) from the paper: 

'Integrated Gaussian Processes for Tracking', Lydeard, Ahmad, Godsill, 2024

––––––––

Scripts:

'functions.py' contains required functions for the included scripts.
    
'1d_track_ex.py' replicates Figure 1, the two 1D curves from the SE and iSE-2 models.
    
'final_double_swap.py' executes the synthetic data experiment, running iSE-1, iSE-2 and SE on 100 data sets generated from each model type, printing the average RMSE at the end.

––––––––

Folders:

'Data' contains the synthetic data, and the code used to generate it, as well as plots for the first data sets from each model (Figure 4).
    
'Plots' contains inference plots (i.e., of tracker performance) on those first data sets (Figure 5).
