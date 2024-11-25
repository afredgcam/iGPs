# iGPs NOT CORRECT
Code implementing integrated Gaussian Processes (iGPs) from the paper: 

'Integrated Gaussian Processes for Tracking', Goodyer, Godsill, 2024

––––––––

Scripts:

'functions.py' contains required functions for the included scripts.
    
'1d_track_ex.py' replicates Figure 1, the two 1D curves from the SE model and the second iSE model.
    
'inference.py' executes the SE model with the first Markovian assumption and the iSE model with the second. It produces Figure 3, on the synthetic data provided in the 'Data' folder, and saves the filtering estimates and variances in the 'Results' folder.

––––––––

Folders:

'Data' contains the synthetic data 'written.npz', its plot 'written.png', and a script to reproduce both: 'written_data.py'.
    
'Results' starts with a placeholder document but contains filtering results when 'inference.py' is run.
