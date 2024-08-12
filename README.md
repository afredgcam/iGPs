# iGPs
Code implementing integrated Gaussian Processes (iGPs) from the paper: 'Integrated Gaussian Processes for Tracking', Goodyer, Godsill, 2024

Scripts:
'functions.py' contains required functions for the included scripts.
'1d_track_ex.py' replicates Figure 1, the two 1D curves from the SE model and the second iSE model.
'inference.py' executes the SE and second iSE models, producing a figure like Figure 2 (but on the synthetic data in the 'Data' folder) and saves the filtering estimates and variances in the 'Results' folder.

Folders:
'Data' contains the synthetic data 'written.npz', its plot 'written.png', and a script to reproduce both 'written_data.py'.
'Results' starts empty but contains filtering results when 'inference.py' is run.
