# generate two 1D tracks from each of iSE and SE

import numpy as np
import functions as f
import matplotlib.pyplot as plt

# set seed
np.random.seed(28)

# specify plot properties
plt.rcParams["figure.figsize"] = [6,3]
plt.rcParams["figure.dpi"] = 500
plt.rcParams['text.usetex'] = True

# choose design choices
num_1d_tracks = 2
Tmax = 100
d = 10
dt = 1

# choose hyperparameters
s2_ise = 1
ell_ise = 2

s2_se = 1
ell_se = 2

# make tracks
x_ise = f.gen_iSE_track(Tmax,d,s2_ise,ell_ise,num_1d_tracks,dt)
x_se = f.gen_SE_track(Tmax,d,s2_se,ell_se,num_1d_tracks,dt)
tracks = [x_ise,x_se]

# show tracks
names = ['iSE','SE']
colours = [['blueviolet','mediumslateblue'],
           ['deeppink','mediumvioletred']]
styles = ['-','-']
t = dt * np.arange(Tmax)

f.plot_gen_tracks(tracks,names,colours,styles,t)