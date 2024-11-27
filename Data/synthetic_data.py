# generate 2D tracks

import numpy as np
import functions as f
from scipy.stats import norm,poisson
import matplotlib.pyplot as plt

# set seed
np.random.seed(28)

# specify plot properties
plt.rcParams["figure.figsize"] = [6,3]
plt.rcParams["figure.dpi"] = 500
plt.rcParams['text.usetex'] = True

# output options
saving = True
num_sets = 100

# choose design choices
dims = 2
Tmax = 100
d = 10
dt = 1
wanting_ise1 = True

# choose hyperparameters
s2_ise = 1
ell_ise = 2

s2_se = 100
ell_se = 7

µ1 = 5
sy = 0.3
µ0 = 5

for set_num in range(num_sets):

    # make tracks
    if wanting_ise1: # match X* distribution of iSE to SE using (0,s2_se)
        x_ise = f.gen_iSE1_track(Tmax,d,s2_ise,ell_ise,0,s2_se,dims,dt)
        # x_ise = np.array([np.arange(Tmax),np.arange(Tmax)]).T/5
    else:
        x_ise = f.gen_iSE2_track(Tmax,d,s2_ise,ell_ise,0,s2_se,dims,dt)
    x_se = f.gen_SE_track(Tmax,d,s2_se,ell_se,dims,dt)
    
    # viewing window for iSE
    padding = 0.2
    VW_ise = np.array([x_ise.min(0), x_ise.max(0)]).T
    VW_ise[:,1] = VW_ise[:,0] + (1 + padding) * (VW_ise[:,1] - VW_ise[:,0])
    VW_ise[:,0] = VW_ise[:,1] - (1 + 2*padding)/(1 + padding) * (VW_ise[:,1] - VW_ise[:,0])
    dists_ise = VW_ise[:,1] - VW_ise[:,0]
    nat_aspect = dists_ise[0] / dists_ise[1]
    aspect_ise = 5 * np.array([nat_aspect,1])
    area_ise = (VW_ise[:,1] - VW_ise[:,0]).prod()
    µ0_ise = µ0
    
    # viewing window for SE
    VW_se = np.array([x_se.min(0), x_se.max(0)]).T
    VW_se[:,1] = VW_se[:,0] + (1 + padding) * (VW_se[:,1] - VW_se[:,0])
    VW_se[:,0] = VW_se[:,1] - (1 + 2*padding)/(1 + padding) * (VW_se[:,1] - VW_se[:,0])
    dists_se = VW_se[:,1] - VW_se[:,0]
    nat_aspect = dists_se[0] / dists_se[1]
    aspect_se = 5 * np.array([nat_aspect,1])
    area_se = (VW_se[:,1] - VW_se[:,0]).prod()
    µ0_se = µ0
    
    # make data for iSE
    associations_ise,data_ise = [],[]
    for k in range(Tmax):
        n0k,n1k = poisson.rvs(µ0_ise),poisson.rvs(µ1)
        associations_ise.append(np.append(np.zeros(n0k),np.ones(n1k)))
        data_0k = dists_ise * np.random.random_sample([n0k,2]) + VW_ise[:,0]
        data_1k = norm.rvs(x_ise[k,:],sy**0.5,size=[n1k,2])
        data_ise.append(np.vstack([data_0k,data_1k]))
    
    # make data for SE
    associations_se,data_se = [],[]
    for k in range(Tmax):
        n0k,n1k = poisson.rvs(µ0_se),poisson.rvs(µ1)
        associations_se.append(np.append(np.zeros(n0k),np.ones(n1k)))
        data_0k = dists_se * np.random.random_sample([n0k,2]) + VW_se[:,0]
        data_1k = norm.rvs(x_se[k,:],sy**0.5,size=[n1k,2])
        data_se.append(np.vstack([data_0k,data_1k]))
    
    # plot the first one as an example
    if set_num == 0:
        # plot iSE
        plt.rcParams["figure.figsize"] = aspect_ise
        f.plot_data(data_ise,'k',0.3)
        [name,col] = ['iSE-1','deepskyblue'] if wanting_ise1 else ['iSE-2','blueviolet']
        f.plot_track(x_ise,col,'-',name)
        f.tidy_plot(VW_ise,x_name='',y_name='',legend_loc=1)
        
        # plot SE
        plt.rcParams["figure.figsize"] = aspect_se
        f.plot_data(data_se,'k',0.3)
        f.plot_track(x_se,'deeppink','-','SE')
        f.tidy_plot(VW_se,x_name='',y_name='',legend_loc=1)
    
    # save all (except plots)
    if saving:
        all_motion_pars = np.array([s2_ise,ell_ise,s2_se,ell_se])
        # iSE
        save_name = f'iSE_1/{set_num}.npz' if wanting_ise1 else f'iSE_2/{set_num}.npz'
        obs_pars = np.array([µ0_ise,µ1,sy,area_ise])
        np.savez(save_name,data=np.array(data_ise,object),track=x_ise,
                 hyperparameters=obs_pars,all_motion_pars=all_motion_pars,
                 associations=np.array(associations_ise,object),VW=VW_ise)
        # SE –– generated alongside iSE_2 data, for reproducibility
        if not wanting_ise1:
            obs_pars = np.array([µ0_se,µ1,sy,area_se])
            np.savez(f'SE/{set_num}.npz',data=np.array(data_se,object),track=x_se,
                     hyperparameters=obs_pars,all_motion_pars=all_motion_pars,
                     associations=np.array(associations_se,object),VW=VW_se)