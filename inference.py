# Nearest Neighbour Kalman Filter for iSE vs SE

import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy as dc
import functions as f


# set seed
np.random.seed(28)

# plot design choices
plt.rcParams["figure.dpi"] = 500
plt.rcParams['text.usetex'] = True

# choose desired plots
wanting_final_plot = True
wanting_plot_every_step = False

# load data
file = np.load('Data/written.npz',allow_pickle=True)
data,truth,VW = list(file['data']),file['track'],file['VW']
[µ0,µ1,sy,clutter_intensity,area] = file['hyperparameters']
Tmax,dt = len(data),1

# correct aspect ratio
x_dist,y_dist = VW[:,1] - VW[:,0]
nat_aspect = x_dist / y_dist
aspect = 5 * np.array([nat_aspect,1])
plt.rcParams["figure.figsize"] = aspect

# make design choices
assoc_threshold = 5
d_se,d_ise = 5,10
d = max(d_se,d_ise)

# choose parameters
sig2_se = 1000
ell_se = 3
sig2_mo_se = sig2_se / 100

sig2_ise = 100
ell_ise = 1
sig2_mo_ise = sig2_ise / 10


# initialise required objects (using ground truth)
t_se = dt * np.arange(d_se,0,-1)
mk_se = [truth[0,:]*np.ones([d_se,2])]
vk_se = [f.SE(t_se,t_se,sig2_mo_se,ell_se)]
X_se,S_se = np.zeros([Tmax+d_se,2]),np.zeros(Tmax+d_se)
X_se[:d_se,:],S_se[:d_se] = dc(truth[0,:]*np.ones([d_se,2])),1e-4
associations_se = -np.ones(Tmax,int)

t_ise = dt * np.arange(d_ise,0,-1)
mk_ise = [truth[0,:]*np.ones([d_ise,2])]
vk_ise = [f.iSE(t_ise,t_ise,sig2_mo_ise,ell_ise)]
X_ise,S_ise = np.zeros([Tmax+d_ise,2]),np.zeros(Tmax+d_ise)
X_ise[:d_ise,:],S_ise[:d_ise] = dc(truth[0,:]*np.ones([d_ise,2])),1e-4
associations_ise = -np.ones(Tmax,int)



### do tracking ###

# for k in range(d,Tmax):
for k in range(Tmax):
    
    # get observations for this time step
    obs = data[k]
    
    # do predict step
    m_pred_se,v_pred_se = f.predict_SE(t_se,mk_se[-1],vk_se[-1],sig2_se,ell_se)
    m_pred_ise,v_pred_ise = f.predict_iSE(t_ise,mk_ise[-1],vk_ise[-1],sig2_ise,
                                          ell_ise)
    
    # skip rest if no data
    if len(obs) != 0:
        
        # consider associating a datum
        ind_se = f.associate(obs,m_pred_se,v_pred_se,sy,assoc_threshold)
        ind_ise = f.associate(obs,m_pred_ise,v_pred_ise,sy,assoc_threshold)
        
        # do update step: SE
        if type(ind_se) == int:
            associations_se[k] = ind_se
            datum_se = obs[ind_se,:]
            m_up_se,v_up_se = f.update(datum_se,m_pred_se,v_pred_se,sy)
        else:
            m_up_se,v_up_se = m_pred_se,v_pred_se
        
        # do update step: iSE
        if type(ind_ise) == int:
            associations_ise[k] = ind_ise
            datum_ise = obs[ind_ise,:]
            m_up_ise,v_up_ise = f.update(datum_ise,m_pred_ise,v_pred_ise,sy)   
        else:
            m_up_ise,v_up_ise = m_pred_ise,v_pred_ise
            
    else:
        m_up_se,v_up_se = m_pred_se,v_pred_se
        m_up_ise,v_up_ise = m_pred_ise,v_pred_ise
    
    
    
    # plotting @ k
    if wanting_plot_every_step:
        
        # data
        f.plot_data(data,'k',0.3,[k])
        
        # uncertainties
        f.add_uncertainty(m_up_se[0,:],v_up_se[0,0],'deeppink')
        f.add_uncertainty(m_up_ise[0,:],v_up_ise[0,0],'blueviolet')
        
        # ground truth
        f.plot_track(truth[max(0,k-d+1):k+1,:],'goldenrod','-','Truth')
        
        # tracks
        f.plot_track(m_up_se,'deeppink',(4,(5,1)),'SE',first_is_last=True)
        f.plot_track(m_up_ise,'blueviolet',(2,(5,1)),'iSE',first_is_last=True)
        
        # show
        f.tidy_plot(VW,title=r'$k$ = '+str(k))
    
    
    
    # record results
    mk_se.append(m_up_se)
    vk_se.append(v_up_se)
    X_se[k+d_se,:] = m_up_se[0,:]
    S_se[k+d_se] = v_up_se[0,0]
    
    mk_ise.append(m_up_ise)
    vk_ise.append(v_up_ise)
    X_ise[k+d_ise,:] = m_up_ise[0,:]
    S_ise[k+d_ise] = v_up_ise[0,0]



### save all results ###

np.savez('Results/SE.npz',m=np.array(mk_se,object),v=np.array(vk_se,object),
         X=X_se,S=S_se)
np.savez('Results/iSE.npz',m=np.array(mk_ise,object),v=np.array(vk_ise,object),
         X=X_ise,S=S_ise)

########################


### final plot / results ###

if wanting_final_plot:
    
    # data
    f.plot_data(data,'k',0.2)
    
    # ground truth
    f.plot_track(truth,'goldenrod','-','Truth')
    
    # tracks
    f.plot_track(X_se,'deeppink',(1,(3,2)),'SE')
    f.plot_track(X_ise,'blueviolet',(0,(3,2)),'iSE')
    
    # show
    f.tidy_plot(VW,title='Full Tracks')
    
    # compute RMSEs
    rmse_se = ((X_se[d_se:,:] - truth)**2).sum(1).mean()**0.5
    rmse_ise = ((X_ise[d_ise:,:] - truth)**2).sum(1).mean()**0.5
    print(f'SE RMSE = {rmse_se}')
    print(f'\niSE RMSE = {rmse_ise}')

############################