# iSE-1 vs iSE-2 vs SE: double swap

import numpy as np
import matplotlib.pyplot as plt
import functions as f
from tqdm import tqdm


# set seed
np.random.seed(28)

# plot design choices
plt.rcParams["figure.dpi"] = 500
plt.rcParams['text.usetex'] = True

# output choices
num_sets = 100
wanting_ise1 = True
wanting_ise2 = True
wanting_se = True
model_names = ['iSE_1','iSE_2','SE']
wanting_model = [wanting_ise1,wanting_ise2,wanting_se]

# data set generative model
data_models = ['iSE_1','iSE_2','SE']

# collect results
rmse = np.zeros([num_sets,3,3]) # index: set number, data model, inference model





for set_num in tqdm(range(num_sets)):

    for model_num in range(len(data_models)):
        
        # load data
        model = data_models[model_num]
        file = np.load(f'Data/{model}/{set_num}.npz',allow_pickle=True)
        data,truth,VW = list(file['data']),file['track'],file['VW']
        [µ0,µ1,sy,area] = file['hyperparameters']
        motion_pars = file['all_motion_pars']
        Tmax,dt = len(data),1
        
        # correct aspect ratio
        x_dist,y_dist = VW[:,1] - VW[:,0]
        nat_aspect = x_dist / y_dist
        aspect = 5 * np.array([nat_aspect,1])
        plt.rcParams["figure.figsize"] = aspect
        
        # make design / parameter choices (common to all methods)
        assoc_threshold = 5
        d = 10
        t = dt * np.arange(d,0,-1)
    
        # make individual parameter choices
        s2_all = [motion_pars[0],motion_pars[0],motion_pars[2]]
        ell_all = [motion_pars[1],motion_pars[1],motion_pars[3]]
    
        # final objects
        X = np.zeros([len(model_names),Tmax,2])
        S = np.zeros([len(model_names),Tmax])
        associations = -np.ones([len(model_names),Tmax],int)
    
    
    
        ##### inference #####
    
        for i in range(len(model_names)):
            
            # skip model if not wanted
            if not wanting_model[i]:
                continue
            
            # current parameters
            s2,ell = s2_all[i],ell_all[i]
            
            # set up
            if i == 0:
                mk = [truth[0,:]*np.ones([d,2])]
                mk[0][:-1] -= mk[0][-1]
            elif i in [1,2]:
                mk = [truth[0,:]*np.ones([d,2])]
            if i == 0:
                vk = [np.eye(d)]
                vk[0][:-1,:-1] = f.iSE(t[1:],t[1:],s2/10,ell)
                vk[0][-1,-1] = s2/10
            elif i == 1:
                vk = [f.iSE(t,t,s2/10,ell)]
            elif i == 2:
                vk = [f.SE(t,t,s2/10,ell)]
            
            ### do tracking ###
            
            for k in range(Tmax):
                
                # observations
                obs = data[k-Tmax]
                
                # predict step
                if i == 0:
                    m_pred,v_pred = f.ise1_pred(t+dt*(k+1),mk[-1],vk[-1],s2,ell)
                elif i == 1:
                    m_pred,v_pred = f.ise2_pred(t,mk[-1],vk[-1],s2,ell)
                elif i == 2:
                    m_pred,v_pred = f.se_pred(t,mk[-1],vk[-1],s2,ell)
                
                # skip rest if no data
                if len(obs) != 0:
                    
                    # association step
                    if i == 0:
                        ind = f.associate_ise1(obs,m_pred,v_pred,sy,assoc_threshold)
                    else:
                        ind = f.associate(obs,m_pred,v_pred,sy,assoc_threshold)
                    
                    # update step
                    if type(ind) == int:
                        associations[i,k] = ind
                        datum = obs[ind,:]
                        if i == 0:
                            m_up,v_up = f.update_ise1(datum,m_pred,v_pred,sy)
                        else:
                            m_up,v_up = f.update(datum,m_pred,v_pred,sy)
                    else:
                        m_up,v_up = m_pred,v_pred
                
                else:
                    m_up,v_up = m_pred,v_pred
                
                # record results
                mk.append(m_up)
                vk.append(v_up)
                X[i,k,:] = m_up[0,:] + m_up[-1,:] * int(i==0)
                S[i,k] = v_up[0,0] + v_up[-1,-1] * int(i==0)
            
            ###################
            
            # drop initial mean and var
            if i != 3:
                mk.pop(0)
                vk.pop(0)
            
        #####################
    
    
    
    
        ### view results ###
        
        # if set_num == 0:
        #     # plot observations
        #     f.plot_data(data,'#555555',0.2)
        #     # plot tracks
        #     f.plot_track(truth,'goldenrod','-','Truth')
        #     if wanting_ise1:
        #         f.add_track_unc(X[0,:,:],S[0,:],'deepskyblue')
        #         f.plot_track(X[0,:,:],'deepskyblue','--','iSE-1')
        #     if wanting_ise2:
        #         f.add_track_unc(X[1,:,:],S[1,:],'blueviolet')
        #         f.plot_track(X[1,:,:],'blueviolet',(0,(3,2)),'iSE-2')
        #     if wanting_se:
        #         f.add_track_unc(X[2,:,:],S[2,:],'deeppink')
        #         f.plot_track(X[2,:,:],'deeppink',(1,(3,2)),'SE')
        #     # show plot neatly
        #     f.tidy_plot(VW,x_name='',y_name='',legend_loc=4)
    
        # compute RMSEs
        rmse_ise1 = ((X[0,:,:] - truth)**2).sum(1).mean()**0.5
        rmse_ise2 = ((X[1,:,:] - truth)**2).sum(1).mean()**0.5
        rmse_se = ((X[2,:,:] - truth)**2).sum(1).mean()**0.5
        rmse[set_num,model_num,:] = [rmse_ise1,rmse_ise2,rmse_se]

# average RMSE
aRMSE = rmse.mean(0)
print()
print(aRMSE)