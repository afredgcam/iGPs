# functions

import numpy as np
from scipy.stats import norm
from scipy.linalg import solve, cholesky
import matplotlib.pyplot as plt
from matplotlib.patches import Circle,PathPatch
from matplotlib.path import Path
from copy import deepcopy as dc



# SE function
def SE(v,w,s2,l):
    
    # make vector / number inputs all valid to use
    v = np.array([v]).reshape([np.max(np.array([v]).shape),1])
    w = np.array([w]).reshape([1,np.max(np.array([w]).shape)])
    
    if l>1e-100:
        return s2 * np.exp(-0.5*((v-w)/l)**2)
    else:
        # if l too small, avoid instability
        plan = np.zeros([v.shape[0],w.shape[1]])
        for i in range(v.shape[0]):
            for j in range(w.shape[1]):
                if v[i,0]==w[0,j]:
                    plan[i,j] = 1
        
        return s2 * plan



# integral of Gaussian cdf
def Xi(v,w,l):
    
    # make vector / number inputs all valid to use
    v = np.array([v]).reshape([np.max(np.array([v]).shape),1])
    w = np.array([w]).reshape([1,np.max(np.array([w]).shape)])
    
    # compute useful sizes
    n = v.shape[0]
    m = w.shape[1]
    
    if l>1e-100:
        
        # compute sufficient n x m matrices for vectorised computations
        v_sq = v * np.ones([1,m])
        w_sq = w * np.ones([n,1])
        dif_sq = v_sq - w_sq
        
        # combine to make kernel formula
        K = dif_sq*norm.cdf(dif_sq/l) + l**2 * norm.pdf(dif_sq,0,l)
        K *= (2*np.pi)**0.5 * l
        
        return K
    
    else:
        
        # reject if l too small
        print('\n l may be too small to compute this matrix \n')
        
        return np.zeros([n,m])



# iSE function
def iSE(v,w,s2,l):
    
    # make vector / number inputs all valid to use
    v = np.array([v]).reshape([np.max(np.array([v]).shape),1])
    w = np.array([w]).reshape([1,np.max(np.array([w]).shape)])
    
    if l>1e-100:
        
        # make 't0' vector for each of v and w
        v0,w0 = np.zeros(v.shape),np.zeros(w.shape)
        # compute K in terms of Xi –– √(2π)𝓁 factor included in Xi function
        K = Xi(v,w0,l) + Xi(v0,w,l) - Xi(v,w,l) - l**2
        
        return s2 * K
    
    else:
        # reject if l too small
        return None



#####################
# final predict steps
#####################

def ise1_pred(t,m,v,s2,l):
    
    # get full prior covariance
    C = iSE(t,t,s2,l)
    
    # compute Fk
    d = m.shape[0]
    ftw = solve(C[1:,1:],C[1:,0])
    F = np.eye(d-1,k=-1)
    F[0,:] = ftw
    F_aug = np.eye(d)
    F_aug[:-1,:-1] = F
    
    # compute Pk
    ptw = C[0,0] - (C[0,1:] * ftw).sum()
    P = np.zeros([d,d])
    P[0,0] = ptw
    
    # compute predict parameters
    m_pred = F_aug @ m
    v_pred = F_aug @ v @ F_aug.T + P
    
    return m_pred,v_pred

def ise2_pred(t,m,v,s2,l):
    
    # get full prior covariance
    C = iSE(t,t,s2,l)
    
    # compute Gk
    d = m.shape[0]
    g = solve(C[1:,1:],C[1:,0])
    G = np.eye(d,k=-1)
    G[0,:-1] = g
    G[0,-1] = 1 - g.sum()
    
    # compute Qk
    q = C[0,0] - (C[0,1:] * g).sum()
    Q = np.zeros([d,d])
    Q[0,0] = q
    
    # compute predict parameters
    m_pred = G @ m
    v_pred = G @ v @ G.T + Q
    
    return m_pred,v_pred

def se_pred(t,m,v,s2,l):
    
    # get full prior covariance
    C = SE(t,t,s2,l)
    
    # compute Fk
    d = m.shape[0]
    ftw = solve(C[1:,1:],C[1:,0])
    F = np.eye(d,k=-1)
    F[0,:-1] = ftw
    
    # compute Pk
    ptw = C[0,0] - (C[0,1:] * ftw).sum()
    P = np.zeros([d,d])
    P[0,0] = ptw
    
    # compute predict parameters
    m_pred = F @ m
    v_pred = F @ v @ F.T + P
    
    return m_pred,v_pred

#####################
#####################
#####################



# do predict step for iSE
def predict_iSE(t,m,v,s2,l,using_SE=False):
    
    # get full prior covariance
    C = SE(t,t,s2,l) if using_SE else iSE(t,t,s2,l)
    
    # compute Gk
    d = m.shape[0]
    g = solve(C[1:,1:],C[1:,0])
    G = np.eye(d,k=-1)
    G[0,:-1] = g.T
    G[0,-1] = 1 - g.sum()
    
    # compute Qk
    q = C[0,0] - (C[0,1:] * g).sum()
    Q = np.zeros([d,d])
    Q[0,0] = q
    
    # compute predict parameters
    m_pred = G @ m
    v_pred = G @ v @ G.T + Q
    
    return m_pred,v_pred



# do predict step for SE
def predict_SE(t,m,v,s2,l):
    
    # get full prior covariance
    C = SE(t,t,s2,l)
    
    # compute Fk
    d = m.shape[0]
    ftw = solve(C[1:,1:],C[1:,0])
    F = np.eye(d,k=-1)
    F[0,:-1] = ftw.T
    
    # compute Qk
    qtw = C[0,0] - (C[0,1:] * ftw).sum()
    Q = np.zeros([d,d])
    Q[0,0] = qtw
    
    # compute predict parameters
    m_pred = F @ m
    v_pred = F @ v @ F.T + Q
    
    return m_pred,v_pred



# consider associating a datum
def associate(data,m,v,sy,threshold):
    
    # compute Mahalanobis distances
    dists = ((data - m[0,:])**2).sum(1)**0.5 / (v[0,0] + sy)**0.5
    
    # associate?
    ind = np.where(dists == dists.min())[0][0]
    if dists[ind] > threshold:
        # nothing close ==> no association
        return None
    else:
        # something close ==> associate 'ind'
        return int(ind)


# association for ise1
def associate_ise1(data,m,v,sy,threshold):
    
    # compute Mahalanobis distances
    dists = ((data - m[0,:] - m[-1,:])**2).sum(1)**0.5
    dists /= (v[0,0] + 2*v[0,-1] + v[-1,-1] + sy)**0.5
    
    # associate?
    ind = np.where(dists == dists.min())[0][0]
    if dists[ind] > threshold:
        # nothing close ==> no association
        return None
    else:
        # something close ==> associate 'ind'
        return int(ind)



# update by the association
def update(datum,m,v,sy):
    
    # get innovation
    y_in = (datum - m[0,:]).reshape([1,-1])
    
    # get optimal kalman gain
    Kgain = v[:,0].reshape([-1,1]) / (v[0,0] + sy)
    
    # compute updated parameters
    m_up = m + Kgain @ y_in
    v_up = v - Kgain @ v[0,:].reshape([1,-1])
    
    return m_up,v_up


# update ise1 by association
def update_ise1(datum,m,v,sy):
    
    # get innovation
    y_in = (datum - m[0,:] - m[-1,:]).reshape([1,-1])
    
    # kalman gain
    Kgain = v[:,[0,-1]].sum(1).reshape([-1,1]) / (v[0,0]+2*v[0,-1]+v[-1,-1]+sy)
    
    # updated parameters
    m_up = m + Kgain @ y_in
    v_up = v - Kgain @ v[[0,-1],:].sum(0).reshape([1,-1])
    
    return m_up,v_up








##### plotting #####


# plot all data 
def plot_data(data,colour,opacity=1,steps=None,indices=[]):
    
    # get data at correct times
    if steps == None:
        steps = range(len(data))
    
    # plot the data
    name = 'Obs.'
    if indices:
        for k in steps:
            plt.plot(data[k][indices,0],data[k][indices,1],c=colour,ls='',
                     marker='2',label=name,alpha=opacity)
            name = ''
    else:
        for k in steps:
            plt.plot(data[k][:,0],data[k][:,1],c=colour,ls='',
                     marker='2',label=name,alpha=opacity)
            name = ''
    
    return None


# plot an entire track
def plot_track(track,colour,line_style,name,mrkrs=['x','.'],
               first_is_last=False):
    # orient track correctly
    if first_is_last:
        track = track[::-1,:]
    # plot (and label) the line
    plt.plot(track[:,0],track[:,1],c=colour,ls=line_style,label=name)
    # plot the start as an 'x'
    plt.plot(track[0,0],track[0,1],c=colour,ls='',marker=mrkrs[0])
    # plot the final location as a '.'
    plt.plot(track[-1,0],track[-1,1],c=colour,ls='',marker=mrkrs[1])
    return None


# add credible set
def add_uncertainty(location,variance,colour,opacity=0.3,cred_level=0.99):
    # convert credibility level into radius
    tail_out = (1 - cred_level) / 2
    num_sd = -norm.ppf(tail_out)
    r = num_sd * variance**0.5
    # add patch to plot
    plt.gca().add_patch(Circle(location,r,ec=colour,fc=colour,alpha=opacity,
                               label='',ls=''))
    return None


# add track uncertainty
def add_track_unc(X,V,colour='blueviolet',opacity=0.3,cred_level=0.95):
    # inputs: X – 2D locations over time, V – variance vector over time
    
    # convert credibility level into radii
    tail_out = (1 - cred_level) / 2
    num_sd = -norm.ppf(tail_out)
    r = num_sd * V**0.5
    
    # just add circle if track is new
    if X.shape[0] == 1:
        
        # add patch to plot
        plt.gca().add_patch(Circle(X[0,:],r,fc=colour,alpha=opacity,label='',
                                   ls=''))
        return None
    
    elif X.shape[0] == 0:
        # do nothing if there is no track
        return None
    
    else:
        
        # get error band (perpendicular to track)
        dx = np.concatenate([[X[1,0] - X[0,0]],
                             X[2:,0] - X[:-2,0],
                             [X[-1,0] - X[-2,0]]])
        dy = np.concatenate([[X[1,1] - X[0,1]],
                             X[2:,1] - X[:-2,1],
                             [X[-1,1] - X[-2,1]]])
        l = np.hypot(dx,dy) + 1e-100
        nx = dy / l
        ny = -dx / l
        
        # get vertices of path
        Xp = X + (r*np.array([nx,ny])).T
        Xn = X - (r*np.array([nx,ny])).T
        top_verts,top_codes = get_semiC_path(X[-1,:],dx[-1],dy[-1],r[-1])
        base_verts,base_codes = get_semiC_path(X[0,:],-dx[0],-dy[0],r[0])
        vertices = np.vstack([Xp,top_verts[1:,:],Xn[::-1],base_verts[1:,:]])
        codes = np.concatenate([[Path.MOVETO],
                                np.full(len(X)-1,Path.LINETO),
                                top_codes[1:],
                                np.full(len(X),Path.LINETO),
                                base_codes[1:]])
        path = Path(vertices,codes)
        # add patch to plot
        plt.gca().add_patch(PathPatch(path,fc=colour,alpha=opacity,label='',
                                      ls=''))

        return None


# get the angles to make a semi-circular path
def get_semiC_path(X,dx,dy,r):
    phi = np.arctan2(dy,dx) * 180/np.pi
    unit_semi_path = Path.arc(phi-90,phi+90)
    verts,codes = dc(unit_semi_path.vertices),dc(unit_semi_path.codes)
    verts *= r
    verts += X
    return verts,codes


# tidy and show the current plot
def tidy_plot(bounds=None,x_name='x (m)',y_name='y (m)',title='',legend_loc=0):
    if type(bounds) != type(None):
        plt.xlim(bounds[0,:])
        plt.ylim(bounds[1,:])
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.title(title)
    if type(legend_loc) != type(None):
        plt.legend(loc=legend_loc,framealpha=1)
    plt.show()
    return None









##### generate track #####


def gen_iSE_track(Tmax,d,s2,l,dim=2,dt=1,first_is_last=False):
    
    # initiate track object
    x = np.zeros([Tmax,dim])
    
    # get prior variance
    t = dt * np.arange(d,0,-1)
    C = iSE(t,t,s2,l)
    
    # sample over initial window
    sqrt_C = cholesky(C)
    x[-d:,:] = sqrt_C.T @ norm.rvs(size=[d,dim])
    
    # common quantities
    g = solve(C[1:,1:],C[1:,0])
    q = C[0,0] - C[0,1:] @ g
    
    for k in range(d,Tmax):
        
        # prepare next sample
        x_star = x[d-k-1,:]
        mean = x_star + g.T @ (x[-k:d-k-1,:] - x_star)
        
        # sample next step
        x[-k-1,:] = norm.rvs(mean,q**0.5)
    
    if not first_is_last:
        x = x[::-1,:]
    
    return x

def gen_SE_track(Tmax,d,s2,l,dim=2,dt=1,first_is_last=False):
    
    # initiate track object
    x = np.zeros([Tmax,dim])
    
    # get prior variance
    t = dt * np.arange(d,0,-1)
    C = SE(t,t,s2,l)
    
    # sample over initial window
    sqrt_C = cholesky(C)
    x[-d:,:] = sqrt_C.T @ norm.rvs(size=[d,dim])
    
    # common quantities
    g = solve(C[1:,1:],C[1:,0])
    q = C[0,0] - C[0,1:] @ g
    
    for k in range(d,Tmax):
        
        # prepare next sample
        mean = g.T @ x[-k:d-k-1,:]
        
        # sample next step
        x[-k-1,:] = norm.rvs(mean,q**0.5)
    
    if not first_is_last:
        x = x[::-1,:]
    
    return x


# make appropriate plot
def plot_gen_tracks(tracks,names,colours,styles,t):
    
    for track,name,col,style in zip(tracks,names,colours,styles):
        
        for dim in range(track.shape[1]):
            label = name if dim == 0 else ''
            # vary colours if needed
            if type(col) == list:
                col_dim = col[dim]
            else:
                col_dim = col
            plt.plot(t,track[:,dim],c=col_dim,ls=style,label=label)
            plt.plot(t[0],track[0,dim],c=col_dim,ls='',marker='x',label='')
            plt.plot(t[-1],track[-1,dim],c=col_dim,ls='',marker='.',label='')
    
    # show plot
    plt.xlabel('time (s)')
    plt.legend()
    plt.show()
    
    return None
    
    
        