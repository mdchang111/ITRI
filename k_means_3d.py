# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 16:56:34 2019

data processing

@author: A40047
"""

import hyperspy.api as hs
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


si=hs.load('itri.hspy')
sy, sx, sz = si.data.shape
noise_amp=0.3
noise=noise_amp*np.random.poisson(lam=1.0,size=si.data.shape)
noise-=np.mean(noise.flatten())
si.data+=noise
sp=si.data
fig_count=0

#%%
sp=sp.reshape((sy*sx,sz))

distorsion=[]
for k in np.arange(1,11):
    km=KMeans(n_clusters=k,
              init='k-means++',
              n_init=10,
              max_iter=1000,
              tol=1e-4,
              random_state=0)
    km.fit(sp)
    distorsion.append(km.inertia_)

fig_count+=1
plt.figure(fig_count)
plt.plot(np.arange(1,11),distorsion,'-ro')
plt.xticks(np.arange(1,11))    
plt.yticks([])
plt.xlabel('Number of clusters',fontsize=16)
plt.ylabel('Distorsion',fontsize=16)
plt.show()

#%%
k=4
km=KMeans(n_clusters=k,
          init='k-means++',
          n_init=10,
          max_iter=1000,
          tol=1e-4,
          random_state=0)
km.fit(sp)
labels=km.labels_
label_map=labels.reshape((sy,sx))

fig_count+=1
plt.figure(fig_count)
plt.imshow(label_map,cmap='Dark2')
plt.xticks([])
plt.yticks([])
plt.colorbar(ticks=[0,1,2,3])

#%%
ref_sp=np.zeros([k,sz])
ori_sp=np.zeros([k,sz])
n_sp=np.zeros([k,1])
for i in range(sy*sx):
    ref_sp[labels[i]]+=sp[i,:]
    ori_sp[labels[i]]=sp[i,:]
    n_sp[labels[i]]+=1

ref_sp/=n_sp
xx=si.axes_manager[-1].axis

fig_count+=1
f=plt.figure(fig_count,figsize=[15,10])
f.subplots_adjust(hspace=0.3, wspace=0.1)
for i in range(k):
    plt.subplot(2,2,i+1)
    plt.plot(xx,ori_sp[i,:],'b',lw=1,label='%i %% noise'%(noise_amp*100))
    plt.plot(xx,ref_sp[i,:],'r',lw=2,label='Average spectrum')
    plt.title('Cluster %i'%(i),fontsize=20)
    plt.xlabel('Energy loss (eV)',fontsize=16)
    plt.ylabel('Counts (a.u.)',fontsize=16)
    plt.xlim([xx[0],xx[-1]])
    plt.yticks([])
    plt.legend(fontsize=12,loc='upper right')
plt.show()

#%%
fig_count+=1
plt.figure(fig_count)
color_code=['yellowgreen','purple','goldenrod','grey']
for i in range(k):
    plt.fill_between(xx,ref_sp[i,:],color=color_code[i],
                     alpha=0.2,label='cluster %i'%(i))
    plt.xlabel('Energy loss (eV)',fontsize=16)
    plt.ylabel('Counts (a.u.)',fontsize=16)
    plt.xlim([xx[0],xx[-1]])
    plt.yticks([])
    plt.legend(fontsize=12,loc='upper right')
plt.show()

#%%
er=np.array([[1680.,1750.],
             [1780.,1880.],
             [1870.,1900.],
             [1900.,2000.]])
clst=['Reds','Oranges','Blues','Greens']
tlst=['Hf','Ta','Si','W']
for i in range(k):
    fig_count+=1
    plt.figure(fig_count)
    plt.imshow(si.isig[er[i,0]:er[i,1]].integrate1D(axis=2),cmap=clst[i])
    plt.xticks([])
    plt.yticks([])
    plt.title(tlst[i],fontsize=18)
    plt.show()
    
#%%
clst=['Greens','Reds','Blues','Oranges']
tlst=['W','Hf','Si','Ta']
for i in range(k):
    fig_count+=1
    plt.figure(fig_count)
    plt.imshow(label_map==i,cmap=clst[i])
    plt.xticks([])
    plt.yticks([])
    plt.title('k-means : '+tlst[i],fontsize=18)
    plt.show()