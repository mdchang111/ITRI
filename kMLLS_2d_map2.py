# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 13:30:55 2019

test for kMLLS

@author: A40047
"""

#%% load packages
import numpy as np
import matplotlib.pyplot as plt
import hyperspy.api as hs

from sklearn.cluster import KMeans
from mpl_toolkits.axes_grid1 import make_axes_locatable

#%% User defined functions
def MLLS(target,ref):
    coeff=(target.dot(ref.T)).dot(np.linalg.inv(ref.dot(ref.T)))
    return coeff

def kMLLS(datum,ref,tolerance):
	it=0
	maxiter=100
	component=np.zeros([datum.shape[0],ref.shape[0]])
	ref_refined=ref
	for i in range(datum.shape[0]):
		component[i,:]=MLLS(datum[i,:],ref)	
	while (component>1+tolerance).any():
		if it>=maxiter:
			break
		it+=1
        
		# component refinement
		for i in range(ref.shape[0]):
			ref_refined[i,:]=datum[component[:,i]>1-tolerance,:].mean(axis=0)

		for i in range(datum.shape[0]):
			component[i,:]=MLLS(datum[i,:],ref_refined)         
	return component, ref_refined

#%% load files
fname='itri-30%PossionNoise.hspy'
si=hs.load(fname)

dataset=si.data
ex=si.axes_manager[-1].axis
    
[sy, sx, sz] = dataset.shape
datum=dataset.reshape([sy*sx,sz])

#%% kMLLS
k=4
km=KMeans(n_clusters=k,
          init='k-means++',
          n_init=10,
          max_iter=300,
          random_state=0)
km.fit(datum)
refs=km.cluster_centers_
components, refs = kMLLS(datum,refs,tolerance=0.1)

#%% plot ref spectra
f2,ax=plt.subplots(1,k,figsize=(9.5,3),dpi=100)
ax=ax.ravel()
for i in range(k):
    ax[i].plot(ex,refs[i,:])
    ax[i].tick_params(axis='both', which='major', labelsize=6)
    ax[i].set_yticks([])
    ax[i].set_xlabel('Energy loss (eV)',fontsize=8)
    ax[i].set_xlim([ex[0],ex[-1]])
    ax[i].set_title('Cluster %i'%(i+1))
f2.tight_layout()


#%% spectrum + map
for i in range(k):
    f3, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5,5), dpi=100)
    ax1.plot(ex,refs[i,:])
    ax1.tick_params(axis='both', which='major', labelsize=6)
    ax1.set_yticks([])
    ax1.set_xlabel('Energy loss (eV)',fontsize=8)
    ax1.set_xlabel('Energy loss (eV)',fontsize=8)
    ax1.set_xlim([ex[0],ex[-1]])
    ax1.set_title('Cluster %i'%(i+1),fontsize=12)
    
    fig2=ax2.imshow(components[:,i].reshape([sy,sx]),
                  cmap='plasma')
    ax2.set_yticks([])
    ax2.set_xticks([])
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(fig2,cax=cax)
    f3.tight_layout()






