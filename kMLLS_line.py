# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 16:05:12 2019

kMLLS for line

@author: A40047
"""
#%% Parameters
fname='Line_SI-1-0.5 PoissonNoise.hspy'

#kMLLS control
k=3
ks=[2,1]		      	# Select component
tolerance=0.08          # tolerance for MLLS
maxiter=10

# smooth: mid-point filter
mid_filt=1
win_m=3

# SI control
sub_si=0
si_e=[350.,-1]

sub_bg=0
bg_e=[380,395]

# plot control
# fix the color
c_ls=[[0.9,0.3,0.1],[0.1,0.9,0.3],[0.3,0.1,0.9],
      [0.9,0.2,0.9],[0.9,0.9,0.2],[0.2,0.9,0.9],
      [0.6,0.0,0.2],[0.2,0.6,0.0],[0.0,0.2,0.6]]

it=0
fs=16
fig_count=0

#%% load packages
import hyperspy.api as hs
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#%% User defined function
def MLLS(target,ref):
    coff=(target.dot(ref.T)).dot(np.linalg.inv(ref.dot(ref.T)))
    return coff

def kMLLS(datum,ref,tolerance):
	global it
	component=np.zeros([datum.shape[0],ref.shape[0]])
	ref_refined=ref*0
	for i in range(datum.shape[0]):
		component[i,:]=MLLS(datum[i,:],ref)	
	while (component>1+tolerance).any():
		if it>=maxiter:
			break
		it+=1
		# component refinement
		ns=np.zeros(ref.shape[0])
		for i in range(ref.shape[0]):
			if (component[:,i]>1).any():
				ref_refined[i,:]=datum[component[:,i]>1-tolerance,:].mean(axis=0)
				ns[i]=(component[:,i]>1-tolerance).sum()
		
		for i in range(datum.shape[0]):
			component[i,:]=MLLS(datum[i,:],ref_refined)
            
	return component, ref_refined, ns

def mid_point_filt(sp,w):
    if w//2 != 0:
        w+=1    
    temp=np.zeros([w,sp.shape[0]])
    for i in range(w):
        temp[i,:]=np.roll(sp,-1*(w//2)+i)
    v_max=np.max(temp,axis=0)
    v_min=np.min(temp,axis=0)
    v_mid=(v_max+v_min)/2
    v_mid[:(w//2)]=sp[:(w//2)]
    v_mid[-1*(w//2)::]=sp[-1*(w//2)::]
    return v_mid

#%% Main script
    
# Load data
si=hs.load(fname)
if sub_si:
    si=si.isig[si_e[0]:si_e[1]]

if sub_bg:
    si=si.remove_background(signal_range=(bg_e[0],bg_e[1]),
                            background_type='PowerLaw',fast=False)

datum=si.data
dx=si.axes_manager[0].axis
ex=si.axes_manager[-1].axis
units=si.axes_manager[0].units

# K-means and the centroids
km = KMeans(n_clusters=k, random_state=0).fit(datum)
label=km.labels_
ksp=km.cluster_centers_                 # Averaging spectrum for each clusters
ns=np.zeros([k,1])						# Quantity for each component
for i in range(k):
    ns[i]=sum(km.labels_==i)

ref0=ksp[ks,:]
component0=np.zeros([datum.shape[0],ref0.shape[0]])
for i in range(datum.shape[0]):
    component0[i,:]=MLLS(datum[i,:],ref0)
 
# kMLLS
component1, ref1, ns1 = kMLLS(datum,ref0,tolerance)
if mid_filt:
    for i in range(ref0.shape[0]):
        component0[:,i]=mid_point_filt(component0[:,i],win_m)
        component1[:,i]=mid_point_filt(component1[:,i],win_m)

# Plot
# k-means centroids
fig_count+=1
f=plt.figure(fig_count,figsize=(6*k*0.9,3))
f.subplots_adjust(hspace=0.2, wspace=0.05)
for i in range(k):
    plt.subplot(1,k,i+1)
    plt.plot(ex,ksp[i,:],lw=3,color=c_ls[i],label='Cluster %i'%(i+1))
    plt.xlabel('Energy loss (eV)',fontsize=fs)
    plt.legend(fontsize=fs)
    plt.xticks(fontsize=fs-2)
    plt.yticks([])
    plt.xlim(ex[[0,-1]])
    if i==0:
        plt.title('k-means clustering',fontsize=fs)
        plt.ylabel('Intensity (a.u.)',fontsize=fs)
plt.show()

# k-means profile
fig_count+=1
f=plt.figure(fig_count)
for i in range(ref0.shape[0]):
    plt.plot(dx,component0[:,i]*100,
             lw=3,color=c_ls[ks[i]],
             label='Cluster %d'%(ks[i]+1))
    plt.xlim(dx[[0,-1]])
    plt.xlabel('Position (%s)'%(units),fontsize=fs)
    plt.ylabel('Fraction (%)',fontsize=fs)
    plt.legend(fontsize=fs,loc='center right')
    plt.title('k-means clustering',fontsize=fs)
    plt.xticks(fontsize=fs-2)
    plt.yticks(rotation='vertical',fontsize=fs-2)
plt.show()

# kMLLS profile
fig_count+=1
f=plt.figure(fig_count)
for i in range(ref0.shape[0]):
    plt.plot(dx,component1[:,i]*100,
             lw=3,color=c_ls[ks[i]],
             label='Cluster %d'%(ks[i]+1))
    plt.xlim(dx[[0,-1]])
    plt.xlabel('Position (%s)'%(units),fontsize=fs)
    plt.ylabel('Fraction (%)',fontsize=fs)
    plt.legend(fontsize=fs,loc='center right')
    plt.title('kMLLS clustering',fontsize=fs)
    plt.xticks(fontsize=fs-2)
    plt.yticks(rotation='vertical',fontsize=fs-2)
plt.show()

fig_count+=1
f=plt.figure(fig_count,figsize=(6*k*0.9,3))
f.subplots_adjust(hspace=0.2, wspace=0.05)
for i in range(ref0.shape[0]):
    plt.subplot(1,k,i+1)
    plt.plot(ex,ref1[i,:],lw=3,color=c_ls[ks[i]],label='Refined cluster %i'%(ks[i]+1))
    plt.xlabel('Energy loss (eV)',fontsize=fs)
    plt.legend(fontsize=fs)
    plt.xticks(fontsize=fs-2)
    plt.yticks([])
    plt.xlim(ex[[0,-1]])
    if i==0:
        plt.title('kMLLS clustering',fontsize=fs)
        plt.ylabel('Intensity (a.u.)',fontsize=fs)
plt.show()