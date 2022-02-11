import numpy as np
from astropy.io import fits
import random
import os
import pandas as pd
import math
#os.chdir('/Users/nikhil/code/Cluster codes') #Comment on cluster
from emcee_functions_norm2_mpi import log_likelihood,log_prior,log_probability
import site
import sys
import emcee
from schwimmbad import MPIPool

os.environ["OMP_NUM_THREADS"] = "1"

#specify input output paths
path_output='/mnt/gpfs3_amd/scratch/naj222/dir1/output/norm2_all_lick_bin/' #Uncomment on cluster
path_input='/mnt/gpfs3_amd/scratch/naj222/dir1/input/'             #Uncomment on cluster

#path_input='/Users/nikhil/Data/emcee_test/Input/'      #Comment on cluster
#path_output='/Users/nikhil/Data/emcee_test/Output/'    #Comment on cluster
 

#specify path to text file that contains MaNGA IDs of galaxies with AGN cut and axis ratio cut
with open(path_input+'sample_txt/TOT.txt') as f:
     Line0 = [line.rstrip('\n') for line in open(path_input+'sample_txt/TOT.txt')]
     
t_index = np.array([5,9,13,17,21,25,27,29])
COLOR_4SSP = fits.open(path_input+'ssp/ssp_bc03_4.color.fits')
DATA_4SSP = COLOR_4SSP[0].data[:,t_index,:]
#BC03 Ages


#Loop runs over each galaxy
i=80
for i in range (80,85):#np.shape(Line0)[0]):
 #get galaxy info from drpall
 drpall = fits.open(path_input+'drpall/drpall-v2_3_1.fits')
 tbdata = drpall[1].data
 indx = np.where(tbdata['mangaid'] == Line0[i])
 zeta = tbdata['nsa_z'][indx][0] #Redshift
 nsa_mass = tbdata['nsa_elpetro_mass'][indx][0]
 #Open the pre computed SSP integrated observables for the specific redshift.
 DATA = fits.open(path_input+'models/bc03_ssp_'+str(zeta)+'_.fits')
 t_age = DATA['Ages'].data[t_index]
 CI = DATA[0].data[:,:,:,t_index]
 CR = DATA[1].data[:,:,:,t_index]
 CB = DATA[2].data[:,:,:,t_index]
 D4 = DATA[3].data[:,:,:,t_index]
 Photo = DATA[4].data[:,2:,:,t_index]
 PhotoAB = DATA[5].data[:,2:,:]
 L_R = DATA[8].data
 L_B = DATA[9].data
 L_I = DATA[10].data
 D_I = DATA[11].data
 EBV_I = DATA[13].data
 Metal_I = DATA[12].data
 D_lmd = DATA['Dlambda_I'].data
 #Open the data file
 hdu = fits.open(path_input+'swim_files/SwiM_'+str(Line0[i])+'.fits')
 bin=0
 for bin in range (0,3):
  bins_idx=bin
  N_bin=hdu[28].data[bins_idx]
  b_array = np.array([0,0.59,0.61,1.27,1.27,1.27,1.27,1.27])
  Nanomags_covar = 1 + b_array*np.log10(N_bin)
  Nanomags_data = hdu[22].data[:,bins_idx]
  Mags_data = 22.5-2.5*np.log10(Nanomags_data)
  Nanomags_error = hdu[23].data[:,-1]*Nanomags_covar
  Mags_error = (Nanomags_error/Nanomags_data)

  Dn4000_data = hdu[18].data[43][bins_idx]
  Dn4000_error = hdu[19].data[43][bins_idx]*(1+1.156*np.log10(N_bin))*2
  Lick_indices_data = hdu[18].data[0:42,:][:,bins_idx]
  Lick_indices_error = hdu[19].data[0:42][:,bins_idx]*(1+1.156*np.log10(N_bin))*2
  Lick_list = np.argwhere(~np.isnan(Lick_indices_data))[:,0]
 
  nwalkers = 22 #Nwalker > 2*Ndims (Mentioned in documentation of Ensemble Sampler)
  nsamples=2000000
  check_steps = 2000
 
  Lg_SM = np.random.uniform(0,6,size=(nwalkers,np.shape(t_index)[0]-1))#Shape=(nwalkers,t-1) ,Last age is Normalized to 1
  SM = 10**(Lg_SM)
  Initial_SM = (SM.T/np.sum(SM,axis=1)).T
  Initial_normed_SM = np.zeros(np.shape(Initial_SM))
  for walker in range (0,nwalkers):
        Initial_normed_SM[walker,:] = np.random.uniform(0,nsa_mass)*Initial_SM[walker,:]
  Initial_Log_SM = np.log10(Initial_normed_SM)
  Initial_E_young = np.random.uniform(0.5,1.5,size=(nwalkers,1)) #Shape=(nwalkers,1)
  Initial_E_old = np.random.uniform(0,Initial_E_young,size=(nwalkers,1)) #Shape=(nwalkers,1)
  Initial_Z = np.random.uniform(-1.9,0.5,size=(nwalkers,1)) #Shape=(nwalkers,1)
  Initial_theta = np.concatenate((Initial_Log_SM,Initial_E_young,Initial_E_old,Initial_Z),axis=1) #Parameters (a1,a2,....a10,E(B-V),[Fe/H])
  ndim = Initial_theta.shape[1]
  pos = [Initial_theta[0,:] + np.random.randn(ndim) for i in range(nwalkers)] # a Nwalker by Ndim matrix
  arglist =(t_age,CI,CR,CB,D4,Photo,PhotoAB,L_R,L_B,L_I,D_I,EBV_I,Metal_I,D_lmd,Lick_indices_data,Dn4000_data,Mags_data,Dn4000_error,Lick_indices_error,Mags_error,nsa_mass,DATA_4SSP,Lick_list)
 
  filename = path_output+str(Line0[i])+'_'+str(bin)
  backend = emcee.backends.HDFBackend(filename)
  backend.reset(nwalkers, ndim)
 
  moves_final = [(emcee.moves.DEMove(), 0.2), (emcee.moves.DESnookerMove(), 0.8),]
  sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,args=arglist,backend=backend,moves=moves_final,threads=96)
  max_n = nsamples
  index = 0
  autocorr = np.empty(max_n)
  old_tau = np.inf
  for sample in sampler.sample(Initial_theta, iterations=max_n, progress=True):
          if sampler.iteration % check_steps:
             continue


          tau = sampler.get_autocorr_time(tol=0)
          autocorr[index] = np.mean(tau)
          index += 1


          converged = np.all(tau * 100 < sampler.iteration)
          if converged:
             break
          old_tau = tau


 
