import numpy as np
from astropy.io import fits
import random
import os
import pandas as pd
import math
#os.chdir('/Users/nikhil/Downloads') #Comment on cluster
from emcee_functions_norm2_mpi import log_likelihood,log_prior,log_probability
from simulation_model import make_models,get_Mbc03,get_dt
import site
import sys
site.addsitedir('/home/naj222') #Uncomment on cluster
import emcee
from schwimmbad import MPIPool

os.environ["OMP_NUM_THREADS"] = "1"
#specify input output paths
path_output='/mnt/gpfs2_4m/scratch/naj222/dir1/output/simulation_noise/' #Uncomment on cluster
path_input='/mnt/gpfs2_4m/scratch/naj222/dir1/input/'             #Uncomment on cluster

#path_input = '/Users/nikhil/Data/emcee_test/Input/'
#specify path to text file that contains MaNGA IDs of galaxies with AGN cut and axis ratio cut
with open(path_input+'sample_txt/swim_ba_agn_cut.txt') as f:
     Line0 = [line.rstrip('\n') for line in open(path_input+'sample_txt/swim_ba_agn_cut.txt')]
     
t_index = np.array([5,9,13,17,21,25,27,29])
COLOR_4SSP = fits.open(path_input+'ssp/ssp_bc03_4.color.fits')
DATA_4SSP = COLOR_4SSP[0].data[:,t_index,:]
#BC03 Ages



drpall = fits.open(path_input+'drpall/drpall-v2_3_1.fits')
tbdata = drpall[1].data
indx = np.where(tbdata['mangaid'] == Line0[0])
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
Z_all = 0.0
e_folding_time = np.array([1,2,4,8,16])
eTau = 2
for eTau in range (2,3):#len(e_folding_time)):
        e_fold_tau = e_folding_time[eTau]
        Simulation_Ebv_1 = 0.5
        Simulation_Ebv_2 = 0.05
        Simulation_Z = 0.0
        dt = get_dt(t_age)
        bc03_M = get_Mbc03(Simulation_Z,DATA_4SSP)
        Simulated_SFH = np.exp(-(14-t_age)/e_fold_tau)
        Simulated_mass_prime = Simulated_SFH*(dt*bc03_M)
        Simulated_mass = (Simulated_mass_prime/np.sum(Simulated_mass_prime))*nsa_mass
        #Simulation_mass = (np.exp(t_age/e_fold_tau)/np.sum(np.exp(t_age/e_fold_tau)))*nsa_mass
        Log_Mass = np.log10(Simulated_mass)
        Initial_SFH = np.append(np.append(np.append(Log_Mass,Simulation_Ebv_1),Simulation_Ebv_2),Simulation_Z)
        Simulated_Log_SFH = np.log10(10**(Initial_SFH[:-3])/(bc03_M*dt))
        Sim_D4,Sim_Lick,Sim_Mag = make_models(Initial_SFH,t_age,CI,CR,CB,D4,Photo,PhotoAB,L_R,L_B,L_I,D_I,EBV_I,Metal_I,D_lmd,nsa_mass,DATA_4SSP)
 
        hdu = fits.open(path_input+'S2N/S2N3.fits')
        S2N_percent=50
        S2N_Lick = hdu[1].data['SN LICK']
        S2N_UV = hdu[1].data['SN UV']
        S2N_SDSS = hdu[1].data['SN SDSS']
        cutoff_manga = np.where(S2N_Lick>np.percentile(S2N_Lick,S2N_percent))[0][0]
        cutoff_swift = np.where(S2N_UV>np.percentile(S2N_UV,S2N_percent))[0][0]
        cutoff_sdss = np.where(S2N_SDSS>np.percentile(S2N_SDSS,S2N_percent))[0][0]
        UV_mag_error = hdu[1].data['MAGS SIG'][cutoff_swift,0:3]
        SDSS_mag_error = hdu[1].data['MAGS SIG'][cutoff_sdss,3:]
        Mags_error = np.append(UV_mag_error,SDSS_mag_error)
        Dn4000_error = hdu[1].data['D4 SIG'][cutoff_manga]
        Lick_indices_error = hdu[1].data['LICK SIG'][cutoff_manga,:]
        good_lick = np.where(Lick_indices_error>0)[0]

        #generate uncertainties
        fraction_array = [0,1,2,3,4,5,6,7,8,9,10]
        for f in range (1,5):#len(fraction_array)):
          fraction = fraction_array[f]
          ED4 = np.random.normal(0,1)*(Dn4000_error)
          ELick = np.zeros(np.shape(Sim_Lick))
          EMags = np.zeros(np.shape(Sim_Mag))
          for lick in range (0,len(Sim_Lick)):
            ELick[lick] = np.random.normal(0,1)*(Lick_indices_error[lick])
          for mag in range (0,len(Sim_Mag)):
            EMags[mag] = np.random.normal(0,1)*(Mags_error[mag])
          #add uncertainties
          Dn4000_data = Sim_D4 + ED4
          Lick_indices_data = Sim_Lick + ELick
          Mags_data = Sim_Mag + EMags
          
          nwalkers = 22 #Nwalker > 2*Ndims (Mentioned in documentation of Ensemble Sampler)
          nsamples = 2000000
          check_steps = 2000
         
          with MPIPool() as pool:
            if not pool.is_master():
                pool.wait()
                sys.exit(0)
                
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

            arglist =(t_age,CI,CR,CB,D4,Photo,PhotoAB,L_R,L_B,L_I,D_I,EBV_I,Metal_I,D_lmd,Lick_indices_data,Dn4000_data,Mags_data,Dn4000_error,Lick_indices_error,Mags_error,nsa_mass,DATA_4SSP,good_lick)
 
            filename = path_output+'_'+str(e_fold_tau)+'_'+'sigma_'+str(fraction)
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


 
