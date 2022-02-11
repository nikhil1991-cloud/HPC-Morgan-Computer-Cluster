import numpy as np
from matplotlib import pyplot as plt
from astropy.io import fits
from scipy import stats
import pandas as pd
import math
import site
import sys
site.addsitedir('/home/naj222')
import emcee
import corner
from itertools import combinations
import extinction

path_input = '/mnt/gpfs3_amd/scratch/naj222/dir1/input/'
path_output = '/mnt/gpfs3_amd/scratch/naj222/dir1/output/norm2_all_lick_bin/'
path_store = '/mnt/gpfs3_amd/scratch/naj222/dir1/output/norm2_R1R2_bin/'

#path_input='/Users/nikhil/Desktop/emcee_test/Input/'    #Comment on cluster
#path_output='/Users/nikhil/Desktop/emcee_test/Output/'  #Comment on cluster
#path_plots='/Users/nikhil/Desktop/emcee_test/'          #Comment on cluster

#specify path to text file that contains MaNGA IDs of galaxies with AGN cut and axis ratio cut
with open(path_input+'sample_txt/TOT.txt') as f:
     Line0 = [line.rstrip('\n') for line in open(path_input+'sample_txt/TOT.txt')]
     
bin_array = [0,1,2]     
bin_index=0   
for bin_index in range (0,len(bin_array)):
  R1_all = np.zeros((np.shape(Line0)[0],10))
  R2_all = np.zeros((np.shape(Line0)[0],10))
  R3_all = np.zeros((np.shape(Line0)[0],10))
  SR1_all = np.zeros((np.shape(Line0)[0],10))
  SR2_all = np.zeros((np.shape(Line0)[0],10))
  SR3_all = np.zeros((np.shape(Line0)[0],10))
  RM1_all = np.zeros((np.shape(Line0)[0],10))
  RM2_all = np.zeros((np.shape(Line0)[0],10))
  RM3_all = np.zeros((np.shape(Line0)[0],10))
  E_LINES = np.zeros((np.shape(Line0)[0],22))
  E_FLUX = np.zeros((np.shape(Line0)[0],22))
  LICK_ALL = np.zeros((np.shape(Line0)[0],42))
  D4000_ALL = np.zeros(np.shape(Line0)[0])
  COLORS_ALL = np.zeros((np.shape(Line0)[0],6))
  M_ALL = np.zeros((np.shape(Line0)[0],8))
  BC03_ALL = np.zeros((np.shape(Line0)[0],8))
  DELTAT_ALL = np.zeros((np.shape(Line0)[0],8))
  EBV_ALL = np.zeros((np.shape(Line0)[0],2))
  METAL_ALL = np.zeros(np.shape(Line0)[0])
  MASS_ALL = np.zeros(np.shape(Line0)[0])
  Z_ALL = np.zeros(np.shape(Line0)[0])
  T1 = np.zeros((np.shape(Line0)[0],10))
  T2 = np.zeros((np.shape(Line0)[0],10))
  T3 = np.zeros((np.shape(Line0)[0],10))
  PARS_TAU = np.zeros((np.shape(Line0)[0],10))
  N_steps = np.zeros(np.shape(Line0)[0])
  EBV_BD = np.zeros(np.shape(Line0)[0])
  EBV_BD_SIG = np.zeros(np.shape(Line0)[0])
  SERSIC = np.zeros(np.shape(Line0)[0])
  #t_index = np.array([5,8,11,17,20,24,27,29]) #Old scheme
  t_index = np.array([5,9,13,17,21,25,27,29]) #New scheme
  COLOR_4SSP = fits.open(path_input+'ssp/ssp_bc03_4.color.fits')
  DATA_4SSP = COLOR_4SSP[0].data[:,t_index,:]
  #BC03 Ages
  i=0
  for i in range (0,np.shape(Line0)[0]):
    #get galaxy info from drpall
    drpall = fits.open(path_input+'drpall/drpall-v2_3_1.fits')
    tbdata = drpall[1].data
    indx = np.where(tbdata['mangaid'] == Line0[i])
    nsa_mass = tbdata['nsa_elpetro_mass'][indx][0]
    zeta = tbdata['nsa_z'][indx][0] #Redshift
    sersic = tbdata['nsa_sersic_n'][indx][0]
    #Open the pre computed SSP integrated observables for the specific redshift.
    DATA = fits.open(path_input+'models/bc03_ssp_'+str(zeta)+'_.fits')
    Ages = DATA[6].data[t_index]
    T_age = np.log10(Ages)
    Tmid = (T_age[1:] + T_age[:-1]) / 2
    Tbeg = np.zeros(np.shape(T_age))
    Tend = np.zeros(np.shape(T_age))
    Tbeg[0:len(Ages)-1] = Tmid
    Tbeg[-1] = np.log10(14)
    Tend[1:len(Ages)] = Tmid
    Tend[0] = -9
    delta_T = 10**Tbeg - 10**Tend
    Beg_Age = 10**(Tbeg)
    comb_all = np.array(list(combinations(Beg_Age[1:-2]+0.001,3)))
    #Open the data file
    #hdu = fits.open("/Volumes/Nikhil/MPL-7_Files/SwiM_binned/SwiM_"+str(Line0[i])+".fits")
    hdu = fits.open(path_input+'swim_files/SwiM_'+str(Line0[i])+'.fits')
    N_bin=hdu[28].data[bin_index]
    b_array = np.array([0,0.59,0.61,1.27,1.27,1.27,1.27,1.27])
    Nanomags_covar = 1 + b_array*np.log10(N_bin)
    #We are dealing with integrated bin within 1.5 Re so the integrated binned data is stored as the last element
    #Read mags from data
    SERSIC[i] = sersic
    E_LINES[i,:] = hdu[20].data[:,bin_index]
    E_FLUX[i,:] = hdu[26].data[:,bin_index]
    Nanomags_data = hdu[22].data[:,bin_index]
    Mags_data = 22.5-2.5*np.log10(Nanomags_data)
    Nanomags_error = np.sqrt(hdu[23].data[:,bin_index])
    Mags_error = (Nanomags_error/Nanomags_data)*Nanomags_covar
    #if Mags_error.max()<0.03:
    ##    Max_error = 0.03
    #else:
    #    Max_error = Mags_error.max()
    #Mags_error = np.clip(Mags_error,0.03,Max_error)
    #Read Lick indices and d4000 from data
    Dn4000_data = hdu[18].data[43][bin_index]
    Dn4000_error = hdu[19].data[43][bin_index]*(1+1.156*np.log10(N_bin))*2
    Lick_indices_data = hdu[18].data[0:42,:][:,bin_index]
    Lick_indices_error = hdu[19].data[0:42][:,bin_index]*(1+1.156*np.log10(N_bin))*2
    HA_HB = hdu[26].data[18][-1]/hdu[26].data[11][bin_index]
    HA_sigma = 0.5*np.sqrt(hdu[26].data[18][bin_index])
    HB_sigma = 0.5*np.sqrt(hdu[26].data[11][bin_index])
    HA_HB_sigma = np.sqrt(((hdu[26].data[18][bin_index]*HB_sigma)**2 + (hdu[26].data[11][bin_index]*HA_sigma)**2)/(hdu[26].data[11][bin_index])**4)
    HA_HB_l = np.array([6564.0,4862.0])
    A_HA_HB = extinction.calzetti00(HA_HB_l, 1.0, 3.1)
    if HA_HB<2.83:
        print(HA_HB)
    Ebv_young = (1/1.24)*np.log10(HA_HB/2.86)*(1/(A_HA_HB[1]-A_HA_HB[0]))
    Ebv_sigma = (1/1.24)*(1/(A_HA_HB[1]-A_HA_HB[0]))*(HA_HB_sigma/HA_HB)
    EBV_BD[i] = Ebv_young
    EBV_BD_SIG[i] = Ebv_sigma
    #generate alpha matrices
    Lick_list = [3,4,8,13,14,21,22,23,24,25] #fe4383,fe5270,fe5335,HdA
    m1_list,m2_list = [0,0,4,4,3,3],[3,1,6,5,6,7]  #w2-u,w2-w1,g-i,g-r,u-i,u-z
    mags_info = np.array(['W2','W1','M2','u','g','r','i','z'])
    #Initialize array
    reader = emcee.backends.HDFBackend(path_output+str(Line0[i])+'_'+str(bin_index))
    tau = reader.get_autocorr_time(tol=0)
    PARS_TAU[i,:] = tau
    chain_len = len(reader.get_chain())
    N_steps[i] = chain_len
    pars = ['Log($M_{0.01}$)','Log($M_{0.03}$)','Log($M_{0.06}$)','Log($M_{0.4}$)','Log($M_{1.2}$)','Log($M_{4}$)','Log($M_{14}$)','$E_{y}$','$E_{o}$','[Fe/H]']
    max_par = np.where(tau==tau.max())[0][0]
    non_converged_par = pars[max_par]
    #tau = reader.get_autocorr_time(quiet=True)
    nburnin= int(2*tau.max())
    thin = int(0.5*tau.min())
    bins=20
    #get flat samples of parameters and ln_probability
    nflat_samples = reader.get_chain(discard=nburnin,thin=thin,flat=True) #Chain thinning
    nflat_lnprob = reader.get_log_prob(discard=nburnin,thin=thin,flat=True)
    nchisq = (nflat_lnprob/(-0.5))/40
    good_samples = np.where((nchisq>=nchisq.min())&(nchisq<nchisq.min()+1))[0]
    flat_samples = nflat_samples[good_samples,:]
    flat_lnprob=nflat_lnprob[good_samples]
    flat_chisq = nchisq[good_samples]
    best_index = np.where(flat_chisq==flat_chisq.min())[0][0]
    new_sfr = flat_samples[best_index,:]
    sigma_logsfr = np.zeros(flat_samples.shape[1])
    par = 0
    for par in range (0,flat_samples.shape[1]):
        sigma_logsfr[par] = (np.sqrt(np.sum((flat_samples[best_index,par] - flat_samples[:,par])**2)/len(flat_samples)))*0.8
    sigma_logsfr = np.append(sigma_logsfr[:-3],0)
    for comb in range (0,len(comb_all)):
         time1=comb_all[comb,0]
         time2=comb_all[comb,1]
         time3=comb_all[comb,2]
         #Calculate mass
         Z_all= np.array([-2.2490,-1.6464,-0.6392,-0.3300,0.0932,0.5595])
         Best_Metal = new_sfr[-1]
         IN_H = np.where(Z_all>Best_Metal)[0][0]
         IN_L = np.where(Z_all<Best_Metal)[0][-1]
         M2 = DATA_4SSP[IN_H,:,6]
         M1 = DATA_4SSP[IN_L,:,6]
         bc03_M = (M1*(Z_all[IN_H]-Best_Metal) + M2*(Best_Metal-Z_all[IN_L]))/(Z_all[IN_H]-Z_all[IN_L])
         #Calculate mass
         best_SM = 10**(new_sfr[:-3])
         last_SM = nsa_mass - np.sum(best_SM)
         #Normalization
         M_ALL[i,:] = np.append(best_SM,last_SM)
         BC03_ALL[i,:] = bc03_M
         DELTAT_ALL[i,:] = delta_T
         EBV_ALL[i,0] = new_sfr[-3]
         EBV_ALL[i,1] = new_sfr[-2]
         METAL_ALL[i] = new_sfr[-1]
         SFH = (np.append(best_SM,last_SM)/bc03_M)/delta_T
         Total_mass = np.append(best_SM,last_SM)/bc03_M

 
         t1 = np.where(Beg_Age<time1)
         t2 = np.where((Beg_Age>time1) & (Beg_Age<time2))
         t3 = np.where((Beg_Age>time2) & (Beg_Age<time3))
         t4 = np.where(Beg_Age>time3)

         R1_all[i,comb] = np.log10((np.sum(Total_mass[t1])/np.sum(delta_T[t1]))/(np.sum(Total_mass[t2])/np.sum(delta_T[t2])))
         R2_all[i,comb] =  np.log10((np.sum(Total_mass[t3])/np.sum(delta_T[t3]))/(np.sum(Total_mass[t4])/np.sum(delta_T[t4])))
         R3_all[i,comb] = np.log10((np.sum(Total_mass[np.c_[t1,t2][0]])/np.sum(delta_T[np.c_[t1,t2][0]]))/(np.sum(Total_mass[np.c_[t3,t4][0]])/np.sum(delta_T[np.c_[t3,t4][0]])))
         SR1_all[i,comb] = np.sqrt(np.sum(sigma_logsfr[t1]**2) + np.sum(sigma_logsfr[t2]**2))
         SR2_all[i,comb] = np.sqrt(np.sum(sigma_logsfr[t3]**2) + np.sum(sigma_logsfr[t4]**2))
         SR3_all[i,comb] = np.sqrt(np.sum(sigma_logsfr[np.c_[t1,t2][0]]**2) + np.sum(sigma_logsfr[np.c_[t3,t4][0]]**2))
         D4000_ALL[i] = Dn4000_data
         LICK_ALL[i,:] = Lick_indices_data
         COLORS_ALL[i,:] = Mags_data[m1_list] - Mags_data[m2_list]
         MASS_ALL[i] = nsa_mass
         Z_ALL[i] = zeta
         T1[i,comb] = Beg_Age[t1][-1]
         T2[i,comb] = Beg_Age[t2][-1]
         T3[i,comb] = Beg_Age[t3][-1]
 



  col1 = Line0
  col2 = R1_all
  col3 = R2_all
  col4 = R3_all
  col5 = D4000_ALL
  col6 = LICK_ALL
  col7 = COLORS_ALL
  col8 = MASS_ALL
  col9 = Z_ALL
  col10 = M_ALL
  col11 = BC03_ALL
  col12 = DELTAT_ALL
  col13 = EBV_ALL
  col14 = METAL_ALL
  col15 = T1
  col16 = T2
  col17 = T3
  col18 = EBV_BD
  col19 = EBV_BD_SIG
  col20 = PARS_TAU
  col21 = N_steps
  col22 = SR1_all
  col23 = SR2_all
  col24 = SR3_all
  col25 = E_LINES
  col26 = E_FLUX
  col27 = SERSIC

  c1 = fits.Column(name='MaNGAID',format='10A',array=col1)
  c2 = fits.Column(name='R1',format='10E',array=col2)
  c3 = fits.Column(name='R2',format='10E',array=col3)
  c4 = fits.Column(name='R3',format='10E',array=col4)
  c5 = fits.Column(name='Dn4000',format='E',array=col5)
  c6 = fits.Column(name='Lick',format='42E',array=col6)
  c7 = fits.Column(name='Colors',format='6E',array=col7)
  c8 = fits.Column(name='nsa_mass',format='E',array=col8)
  c9 = fits.Column(name='redshift',format='E',array=col9)
  c10 = fits.Column(name='M*',format='8E',array=col10)
  c11 = fits.Column(name='bc03_M[Fe/H]',format='8E',array=col11)
  c12 = fits.Column(name='dT',format='8E',array=col12)
  c13 = fits.Column(name='E(B-V)[Y,O]',format='2E',array=col13)
  c14 = fits.Column(name='[Fe/H]',format='E',array=col14)
  c15 = fits.Column(name='t1',format='10E',array=col15)
  c16 = fits.Column(name='t2',format='10E',array=col16)
  c17 = fits.Column(name='t3',format='10E',array=col17)
  c18 = fits.Column(name='E(B-V)[HA/HB]',format='E',array=col18)
  c19 = fits.Column(name='E(B-V)[HA/HB] Sigma',format='E',array=col19)
  c20 = fits.Column(name='Tau[10]',format='10E',array=col20)
  c21 = fits.Column(name='N_steps',format='E',array=col21)
  c22 = fits.Column(name='R1_Sigma',format='10E',array=col22)
  c23 = fits.Column(name='R2_Sigma',format='10E',array=col23)
  c24 = fits.Column(name='R3_Sigma',format='10E',array=col24)
  c25 = fits.Column(name='ELINE EW',format='22E',array=col25)
  c26 = fits.Column(name='ELINE Flux',format='22E',array=col26)
  c27 = fits.Column(name='sersic',format='E',array=col27)

  hdu = fits.BinTableHDU.from_columns([c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15,c16,c17,c18,c19,c20,c21,c22,c23,c24,c25,c26,c27])
  hdu.writeto(path_store+'R1R2_n2al_mle'+str(bin_index)+'.fits')
