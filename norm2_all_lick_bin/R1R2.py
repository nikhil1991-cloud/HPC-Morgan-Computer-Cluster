import numpy as np
from matplotlib import pyplot as plt
from astropy.io import fits
from scipy import stats
import pandas as pd
import math
import site
import sys
import os
site.addsitedir('/home/naj222')
#os.chdir('/Users/nikhil/Downloads/')
from simulation_model import get_Mbc03,get_dt
import emcee
import corner
from itertools import combinations
import extinction

path_input = '/mnt/gpfs3_amd/scratch/naj222/dir1/input/'
path_output = '/mnt/gpfs3_amd/scratch/naj222/dir1/output/norm2_all_lick_bin/'
path_store = '/mnt/gpfs3_amd/scratch/naj222/dir1/output/norm2_R1R2_bin/'



#specify path to text file that contains MaNGA IDs of galaxies with AGN cut and axis ratio cut
with open(path_input+'sample_txt/TOT.txt') as f:
     Line0 = [line.rstrip('\n') for line in open(path_input+'sample_txt/TOT.txt')]

bin_array = [0,1,2]
bin_index=0
for bin_index in range (0,len(bin_array)):
 R1_mode = np.zeros(np.shape(Line0)[0])
 R2_mode= np.zeros(np.shape(Line0)[0])
 R3_mode = np.zeros(np.shape(Line0)[0])
 R1_median = np.zeros(np.shape(Line0)[0])
 R2_median= np.zeros(np.shape(Line0)[0])
 R3_median = np.zeros(np.shape(Line0)[0])
 R1_mle = np.zeros(np.shape(Line0)[0])
 R2_mle= np.zeros(np.shape(Line0)[0])
 R3_mle = np.zeros(np.shape(Line0)[0])
 SR1_all = np.zeros(np.shape(Line0)[0])
 SR2_all = np.zeros(np.shape(Line0)[0])
 SR3_all = np.zeros(np.shape(Line0)[0])
 SR1_lower_all = np.zeros(np.shape(Line0)[0])
 SR2_lower_all = np.zeros(np.shape(Line0)[0])
 SR3_lower_all = np.zeros(np.shape(Line0)[0])
 SR1_upper_all = np.zeros(np.shape(Line0)[0])
 SR2_upper_all = np.zeros(np.shape(Line0)[0])
 SR3_upper_all = np.zeros(np.shape(Line0)[0])
 E_LINES = np.zeros((np.shape(Line0)[0],22))
 E_FLUX = np.zeros((np.shape(Line0)[0],22))
 LICK_ALL = np.zeros((np.shape(Line0)[0],42))
 D4000_ALL = np.zeros(np.shape(Line0)[0])
 COLORS_ALL = np.zeros((np.shape(Line0)[0],6))
 Par_median = np.zeros((np.shape(Line0)[0],10))
 Par_mode = np.zeros((np.shape(Line0)[0],10))
 Par_mle = np.zeros((np.shape(Line0)[0],10))
 Par_sigma_upper = np.zeros((np.shape(Line0)[0],10))
 Par_sigma_lower = np.zeros((np.shape(Line0)[0],10))
 Mass_bc03_median = np.zeros((np.shape(Line0)[0],8))
 Mass_bc03_mle = np.zeros((np.shape(Line0)[0],8))
 Mass_bc03_mode = np.zeros((np.shape(Line0)[0],8))
 DELTAT_ALL = np.zeros((np.shape(Line0)[0],8))
 MASS_ALL = np.zeros(np.shape(Line0)[0])
 T1 = np.zeros(np.shape(Line0)[0])
 T2 = np.zeros(np.shape(Line0)[0])
 T3 = np.zeros(np.shape(Line0)[0])
 PARS_TAU = np.zeros((np.shape(Line0)[0],10))
 N_steps = np.zeros(np.shape(Line0)[0])
 SERSIC = np.zeros(np.shape(Line0)[0])
 TF_median = np.zeros(np.shape(Line0)[0])
 TF_mode = np.zeros(np.shape(Line0)[0])
 TF_mle = np.zeros(np.shape(Line0)[0])
 TF_sigma = np.zeros(np.shape(Line0)[0])
 TB_median = np.zeros(np.shape(Line0)[0])
 TB_mode = np.zeros(np.shape(Line0)[0])
 TB_mle = np.zeros(np.shape(Line0)[0])
 TB_sigma = np.zeros(np.shape(Line0)[0])
 TMASS = np.zeros(np.shape(Line0)[0])
 TMASS_lower = np.zeros(np.shape(Line0)[0])
 TMASS_upper = np.zeros(np.shape(Line0)[0])


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
    #Open the data file
    #hdu = fits.open("/Volumes/Nikhil/MPL-7_Files/SwiM_binned/SwiM_"+str(Line0[i])+".fits")
    hdu = fits.open(path_input+'swim_files/SwiM_'+str(Line0[i])+'.fits')
    #We are dealing with integrated bin within 1.5 Re so the integrated binned data is stored as the last element
    #Read mags from data
    N_bin=hdu[28].data[-1]
    b_array = np.array([0,0.59,0.61,1.27,1.27,1.27,1.27,1.27])
    Nanomags_covar = 1 + b_array*np.log10(N_bin)
    SERSIC[i] = sersic
    E_LINES[i,:] = hdu[20].data[:,-1]
    E_FLUX[i,:] = hdu[26].data[:,-1]
    MASS_ALL[i] = nsa_mass
    Nanomags_data = hdu[22].data[:,-1]
    Mags_data = 22.5-2.5*np.log10(Nanomags_data)
    Nanomags_error = hdu[23].data[:,-1]*Nanomags_covar
    Mags_error = (Nanomags_error/Nanomags_data)
    Dn4000_data = hdu[18].data[43][-1]
    Dn4000_error = hdu[19].data[43][-1]*(1+1.156*np.log10(N_bin))*2
    Lick_indices_data = hdu[18].data[0:42,:][:,-1]
    Lick_indices_error = hdu[19].data[0:42][:,-1]*(1+1.156*np.log10(N_bin))*2

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
    #tau = reader.get_autocorr_time(quiet=True)
    if np.isnan(tau.max()):
       max_tau = 100
    else:
       max_tau = tau.max()
    nburnin= int(2*max_tau)
    thin = int(0.5*max_tau)
    bins=20
    #get flat samples of parameters and ln_probability
    #get flat samples of parameters and ln_probability
    nflat_samples = reader.get_chain(discard=nburnin,thin=thin,flat=True) #Chain thinning
    nflat_lnprob = reader.get_log_prob(discard=nburnin,thin=thin,flat=True)
    nchisq = (nflat_lnprob/(-0.5))/39
    best_index = np.where(nchisq==nchisq.min())[0][0]
    new_sfr = np.zeros((3,np.shape(nflat_samples)[1]))
    new_sfr[0,:] = nflat_samples[best_index,:] #mle
    new_sfr[1,:] = np.median(nflat_samples,axis=0) #median
    new_sfr[2,:] = stats.mode(nflat_samples)[0][0] #mode
    par = 0
    for par in range (0,nflat_samples.shape[1]):
      l_index = np.where(nflat_samples[:,par]<nflat_samples[best_index,par])[0]
      h_index = np.where(nflat_samples[:,par]>nflat_samples[best_index,par])[0]
      if len(l_index)>0:
        Par_sigma_lower[i,par] =0.5*(np.sqrt(np.sum((nflat_samples[l_index,par] - nflat_samples[best_index,par])**2)/len(l_index)))
      if len(h_index)>0:
        Par_sigma_upper[i,par] =0.5*(np.sqrt(np.sum((nflat_samples[h_index,par] - nflat_samples[best_index,par])**2)/len(h_index)))

    Par_median[i] = new_sfr[1]
    Par_mle[i] = new_sfr[0]
    Par_mode[i] = new_sfr[2]
    Mass_bc03_mle[i] = get_Mbc03(new_sfr[0,-1],DATA_4SSP)
    Mass_bc03_mode[i] = get_Mbc03(new_sfr[2,-1],DATA_4SSP)
    Mass_bc03_median[i] = get_Mbc03(new_sfr[1,-1],DATA_4SSP)
    comb_all = np.array(list(combinations(Beg_Age[1:-2]+0.001,3)))
    time1=comb_all[4,0]
    time2=comb_all[4,1]
    time3=comb_all[4,2]
    #Calculate mass
    Z_all= np.array([-2.2490,-1.6464,-0.6392,-0.3300,0.0932,0.5595])

    
    t1p = np.where(Beg_Age<time1)
    t2p = np.where((Beg_Age>time1) & (Beg_Age<time2))
    t3p = np.where((Beg_Age>time2) & (Beg_Age<time3))
    t4p = np.where(Beg_Age>time3)
         
    Rem_mass = np.zeros(np.shape(nflat_lnprob))
    R1_all = np.zeros(np.shape(nflat_lnprob))
    R2_all = np.zeros(np.shape(nflat_lnprob))
    R3_all = np.zeros(np.shape(nflat_lnprob))
    TF_all = np.zeros(np.shape(nflat_lnprob))
    TB_all = np.zeros(np.shape(nflat_lnprob))
    TM_all = np.zeros(np.shape(nflat_lnprob))
    iter=0
    for iter in range (0,len(nflat_samples)):
            Rem_mass[iter] = np.log10(nsa_mass - np.sum(10**(nflat_samples[iter,:-3])))
            M_today_all = 10**(np.append(nflat_samples[iter,:-3],Rem_mass[iter]))
            Best_Metal_all = nflat_samples[iter,-1]
            IN_H_all = np.where(Z_all>Best_Metal_all)[0][0]
            IN_L_all = np.where(Z_all<Best_Metal_all)[0][-1]
            M2_all = DATA_4SSP[IN_H_all,:,6]
            M1_all = DATA_4SSP[IN_L_all,:,6]
            bc03_Mall = (M1_all*(Z_all[IN_H_all]-Best_Metal_all) + M2_all*(Best_Metal_all-Z_all[IN_L_all]))/(Z_all[IN_H_all]-Z_all[IN_L_all])
            M_total_all = M_today_all/bc03_Mall
            TM_all[iter] = np.sum(M_today_all*Ages)/np.sum(M_today_all)
            l_tm = np.where(TM_all<TM_all[best_index])[0]
            h_tm = np.where(TM_all>TM_all[best_index])[0]
            if len(l_tm)>0:
               TMASS_lower[i] = 0.5*np.sqrt(np.sum((TM_all[l_tm] - TM_all[best_index])**2)/len(l_tm))
            if len(h_tm)>0:
               TMASS_upper[i] = 0.5*np.sqrt(np.sum((TM_all[h_tm] - TM_all[best_index])**2)/len(h_tm))
            SFH_all = M_total_all/delta_T
            R1_all[iter] = np.log10((np.sum(M_total_all[t1p])/np.sum(delta_T[t1p]))/(np.sum(M_total_all[t2p])/np.sum(delta_T[t2p])))
            lr1 = np.where(R1_all<R1_all[best_index])[0]
            hr1 = np.where(R1_all>R1_all[best_index])[0]
            if len(lr1)>0:
               SR1_lower_all[i] =0.5*np.sqrt(np.sum((R1_all[lr1] - R1_all[best_index])**2)/len(lr1))
            if len(hr1)>0:
               SR1_upper_all[i] =0.5*np.sqrt(np.sum((R1_all[hr1] - R1_all[best_index])**2)/len(hr1))
            R2_all[iter] =  np.log10((np.sum(M_total_all[t3p])/np.sum(delta_T[t3p]))/(np.sum(M_total_all[t4p])/np.sum(delta_T[t4p])))
            lr2 = np.where(R2_all<R2_all[best_index])[0]
            hr2 = np.where(R2_all>R2_all[best_index])[0]
            if len(lr2)>0:
               SR2_lower_all[i] =0.5*np.sqrt(np.sum((R2_all[lr2] - R2_all[best_index])**2)/len(lr2))
            if len(hr2)>0:
               SR2_upper_all[i] =0.5*np.sqrt(np.sum((R2_all[hr2] - R2_all[best_index])**2)/len(hr2))
            R3_all[iter] = np.log10((np.sum(M_total_all[np.c_[t1p,t2p][0]])/np.sum(delta_T[np.c_[t1p,t2p][0]]))/(np.sum(M_total_all[np.c_[t3p,t4p][0]])/np.sum(delta_T[np.c_[t3p,t4p][0]])))
            lr3 = np.where(R3_all<R3_all[best_index])[0]
            hr3 = np.where(R3_all>R3_all[best_index])[0]
            if len(lr3)>0:
               SR3_lower_all[i] =0.5*np.sqrt(np.sum((R3_all[lr3] - R3_all[best_index])**2)/len(lr3))
            if len(hr3)>0:
               SR3_upper_all[i] =0.5*np.sqrt(np.sum((R3_all[hr3] - R3_all[best_index])**2)/len(hr3))
            cumulative_mass = np.cumsum(M_today_all)
            TF_all[iter] = np.interp(0.5,cumulative_mass/nsa_mass,Ages)
            Max_index = np.argmax(SFH_all)
            TB_all[iter] = Ages[Max_index]
    TMASS[i] = TM_all[best_index]
    R1_mle[i] = R1_all[best_index]
    R2_mle[i] = R2_all[best_index]
    R3_mle[i] = R3_all[best_index]
    TF_mle[i] = TF_all[best_index]
    TB_mle[i] = TB_all[best_index]
    R1_median[i] = np.median(R1_all)
    R2_median[i] = np.median(R2_all)
    R3_median[i] = np.median(R3_all)
    TF_median[i] = np.median(TF_all)
    TB_median[i] = np.median(TB_all)
    TF_sigma[i] = np.std(TF_all)
    TB_sigma[i] = np.std(TB_all)
    PAR_ALL = np.c_[R1_all,R2_all,R3_all,TF_all,TB_all]
    R1_mode[i],R2_mode[i],R3_mode[i],TF_mode[i],TB_mode[i] = stats.mode(PAR_ALL)[0][0]
    SR1_all[i] = np.std(R1_all)
    SR2_all[i] = np.std(R2_all)
    SR3_all[i] = np.std(R3_all)
    DELTAT_ALL[i,:] = delta_T
    T1[i] = Beg_Age[t1p][-1]
    T2[i] = Beg_Age[t2p][-1]
    T3[i] = Beg_Age[t3p][-1]

 col1 = Line0
 col2 = R1_mle
 col3 = R2_mle
 col4 = R3_mle
 col5 = R1_mode
 col6 = R2_mode
 col7 = R3_mode
 col8 = R1_median
 col9 = R2_median
 col10 = R3_median
 col11 = SR1_all
 col12 = SR2_all
 col13 = SR3_all
 col14 = Par_mle
 col15 = Par_mode
 col16 = Par_median
 col17 = Mass_bc03_mle
 col18 = Mass_bc03_mode
 col19 = Mass_bc03_median
 col20 = DELTAT_ALL
 col21 = TF_mle
 col22 = TF_mode
 col23 = TF_median
 col24 = TF_sigma
 col25 = PARS_TAU
 col26 = N_steps
 col27 = T1
 col28 = T2
 col29 = T3
 col30 = MASS_ALL
 col31 = E_LINES
 col32 = E_FLUX
 col33 = SERSIC
 col34 = TB_mle
 col35 = TB_mode
 col36 = TB_median
 col37 = TB_sigma
 col38 = Par_sigma_lower
 col39 = Par_sigma_upper
 col40 = SR1_lower_all
 col41 = SR1_upper_all
 col42 = SR2_lower_all
 col43 = SR2_upper_all
 col44 = SR3_lower_all
 col45 = SR3_upper_all
 col46 = TMASS
 col47 = TMASS_lower
 col48 = TMASS_upper

 c1 = fits.Column(name='MaNGAID',format='10A',array=col1)
 c2 = fits.Column(name='R1_mle',format='E',array=col2)
 c3 = fits.Column(name='R2_mle',format='E',array=col3)
 c4 = fits.Column(name='R3_mle',format='E',array=col4)
 c5 = fits.Column(name='R1_mode',format='E',array=col5)
 c6 = fits.Column(name='R2_mode',format='E',array=col6)
 c7 = fits.Column(name='R3_mode',format='E',array=col7)
 c8 = fits.Column(name='R1_median',format='E',array=col8)
 c9 = fits.Column(name='R2_median',format='E',array=col9)
 c10 = fits.Column(name='R3_median',format='E',array=col10)
 c11 = fits.Column(name='R1_sigma',format='E',array=col11)
 c12 = fits.Column(name='R2_sigma',format='E',array=col12)
 c13 = fits.Column(name='R3_sigma',format='E',array=col13)
 c14 = fits.Column(name='Pars_mle',format='10E',array=col14)
 c15 = fits.Column(name='Pars_mode',format='10E',array=col15)
 c16 = fits.Column(name='Pars_median',format='10E',array=col16)
 c17 = fits.Column(name='M_bc03_mle',format='8E',array=col17)
 c18 = fits.Column(name='M_bc03_mode',format='8E',array=col18)
 c19 = fits.Column(name='M_bc03_median',format='8E',array=col19)
 c20 = fits.Column(name='dT',format='8E',array=col20)
 c21 = fits.Column(name='Tf_mle',format='E',array=col21)
 c22 = fits.Column(name='Tf_mode',format='E',array=col22)
 c23 = fits.Column(name='Tf_median',format='E',array=col23)
 c24 = fits.Column(name='Tf_sigma',format='E',array=col24)
 c25 = fits.Column(name='auto_corr',format='10E',array=col25)
 c26 = fits.Column(name='N_steps',format='E',array=col26)
 c27 = fits.Column(name='t1',format='E',array=col27)
 c28 = fits.Column(name='t2',format='E',array=col28)
 c29 = fits.Column(name='t3',format='E',array=col29)
 c30 = fits.Column(name='nsa_mass',format='E',array=col30)
 c31 = fits.Column(name='ELINE EW',format='22E',array=col31)
 c32 = fits.Column(name='ELINE Flux',format='22E',array=col32)
 c33 = fits.Column(name='sersic',format='E',array=col33)
 c34 = fits.Column(name='Tb_mle',format='E',array=col34)
 c35 = fits.Column(name='Tb_mode',format='E',array=col35)
 c36 = fits.Column(name='Tb_median',format='E',array=col36)
 c37 = fits.Column(name='Tb_sigma',format='E',array=col37)
 c38 = fits.Column(name='Par_sigma_lower',format='10E',array=col38)
 c39 = fits.Column(name='Par_sigma_upper',format='10E',array=col39)
 c40 = fits.Column(name='R1_sigma_lower',format='E',array=col40)
 c41 = fits.Column(name='R1_sigma_upper',format='E',array=col41)
 c42 = fits.Column(name='R2_sigma_lower',format='E',array=col42)
 c43 = fits.Column(name='R2_sigma_upper',format='E',array=col43)
 c44 = fits.Column(name='R3_sigma_lower',format='E',array=col44)
 c45 = fits.Column(name='R3_sigma_upper',format='E',array=col45)
 c46 = fits.Column(name='T mass Gyr',format='E',array=col46)
 c47 = fits.Column(name='T mass lower',format='E',array=col47)
 c48 = fits.Column(name='T mass upper',format='E',array=col48)



 hdu = fits.BinTableHDU.from_columns([c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15,c16,c17,c18,c19,c20,c21,c22,c23,c24,c25,c26,c27,c28,c29,c30,c31,c32,c33,c34,c35,c36,c37,c38,c39,c40,c41,c42,c43,c44,c45,c46,c47,c48])
 hdu.writeto(path_store+'R1R2_binned_'+str(bin_index)+'.fits')
