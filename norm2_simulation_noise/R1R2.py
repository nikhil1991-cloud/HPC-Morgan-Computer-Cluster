import numpy as np
from matplotlib import pyplot as plt
from astropy.io import fits
from scipy import stats
import pandas as pd
import math
import site
import sys
import os
#os.chdir('/Users/nikhil/Downloads/')
from simulation_model import make_models,get_Mbc03,get_dt
site.addsitedir('/home/naj222')
import emcee
import corner
from itertools import combinations
import extinction

path_input = '/mnt/gpfs2_4m/scratch/naj222/dir1/input/'
path_output = '/mnt/gpfs2_4m/scratch/naj222/dir1/output/simulation_noise/'
path_store = '/mnt/gpfs2_4m/scratch/naj222/dir1/output/norm2_R1R2/'

#path_input='/Users/nikhil/Data/emcee_test/Input/'    #Comment on cluster
#path_output='/Users/nikhil/Data/emcee_test/Output/'  #Comment on cluster
#path_plots='/Users/nikhil/Data/emcee_test/'          #Comment on cluster

#specify path to text file that contains MaNGA IDs of galaxies with AGN cut and axis ratio cut
with open(path_input+'sample_txt/swim_ba_agn_cut.txt') as f:
     Line0 = [line.rstrip('\n') for line in open(path_input+'sample_txt/swim_ba_agn_cut.txt')]
     
Z_all= np.array([-2.2490,-1.6464,-0.6392,-0.3300,0.0932,0.5595])
fraction_array = [0,1,2,3,4,5,6,7,8,9,10]
f=0
for f in range (0,len(fraction_array)):
 fraction = fraction_array[f]
 R1_median = np.zeros(5)
 R2_median = np.zeros(5)
 R3_median = np.zeros(5)
 R1_mode = np.zeros(5)
 R2_mode = np.zeros(5)
 R3_mode = np.zeros(5)
 R1_mle = np.zeros(5)
 R2_mle = np.zeros(5)
 R3_mle = np.zeros(5)
 R1_sim = np.zeros(5)
 R2_sim = np.zeros(5)
 R3_sim = np.zeros(5)
 R1_fsim = np.zeros(5)
 R2_fsim = np.zeros(5)
 R3_fsim = np.zeros(5)
 SR1_all = np.zeros(5)
 SR2_all = np.zeros(5)
 SR3_all = np.zeros(5)
 SR1_cut_all = np.zeros(5)
 SR2_cut_all = np.zeros(5)
 SR3_cut_all = np.zeros(5)
 LICK_ALL = np.zeros((5,42))
 D4000_ALL = np.zeros(5)
 COLORS_ALL = np.zeros((5,6))
 Par_median = np.zeros((5,10))
 Par_mle = np.zeros((5,10))
 Par_mode = np.zeros((5,10))
 Mass_bc03_median = np.zeros((5,8))
 Mass_bc03_mle = np.zeros((5,8))
 Mass_bc03_mode = np.zeros((5,8))
 M_SIM = np.zeros((5,8))
 BC03_SIM = np.zeros((5,8))
 DELTAT_ALL = np.zeros((5,8))
 M_FSIM = np.zeros((5,500))
 BC03_FSIM = np.zeros((5,500))
 DELTATF_ALL = np.zeros((5,500))
 MASS_ALL = np.zeros(5)
 T1 = np.zeros(5)
 T2 = np.zeros(5)
 T3 = np.zeros(5)
 PARS_TAU = np.zeros((5,10))
 N_steps = np.zeros(5)
 EFOLD = np.zeros(5)
 TF_SIM = np.zeros(5)
 TF_FSIM = np.zeros(5)
 TF_DAT_MLE = np.zeros(5)
 TF_DAT_MED = np.zeros(5)
 TF_DAT_MOD = np.zeros(5)
 TF_DAT_SIG = np.zeros(5)


 t_index = np.array([5,9,13,17,21,25,27,29]) #New scheme
 COLOR_4SSP = fits.open(path_input+'ssp/ssp_bc03_4.color.fits')
 DATA_4SSP = COLOR_4SSP[0].data[:,t_index,:]
 #BC03 Ages
 e_folding_time = np.array([1,2,4,8,16])
 eTau = 0
 for eTau in range (0,len(e_folding_time)):
    e_fold_tau = e_folding_time[eTau]
    drpall = fits.open(path_input+'drpall/drpall-v2_3_1.fits')
    tbdata = drpall[1].data
    indx = np.where(tbdata['mangaid'] == Line0[0])
    nsa_mass = tbdata['nsa_elpetro_mass'][indx][0]
    zeta = tbdata['nsa_z'][indx][0] #Redshift
    sersic = tbdata['nsa_sersic_n'][indx][0]
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
    delta_lambda_lick = DATA[11].data
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
    #Calculate delta_T_fine
    t_fine = np.logspace(np.log10(t_age[0]),np.log10(t_age[-1]),500)
    T_age_fine = np.log10(t_fine)
    Tbeg_fine = np.zeros(np.shape(T_age_fine))
    Tmid_fine = (T_age_fine[1:] + T_age_fine[:-1]) / 2
    Tbeg_fine[0:len(t_fine)-1] = Tmid_fine
    Tbeg_fine[-1] = np.log10(14)
    Beg_Age_fine = 10**(Tbeg_fine)
    
    comb_all = np.array(list(combinations(Beg_Age[1:-2]+0.001,3)))
    
    Simulation_Ebv_1 = 0.5
    Simulation_Ebv_2 = 0.05
    Simulation_Z = 0.0
    
    dt = get_dt(t_age)
    bc03_M = get_Mbc03(Simulation_Z,DATA_4SSP)
    Simulated_SFH = np.exp(-(14-t_age)/e_fold_tau)
    Simulated_mass_prime = Simulated_SFH*(dt*bc03_M)
    Simulated_mass = (Simulated_mass_prime/np.sum(Simulated_mass_prime))*nsa_mass
    Simulated_total_mass = Simulated_mass/bc03_M
    #Simulation_mass = (np.exp(t_age/e_fold_tau)/np.sum(np.exp(t_age/e_fold_tau)))*nsa_mass
    Log_Mass = np.log10(Simulated_mass)
    Initial_SFH = np.append(np.append(np.append(Log_Mass,Simulation_Ebv_1),Simulation_Ebv_2),Simulation_Z)
    Simulated_Log_SFH = np.log10(10**(Initial_SFH[:-3])/(bc03_M*dt))
    Sim_D4,Sim_Lick,Sim_Mag = make_models(Initial_SFH,t_age,CI,CR,CB,D4,Photo,PhotoAB,L_R,L_B,L_I,D_I,EBV_I,Metal_I,D_lmd,nsa_mass,DATA_4SSP)
  
  
    # fine 500 bins SFH
    t_fine = np.logspace(np.log10(t_age[0]),np.log10(t_age[-1]),500)
    dt_fine = get_dt(t_fine)
    bc03_M_fine = np.interp(t_fine,t_age,bc03_M)
    Simulated_SFH_fine = np.exp(t_fine/e_fold_tau)
    Simulated_mass_prime_fine = Simulated_SFH_fine*(dt_fine*bc03_M_fine)
    Simulated_mass_fine = (Simulated_mass_prime_fine/np.sum(Simulated_mass_prime_fine))*nsa_mass
    Simulated_total_mass_fine = Simulated_mass_fine/bc03_M_fine
  
     #calculation of formation time
    cumulative_m = np.cumsum(Simulated_mass)
    cumulative_mf = np.cumsum(Simulated_mass_fine)

    #formation times for fine and discreet SFH
    TF_SIM[eTau] = np.interp(0.5,cumulative_m/nsa_mass,t_age)
    TF_FSIM[eTau] = np.interp(0.5,cumulative_mf/nsa_mass,t_fine)
    M_SIM[eTau] = Simulated_total_mass
    BC03_SIM[eTau] = bc03_M
    DELTAT_ALL[eTau] = dt
    M_FSIM[eTau] = Simulated_total_mass_fine
    BC03_FSIM[eTau] = bc03_M_fine
    DELTATF_ALL[eTau] = dt_fine
    
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
    
    #Read Lick indices and d4000 from data
    m1_list,m2_list = [0,0,4,4,3,3],[3,1,6,5,6,7]  #w2-u,w2-w1,g-i,g-r,u-i,u-z
    mags_info = np.array(['W2','W1','M2','u','g','r','i','z'])
    Mags_data = Sim_Mag
    Dn4000_data = Sim_D4
    Lick_indices_data = Sim_Lick
    D4000_ALL[eTau] = Dn4000_data
    LICK_ALL[eTau,:] = Lick_indices_data
    COLORS_ALL[eTau,:] = Mags_data[m1_list] - Mags_data[m2_list]
    MASS_ALL[eTau] = nsa_mass
    EFOLD[eTau] = e_fold_tau

    #Read emcee output files
    reader = emcee.backends.HDFBackend(path_output+'_'+str(e_fold_tau)+'_'+'sigma_'+str(fraction))
    tau = reader.get_autocorr_time(tol=0)
    PARS_TAU[eTau,:] = tau
    chain_len = len(reader.get_chain())
    N_steps[eTau] = chain_len
    nburnin= int(2*tau.max())
    thin = int(0.5*tau.min())
    bins=20
    
    #get flat samples of parameters and ln_probability
    nflat_samples = reader.get_chain(discard=nburnin,thin=thin,flat=True) #Chain thinning
    nflat_lnprob = reader.get_log_prob(discard=nburnin,thin=thin,flat=True)
    nchisq = (nflat_lnprob/(-0.5))/39
    best_index = np.where(nchisq==nchisq.min())[0][0]
    new_sfr = np.zeros((3,np.shape(nflat_samples)[1]))
    new_sfr[0,:] = nflat_samples[best_index,:] #mle
    new_sfr[1,:] = np.median(nflat_samples,axis=0) #median
    new_sfr[2,:] = stats.mode(nflat_samples)[0][0] #mode
    Par_median[eTau] = new_sfr[0]
    Par_mle[eTau] = new_sfr[1]
    Par_mode[eTau] = new_sfr[2]
    Mass_bc03_mle[eTau] = get_Mbc03(new_sfr[0,-1],DATA_4SSP)
    Mass_bc03_mode[eTau] = get_Mbc03(new_sfr[2,-1],DATA_4SSP)
    Mass_bc03_median[eTau] = get_Mbc03(new_sfr[1,-1],DATA_4SSP)
    
    
    time1=comb_all[4,0]
    time2=comb_all[4,1]
    time3=comb_all[4,2]

    #for discreet bins
    t1p = np.where(Beg_Age<time1)
    t2p = np.where((Beg_Age>time1) & (Beg_Age<time2))
    t3p = np.where((Beg_Age>time2) & (Beg_Age<time3))
    t4p = np.where(Beg_Age>time3)
    #for fine bins
    t1f = np.where(Beg_Age_fine<time1)
    t2f = np.where((Beg_Age_fine>time1) & (Beg_Age_fine<time2))
    t3f = np.where((Beg_Age_fine>time2) & (Beg_Age_fine<time3))
    t4f = np.where(Beg_Age_fine>time3)
    #tau = reader.get_autocorr_time(quiet=True)
    Rem_mass = np.zeros(np.shape(nflat_lnprob))
    R1_all = np.zeros(np.shape(nflat_lnprob))
    R2_all = np.zeros(np.shape(nflat_lnprob))
    R3_all = np.zeros(np.shape(nflat_lnprob))
    TF_all = np.zeros(np.shape(nflat_lnprob))
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
            R1_all[iter] = np.log10((np.sum(M_total_all[t1p])/np.sum(delta_T[t1p]))/(np.sum(M_total_all[t2p])/np.sum(delta_T[t2p])))
            R2_all[iter] =  np.log10((np.sum(M_total_all[t3p])/np.sum(delta_T[t3p]))/(np.sum(M_total_all[t4p])/np.sum(delta_T[t4p])))
            R3_all[iter] = np.log10((np.sum(M_total_all[np.c_[t1p,t2p][0]])/np.sum(delta_T[np.c_[t1p,t2p][0]]))/(np.sum(M_total_all[np.c_[t3p,t4p][0]])/np.sum(delta_T[np.c_[t3p,t4p][0]])))
            cumulative_mass = np.cumsum(M_today_all)
            TF_all[iter] = np.interp(0.5,cumulative_mass/nsa_mass,t_age)

    R1_mle[eTau] = R1_all[best_index]
    R2_mle[eTau] = R2_all[best_index]
    R3_mle[eTau] = R3_all[best_index]
    TF_DAT_MLE[eTau] = TF_all[best_index]
    R1_median[eTau] = np.median(R1_all)
    R2_median[eTau] = np.median(R2_all)
    R3_median[eTau] = np.median(R3_all)
    TF_DAT_MED[eTau] = np.median(TF_all)
    TF_DAT_SIG[eTau] = np.std(TF_all)
    PAR_ALL = np.c_[R1_all,R2_all,R3_all,TF_all]
    R1_mode[eTau],R2_mode[eTau],R3_mode[eTau],TF_DAT_MOD[eTau] = stats.mode(PAR_ALL)[0][0]
    SR1_all[eTau] = np.std(R1_all)
    SR2_all[eTau] = np.std(R2_all)
    SR3_all[eTau] = np.std(R3_all)

    R1_sim[eTau] = np.log10((np.sum(Simulated_total_mass[t1p])/np.sum(dt[t1p]))/(np.sum(Simulated_total_mass[t2p])/np.sum(dt[t2p])))
    R2_sim[eTau] = np.log10((np.sum(Simulated_total_mass[t3p])/np.sum(dt[t3p]))/(np.sum(Simulated_total_mass[t4p])/np.sum(dt[t4p])))
    R3_sim[eTau] = np.log10((np.sum(Simulated_total_mass[np.c_[t1p,t2p][0]])/np.sum(dt[np.c_[t1p,t2p][0]]))/(np.sum(Simulated_total_mass[np.c_[t3p,t4p][0]])/np.sum(dt[np.c_[t3p,t4p][0]])))
  
    R1_fsim[eTau] = np.log10((np.sum(Simulated_total_mass_fine[t1f])/np.sum(dt_fine[t1f]))/(np.sum(Simulated_total_mass_fine[t2f])/np.sum(dt_fine[t2f])))
    R2_fsim[eTau] = np.log10((np.sum(Simulated_total_mass_fine[t3f])/np.sum(dt_fine[t3f]))/(np.sum(Simulated_total_mass_fine[t4f])/np.sum(dt_fine[t4f])))
    R3_fsim[eTau] = np.log10((np.sum(Simulated_total_mass_fine[np.c_[t1f,t2f][0]])/np.sum(dt_fine[np.c_[t1f,t2f][0]]))/(np.sum(Simulated_total_mass_fine[np.c_[t3f,t4f][0]])/np.sum(dt_fine[np.c_[t3f,t4f][0]])))
 
         
    T1[eTau] = Beg_Age[t1p][-1]
    T2[eTau] = Beg_Age[t2p][-1]
    T3[eTau] = Beg_Age[t3p][-1]
         
 



 col1 = e_folding_time
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
 col20 = M_SIM
 col21 = M_FSIM
 col22 = DELTAT_ALL
 col23 = DELTATF_ALL
 col24 = BC03_SIM
 col25 = BC03_FSIM
 col26 = TF_SIM
 col27 = TF_FSIM
 col28 = TF_DAT_MLE
 col29 = TF_DAT_MOD
 col30 = TF_DAT_MED
 col31 = PARS_TAU
 col32 = N_steps
 col33 = T1
 col34 = T2
 col35 = T3
 col36 = TF_DAT_SIG
 col37 = R1_sim
 col38 = R2_sim
 col39 = R3_sim
 col40 = R1_fsim
 col41 = R2_fsim
 col42 = R3_fsim

 c1 = fits.Column(name='e_fold',format='E',array=col1)
 c2 = fits.Column(name='R1_mle',format='E',array=col2)
 c3 = fits.Column(name='R2_mle',format='E',array=col3)
 c4 = fits.Column(name='R3_mle',format='E',array=col4)
 c5 = fits.Column(name='R1_mode',format='E',array=col5)
 c6 = fits.Column(name='R2_mode',format='E',array=col6)
 c7 = fits.Column(name='R3_mode',format='E',array=col7)
 c8 = fits.Column(name='R1_median',format='E',array=col8)
 c9 = fits.Column(name='R2_median',format='E',array=col9)
 c10 = fits.Column(name='R3_median',format='E',array=col10)
 c11 = fits.Column(name='sigma_R1',format='E',array=col11)
 c12 = fits.Column(name='sigma_R2',format='E',array=col12)
 c13 = fits.Column(name='sigma_R3',format='E',array=col13)
 c14 = fits.Column(name='mle_parameters',format='10E',array=col14)
 c15 = fits.Column(name='mode_parameters',format='10E',array=col15)
 c16 = fits.Column(name='median_parameters',format='10E',array=col16)
 c17 = fits.Column(name='frac_mass_mle',format='8E',array=col17)
 c18 = fits.Column(name='frac_mass_mode',format='8E',array=col18)
 c19 = fits.Column(name='frac_mass_median',format='8E',array=col19)
 c20 = fits.Column(name='Mass_simulation',format='8E',array=col20)
 c21 = fits.Column(name='Mass_fsimulation',format='500E',array=col21)
 c22 = fits.Column(name='dt',format='8E',array=col22)
 c23 = fits.Column(name='dt_f',format='500E',array=col23)
 c24 = fits.Column(name='frac_mass_sim',format='8E',array=col24)
 c25 = fits.Column(name='frac_mass_fsim',format='500E',array=col25)
 c26 = fits.Column(name='tf_sim',format='E',array=col26)
 c27 = fits.Column(name='tf_fsim',format='E',array=col27)
 c28 = fits.Column(name='tf_mle',format='E',array=col28)
 c29 = fits.Column(name='tf_mode',format='E',array=col29)
 c30 = fits.Column(name='tf_median',format='E',array=col30)
 c31 = fits.Column(name='autocorr',format='10E',array=col31)
 c32 = fits.Column(name='N_steps',format='E',array=col32)
 c33 = fits.Column(name='T1',format='E',array=col33)
 c34 = fits.Column(name='T2',format='E',array=col34)
 c35 = fits.Column(name='T3',format='E',array=col35)
 c36 = fits.Column(name='tf_sig',format='E',array=col36)
 c37 = fits.Column(name='R1_sim',format='E',array=col37)
 c38 = fits.Column(name='R2_sim',format='E',array=col38)
 c39 = fits.Column(name='R3_sim',format='E',array=col39)
 c40 = fits.Column(name='R1_fsim',format='E',array=col40)
 c41 = fits.Column(name='R2_fsim',format='E',array=col41)
 c42 = fits.Column(name='R3_fsim',format='E',array=col42)

 
 
 hdu = fits.BinTableHDU.from_columns([c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15,c16,c17,c18,c19,c20,c21,c22,c23,c24,c25,c26,c27,c28,c29,c30,c31,c32,c33,c34,c35,c36,c37,c38,c39,c40,c41,c42])
 hdu.writeto(path_store+'R1R2_sim_'+str(fraction)+'.fits')
