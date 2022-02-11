import numpy as np
from matplotlib import pyplot as plt
from astropy.io import fits
from scipy import stats
import pandas as pd
import math
import os
os.chdir('/Users/nikhil/code/mcc/code/norm2_simulation_noise') #Comment on cluster
from plot_functions_norm2 import make_mods
from simulation_model import make_models,get_Mbc03,get_dt
import site
import sys
#site.addsitedir('/home/naj222') #Uncomment on cluster
import emcee
import corner
from itertools import combinations

#path_input = '/mnt/gpfs2_4m/scratch/naj222/dir1/input/'                #Uncomment on cluster
#path_output = '/mnt/gpfs2_4m/scratch/naj222/dir1/output/simulation_noise/'#Uncomment on cluster
#path_plots = '/mnt/gpfs2_4m/scratch/naj222/dir1/plots/simulation_noise/'      #Uncomment on cluster

path_input='/Users/nikhil/Data/emcee_test/Input/'    #Comment on cluster
path_output='/Users/nikhil/Data/emcee_test/Output/'  #Comment on cluster
path_plots='/Users/nikhil/Data/emcee_test/'          #Comment on cluster
#specify path to text file that contains MaNGA IDs of galaxies with AGN cut and axis ratio cut
with open(path_input+'sample_txt/swim_ba_agn_cut.txt') as f:
     Line0 = [line.rstrip('\n') for line in open(path_input+'sample_txt/swim_ba_agn_cut.txt')]
    
Z_all= np.array([-2.2490,-1.6464,-0.6392,-0.3300,0.0932,0.5595])
#Read Lick indices
mf = pd.read_csv(path_input+'ssp/spec1.txt', comment='#', header=None, delim_whitespace=True)
Indices = np.array(mf[2])
H_Blue = np.array(mf[9])
L_Blue = np.array(mf[8])
H_Red = np.array(mf[13])
L_Red = np.array(mf[12])
H_Index = np.array(mf[5])
L_Index = np.array(mf[4])
Unit_indx = np.array(mf[16])
mag_index = Unit_indx == 'mag'
Rmid = (H_Red + L_Red)/2
Bmid = (H_Blue + L_Blue)/2
Imid = (H_Index + L_Index)/2
    
t_index = np.array([5,9,13,17,21,25,27,29]) #New Scheme
COLOR_4SSP = fits.open(path_input+'ssp/ssp_bc03_4.color.fits')
DATA_4SSP = COLOR_4SSP[0].data[:,t_index,:]
#BC03 Ages

Z_prime = 0.0
e_folding_time = np.array([1,2,4,8,16])
fraction_array = [0,1,2,3,4,5,6,7,8,9,10]
f=0
for f in range (0,1):#len(fraction_array)):
 fraction = fraction_array[f]
 e=0
 for e in range (0,1):#len(e_folding_time)):
  e_fold_tau = e_folding_time[e]
  #get galaxy info from drpall
  drpall = fits.open(path_input+'drpall/drpall-v2_3_1.fits')
  tbdata = drpall[1].data
  indx = np.where(tbdata['mangaid'] == Line0[0])
  nsa_mass = tbdata['nsa_elpetro_mass'][indx][0]
  zeta = tbdata['nsa_z'][indx][0] #Redshift
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
  #Open the data file
  Simulation_Ebv_1 = 0.5
  Simulation_Ebv_2 = 0.05
  Simulation_Z = Z_prime
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
  
  
  #fine 500 bins SFH
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
  TF_sim = np.interp(0.5,cumulative_m/nsa_mass,t_age)
  TF_fsim = np.interp(0.5,cumulative_mf/nsa_mass,t_fine)

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
        
  m1_list,m2_list = [0,0,4,4,3,3],[3,1,6,5,6,7]  #w2-u,w1-u,g-i,g-r,u-i,u-z
  mags_info = np.array(['W2','W1','M2','u','g','r','i','z'])
  #Read mags from data
  Mags_data = Sim_Mag
  data_color = Mags_data[m1_list] - Mags_data[m2_list]
  #Read Lick indices and d4000 from data
  Dn4000_data = Sim_D4
  Lick_indices_data = Sim_Lick
  Lick_mag = -2.5*np.log10((delta_lambda_lick[mag_index]-Lick_indices_data[mag_index])/delta_lambda_lick[mag_index])
  Lick_ang = Lick_indices_data[~mag_index]
  #Initialize array
  reader = emcee.backends.HDFBackend(path_output+'_'+str(e_fold_tau)+'_'+'sigma_'+str(fraction))
  tau = reader.get_autocorr_time(tol=0)
  chain_len = len(reader.get_chain())
  pars = ['Log($M_{0.01}$)','Log($M_{0.03}$)','Log($M_{0.16}$)','Log($M_{0.4}$)','Log($M_{1.2}$)','Log($M_{4}$)','Log($M_{7.7}$)','$E_{y}$','$E_{o}$','[Fe/H]']
  max_par = np.where(tau==tau.max())[0][0]
  non_converged_par = pars[max_par]
  dof_mcmc = 7
  bins = 20
  #Calculate delta_T (time widths)
  Ages = DATA['Ages'].data[t_index]
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
  #calculate delta_T for fine SFH
  T_age_fine = np.log10(t_fine)
  Tbeg_fine = np.zeros(np.shape(T_age_fine))
  Tmid_fine = (T_age_fine[1:] + T_age_fine[:-1]) / 2
  Tbeg_fine[0:len(t_fine)-1] = Tmid_fine
  Tbeg_fine[-1] = np.log10(14)
  Beg_Age_fine = 10**(Tbeg_fine)
  
  comb_all = np.array(list(combinations(Beg_Age[1:-2]+0.001,3)))
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
  nburnin= int(2*tau.max())
  thin = int(0.5*tau.min())
  chi_limit = [0.5,1.0]
  #get flat samples of parameters and ln_probability
  nflat_samples = reader.get_chain(discard=nburnin,thin=thin,flat=True) #Chain thinning
  nflat_lnprob = reader.get_log_prob(discard=nburnin,thin=thin,flat=True)
  nchisq = (nflat_lnprob/(-0.5))/39
  Rem_mass = np.zeros(np.shape(nflat_lnprob))
  R1_all = np.zeros(np.shape(nflat_lnprob))
  R2_all = np.zeros(np.shape(nflat_lnprob))
  R3_all = np.zeros(np.shape(nflat_lnprob))
  TF_all = np.zeros(np.shape(nflat_lnprob))
  s3_all = np.zeros(np.shape(nflat_lnprob))
  s4_all = np.zeros(np.shape(nflat_lnprob))
  s1_all = np.zeros(np.shape(nflat_lnprob))
  s2_all = np.zeros(np.shape(nflat_lnprob))
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

     s3_all[iter] = np.log10((np.sum(M_total_all[t3p])/np.sum(delta_T[t3p])))
     s4_all[iter] = np.log10((np.sum(M_total_all[t4p])/np.sum(delta_T[t4p])))
     s1_all[iter] = np.log10((np.sum(M_total_all[t1p])/np.sum(delta_T[t1p])))
     s2_all[iter] = np.log10((np.sum(M_total_all[t2p])/np.sum(delta_T[t2p])))
  nflat_SFH = np.c_[nflat_samples[:,:-3],Rem_mass,nflat_samples[:,-3:]]
  good_samples_1 = np.where((nchisq>=nchisq.min())&(nchisq<nchisq.min()+chi_limit[0]))[0]
  good_samples_2 = np.where((nchisq>=nchisq.min())&(nchisq<nchisq.min()+chi_limit[1]))[0]
  
  
  
  R1_sim = np.log10((np.sum(Simulated_total_mass[t1p])/np.sum(dt[t1p]))/(np.sum(Simulated_total_mass[t2p])/np.sum(dt[t2p])))
  R2_sim = np.log10((np.sum(Simulated_total_mass[t3p])/np.sum(dt[t3p]))/(np.sum(Simulated_total_mass[t4p])/np.sum(dt[t4p])))
  R3_sim = np.log10((np.sum(Simulated_total_mass[np.c_[t1p,t2p][0]])/np.sum(dt[np.c_[t1p,t2p][0]]))/(np.sum(Simulated_total_mass[np.c_[t3p,t4p][0]])/np.sum(dt[np.c_[t3p,t4p][0]])))
  
  R1_fsim = np.log10((np.sum(Simulated_total_mass_fine[t1f])/np.sum(dt_fine[t1f]))/(np.sum(Simulated_total_mass_fine[t2f])/np.sum(dt_fine[t2f])))
  R2_fsim = np.log10((np.sum(Simulated_total_mass_fine[t3f])/np.sum(dt_fine[t3f]))/(np.sum(Simulated_total_mass_fine[t4f])/np.sum(dt_fine[t4f])))
  R3_fsim = np.log10((np.sum(Simulated_total_mass_fine[np.c_[t1f,t2f][0]])/np.sum(dt_fine[np.c_[t1f,t2f][0]]))/(np.sum(Simulated_total_mass_fine[np.c_[t3f,t4f][0]])/np.sum(dt_fine[np.c_[t3f,t4f][0]])))
  
  s1_sim = np.log10((np.sum(Simulated_total_mass[t1p])/np.sum(dt[t1p])))
  s2_sim = np.log10((np.sum(Simulated_total_mass[t2p])/np.sum(dt[t2p])))
  s3_sim = np.log10((np.sum(Simulated_total_mass[t3p])/np.sum(dt[t3p])))
  s4_sim = np.log10((np.sum(Simulated_total_mass[t4p])/np.sum(dt[t4p])))
  good_samples = good_samples_1
  flat_samples = nflat_samples[good_samples,:]
  flat_lnprob=nflat_lnprob[good_samples]
  flat_chi =nchisq[good_samples]
  #Get best fit parameters
  new_sfr = np.zeros((3,np.shape(nflat_samples)[1]))
  best_index = np.where(nchisq==nchisq.min())[0][0]
  new_sfr[0,:] = nflat_samples[best_index,:] #mle
  new_sfr[1,:] = np.median(nflat_samples,axis=0) #median
  new_sfr[2,:] = stats.mode(nflat_samples)[0][0] #mode
  statistics = ['MLE','Median','Mode']
  stat_c = ['r','g','b']
  sigma_logsfr = np.zeros(nflat_samples.shape[1])
  upper_limits = np.zeros(nflat_samples.shape[1])
  lower_limits = np.zeros(nflat_samples.shape[1])
  par = 0
  for par in range (0,nflat_samples.shape[1]):
     sigma_logsfr[par] = (np.sqrt(np.sum((np.median(nflat_samples[:,par]) - nflat_samples[:,par])**2)/len(nflat_samples)))
     if np.max(nflat_samples[:,par]) < (np.median(nflat_samples[:,par]) + sigma_logsfr[par]):
        upper_limits[par] = 1
     if np.min(nflat_samples[:,par]) > (np.median(nflat_samples[:,par]) - sigma_logsfr[par]):
        lower_limits[par] = 1
  sigma_logsfr = np.append(sigma_logsfr[:-3]*0.8,0)
  upper_limits = np.append(upper_limits[:-3],0)
  lower_limits = np.append(lower_limits[:-3],0)
  upper_limits.astype(bool)
  lower_limits.astype(bool)


  ##Calculate mass
  SFH = np.zeros(np.shape(new_sfr[:,:-2]))
  Mass_present = np.zeros(np.shape(new_sfr[:,:-2]))
  NW2MU = np.zeros(3)
  NW2W1 = np.zeros(3)
  NGMR = np.zeros(3)
  NUMI = np.zeros(3)
  NUMZ = np.zeros(3)
  ND4 = np.zeros(3)
  NGMI = np.zeros(3)
  NHD = np.zeros(3)
  NFE = np.zeros(3)
  NALICK_ang = np.zeros((len(Lick_ang),3))
  NALICK_mag = np.zeros((len(Lick_mag),3))
  diff_lick_ang = np.zeros((len(Lick_ang),3))
  diff_lick_mag = np.zeros((len(Lick_mag),3))
  diff_colors = np.zeros((6,3))
  diff_D4 = np.zeros(3)
  stat=0
  for stat in range (0,3):
    Best_Metal = new_sfr[stat,-1]
    IN_H = np.where(Z_all>Best_Metal)[0][0]
    IN_L = np.where(Z_all<Best_Metal)[0][-1]
    M2 = DATA_4SSP[IN_H,:,6]
    M1 = DATA_4SSP[IN_L,:,6]
    bc03_M = (M1*(Z_all[IN_H]-Best_Metal) + M2*(Best_Metal-Z_all[IN_L]))/(Z_all[IN_H]-Z_all[IN_L])
    #Calculate mass
    best_SM = 10**(new_sfr[stat,:-3])
    last_SM = nsa_mass - np.sum(best_SM)
    All_SM = np.append(best_SM,last_SM)
    Mass_present[stat,:] = All_SM
    SFH[stat,:] = np.log10(All_SM/(bc03_M*delta_T))
    SFH_sigma = sigma_logsfr
    #generate best fit obs
    ND4[stat],NL,NM = make_mods(new_sfr[stat],DATA,t_index,nsa_mass,DATA_4SSP)
    NALICK_mag[:,stat] = (-2.5*np.log10((delta_lambda_lick[mag_index]-NL[mag_index])/delta_lambda_lick[mag_index]))
    NALICK_ang[:,stat] = NL[~mag_index]
    diff_lick_ang[:,stat] = NALICK_ang[:,stat] - Lick_ang
    diff_lick_mag[:,stat] = NALICK_mag[:,stat] - Lick_mag
    diff_D4[stat] = ND4[stat] - Dn4000_data
    NW2MU[stat] = NM[0] - NM[3]
    NGMI[stat] = NM[4] - NM[6]
    NW2W1[stat] = NM[0] - NM[1]
    NUMI[stat] = NM[3] - NM[6]
    NGMR[stat] = NM[4] - NM[5]
    NUMZ[stat] = NM[3] - NM[7]
    NHD[stat] = NL[21]
    NFE[stat] = NL[4]
    model_bf_colors = np.array([NW2MU[stat],NW2W1[stat],NGMI[stat],NGMR[stat],NUMI[stat],NUMZ[stat]])
    diff_colors[:,stat] = model_bf_colors-data_color
    
  Indices_mag = Indices[mag_index].tolist()
  Indices_ang = Indices[~mag_index].tolist()
  Indices_color = ['W2-u','W2-W1','g-i','g-r','u-i','u-z']
  num_ticks_ang = np.linspace(0,len(Lick_ang)-1,len(Lick_ang))
  num_ticks_mag = np.linspace(0,len(Lick_mag)-1,len(Lick_mag))
  num_ticks_color = np.linspace(0,5,6)
  R1_mle = R1_all[best_index]
  R2_mle = R2_all[best_index]
  R3_mle = R3_all[best_index]
  TF_mle = TF_all[best_index]
  R1_median = np.median(R1_all)
  R2_median = np.median(R2_all)
  R3_median = np.median(R3_all)
  TF_median = np.median(TF_all)
  PAR_ALL = np.c_[R1_all,R2_all,R3_all,TF_all]
  
  R1_mode,R2_mode,R3_mode,TF_mode = stats.mode(PAR_ALL)[0][0]
  s1_mle = s1_all[best_index]
  s2_mle = s2_all[best_index]
  s3_mle = s3_all[best_index]
  s4_mle = s4_all[best_index]
  s1_median = np.median(s1_all)
  s2_median = np.median(s2_all)
  s3_median = np.median(s3_all)
  s4_median = np.median(s4_all)
  #define fontsize, padding etc for plots
  ft_12 =100
  ft_22 =100
  ft_1 = 100
  lw=10
  ltw = 49
  ltwm = 25
  pd=3
    
  fig = plt.figure()

  plt.subplot(3,1,1)
  stat=0
  for stat in range (0,3):
     plt.plot(diff_lick_ang[:,stat],label=statistics[stat],c=stat_c[stat],linewidth=lw)
  num=0
  for num in range (0,len(num_ticks_ang)):
     plt.axvline(x=num_ticks_ang[num],linewidth=lw-4,c='grey')
  plt.axhline(0,linewidth=lw-4,c='grey')
  plt.xticks(num_ticks_ang,Indices_ang,rotation=20,fontsize=ft_12-80)
  #plt.xticklabels(Indices_ang,rotation=20,fontsize=ft_12-80)
  plt.ylabel('$\\Delta$Lick[Ang]',fontsize=ft_12,labelpad=pd, fontweight='bold')
  plt.legend(loc='upper right',fontsize=ft_22,frameon=False)
  plt.minorticks_on()
  plt.tick_params(which='major',length=ltw,width=lw,direction='in',labelsize=ft_1,pad=pd,axis='x',bottom=True,top=True,left=True,right=True)
  plt.tick_params(which='minor',length=ltwm,width=lw,direction='in',labelsize=ft_1,pad=pd,axis='x',bottom=True,top=True,left=True,right=True)
  plt.tick_params(which='major',length=ltw,width=lw,direction='in',labelsize=ft_1,pad=pd,axis='y',bottom=True,top=True,left=True,right=True)
  plt.tick_params(which='minor',length=ltwm,width=lw,direction='in',labelsize=ft_1,pad=pd,axis='y',bottom=True,top=True,left=True,right=True)
  for axis in ['top','bottom','left','right']:
    plt.gca().spines[axis].set_linewidth(lw)
     

  plt.subplot(3,3,5)
  for stat in range (0,3):
     plt.plot(T_age,SFH[stat,:],linewidth=lw,c=stat_c[stat])
  plt.plot(T_age,Simulated_Log_SFH,c='k',linewidth=lw)
  plt.errorbar(T_age,SFH[1,:],yerr=SFH_sigma,ecolor='black',linewidth=lw,capsize=1.2*lw,capthick=lw,fmt='none',uplims=upper_limits,lolims=lower_limits)
  plt.xlabel('Log($t_{age}$)',fontsize=ft_12,labelpad=pd, fontweight='bold')
  plt.ylabel('Log(SFR $M_{\odot}/Gyr$)',fontsize=ft_12,labelpad=pd, fontweight='bold')
  plt.axvline(np.log10(Beg_Age[t1p][-1]),c='k',linewidth=lw-2,label=str(np.round(Beg_Age[t1p][-1],2)))
  plt.axvline(np.log10(Beg_Age[t2p][-1]),c='k',linewidth=lw-2,label=str(np.round(Beg_Age[t2p][-1],2)))
  plt.axvline(np.log10(Beg_Age[t3p][-1]),c='k',linewidth=lw-2,label=str(np.round(Beg_Age[t3p][-1],2)))
  plt.legend(loc='upper left',fontsize=ft_22-30,frameon=False)
  plt.minorticks_on()
  plt.tick_params(which='major',length=ltw,width=lw,direction='in',labelsize=ft_1,pad=pd,axis='x',bottom=True,top=True,left=True,right=True)
  plt.tick_params(which='minor',length=ltwm,width=lw,direction='in',labelsize=ft_1,pad=pd,axis='x',bottom=True,top=True,left=True,right=True)
  plt.tick_params(which='major',length=ltw,width=lw,direction='in',labelsize=ft_1,pad=pd,axis='y',bottom=True,top=True,left=True,right=True)
  plt.tick_params(which='minor',length=ltwm,width=lw,direction='in',labelsize=ft_1,pad=pd,axis='y',bottom=True,top=True,left=True,right=True)
  for axis in ['top','bottom','left','right']:
    plt.gca().spines[axis].set_linewidth(lw)
    
  plt.subplot(3,3,4)
  iter=0
  for iter in range (0,len(flat_samples)):
    sfr = flat_samples[iter,:]
    Best_Z = sfr[-1]
    In_H = np.where(Z_all>Best_Z)[0][0]
    In_L = np.where(Z_all<Best_Z)[0][-1]
    Mas2 = DATA_4SSP[In_H,:,6]
    Mas1 = DATA_4SSP[In_L,:,6]
    m_bc03 = (Mas1*(Z_all[In_H]-Best_Z) + Mas2*(Best_Z-Z_all[In_L]))/(Z_all[In_H]-Z_all[In_L])
    b_SM = 10**(sfr[:-3])
    l_SM = nsa_mass - np.sum(b_SM)
    asm = np.append(b_SM,l_SM)
    sfh = np.log10(asm/(m_bc03*delta_T))
    plt.plot(T_age,sfh,linewidth=lw)
  plt.plot(T_age,Simulated_Log_SFH,c='k',linewidth=lw+2)
  plt.axvline(np.log10(Beg_Age[t1p][-1]),c='k',linewidth=lw-2)
  plt.axvline(np.log10(Beg_Age[t2p][-1]),c='k',linewidth=lw-2)
  plt.axvline(np.log10(Beg_Age[t3p][-1]),c='k',linewidth=lw-2)
  plt.xlabel('Log($t_{age}$)',fontsize=ft_12,labelpad=pd, fontweight='bold')
  plt.ylabel('Log(SFR[$\chi^2_{min}$,$\chi^2_{min}$+'+str(chi_limit[0])+']$ (M_{\odot}/Gyr$)) ',fontsize=ft_12,labelpad=pd, fontweight='bold')
  plt.minorticks_on()
  plt.tick_params(which='major',length=ltw,width=lw,direction='in',labelsize=ft_1,pad=pd,axis='x',bottom=True,top=True,left=True,right=True)
  plt.tick_params(which='minor',length=ltwm,width=lw,direction='in',labelsize=ft_1,pad=pd,axis='x',bottom=True,top=True,left=True,right=True)
  plt.tick_params(which='major',length=ltw,width=lw,direction='in',labelsize=ft_1,pad=pd,axis='y',bottom=True,top=True,left=True,right=True)
  plt.tick_params(which='minor',length=ltwm,width=lw,direction='in',labelsize=ft_1,pad=pd,axis='y',bottom=True,top=True,left=True,right=True)
  for axis in ['top','bottom','left','right']:
    plt.gca().spines[axis].set_linewidth(lw)


  plt.subplot(3,3,6)
  stat=0
  for stat in range (0,3):
     plt.plot(diff_colors[:,stat],label=statistics[stat],c=stat_c[stat],linewidth=lw)
  num=0
  for num in range (0,len(num_ticks_color)):
      plt.axvline(x=num_ticks_color[num],linewidth=lw-4,c='grey')
  plt.axhline(0,linewidth=lw-4,c='grey')
  plt.xticks(num_ticks_color,Indices_color,rotation=20,fontsize=ft_12-80)
#  plt.xticklabels(Indices_color,rotation=20,fontsize=ft_12-80)
  plt.ylabel('$\\Delta$Color mags',fontsize=ft_12,labelpad=pd, fontweight='bold')
  plt.minorticks_on()
  plt.tick_params(which='major',length=ltw,width=lw,direction='in',labelsize=ft_1,pad=pd,axis='x',bottom=True,top=True,left=True,right=True)
  plt.tick_params(which='minor',length=ltwm,width=lw,direction='in',labelsize=ft_1,pad=pd,axis='x',bottom=True,top=True,left=True,right=True)
  plt.tick_params(which='major',length=ltw,width=lw,direction='in',labelsize=ft_1,pad=pd,axis='y',bottom=True,top=True,left=True,right=True)
  plt.tick_params(which='minor',length=ltwm,width=lw,direction='in',labelsize=ft_1,pad=pd,axis='y',bottom=True,top=True,left=True,right=True)
  for axis in ['top','bottom','left','right']:
    plt.gca().spines[axis].set_linewidth(lw)
    
  #plt.subplot(3,3,7)
  #plt.scatter(s1_all[good_samples],s2_all[good_samples],c=nflat_samples[good_samples,-1],s=5000,marker='.')
  #t1y = plt.gca().yaxis.get_offset_text()
  #t1y.set_size(ft_12)
  #t1x = plt.gca().xaxis.get_offset_text()
  #t1x.set_size(ft_12)
  #plt.xlabel('Log(s1)',fontsize=ft_12,labelpad=pd, fontweight='bold')
  #plt.ylabel('Log(s2)',fontsize=ft_12,labelpad=pd, fontweight='bold')
  #plt.minorticks_on()
  #plt.tick_params(which='major',length=ltw,width=lw,direction='in',labelsize=ft_1,pad=pd,axis='x',bottom=True,top=True,left=True,right=True)
  #plt.tick_params(which='minor',length=ltwm,width=lw,direction='in',labelsize=ft_1,pad=pd,axis='x',bottom=True,top=True,left=True,right=True)
  #plt.tick_params(which='major',length=ltw,width=lw,direction='in',labelsize=ft_1,pad=pd,axis='y',bottom=True,top=True,left=True,right=True)
  #plt.tick_params(which='minor',length=ltwm,width=lw,direction='in',labelsize=ft_1,pad=pd,axis='y',bottom=True,top=True,left=True,right=True)
  #for axis in ['top','bottom','left','right']:
  #  plt.gca().spines[axis].set_linewidth(lw)



  idx_r2 = np.where(s3_all[good_samples]>7)

  plt.subplot(3,3,8)
  plt.scatter(s3_all[good_samples],s4_all[good_samples],c=nflat_samples[good_samples,-1],s=5000,marker='.')
  cbar = plt.colorbar()
  cbar.ax.tick_params(labelsize=ft_12)
  cbar.set_label('$[Fe/H]$',fontsize=ft_12)
  plt.xlim([7,None])
  #plt.ylim([None,np.max(s4_all[good_samples])])
  t2y = plt.gca().yaxis.get_offset_text()
  t2y.set_size(ft_12)
  t2x = plt.gca().xaxis.get_offset_text()
  t2x.set_size(ft_12)
  plt.xlabel('Log(s3)',fontsize=ft_12,labelpad=pd, fontweight='bold')
  plt.ylabel('Log(s4)',fontsize=ft_12,labelpad=pd, fontweight='bold')
  plt.minorticks_on()
  plt.tick_params(which='major',length=ltw,width=lw,direction='in',labelsize=ft_1,pad=pd,axis='x',bottom=True,top=True,left=True,right=True)
  plt.tick_params(which='minor',length=ltwm,width=lw,direction='in',labelsize=ft_1,pad=pd,axis='x',bottom=True,top=True,left=True,right=True)
  plt.tick_params(which='major',length=ltw,width=lw,direction='in',labelsize=ft_1,pad=pd,axis='y',bottom=True,top=True,left=True,right=True)
  plt.tick_params(which='minor',length=ltwm,width=lw,direction='in',labelsize=ft_1,pad=pd,axis='y',bottom=True,top=True,left=True,right=True)
  for axis in ['top','bottom','left','right']:
    plt.gca().spines[axis].set_linewidth(lw)


  plt.subplot(3,3,9)
  plt.scatter(s3_all[good_samples],s4_all[good_samples],c=R2_all[good_samples],s=5000,vmin=R2_all[good_samples][idx_r2].min(),vmax=R2_all[good_samples][idx_r2].max(),marker='.')
  cbar = plt.colorbar()
  cbar.ax.tick_params(labelsize=ft_12)
  cbar.set_label('$R2$',fontsize=ft_12)
  plt.xlim([7,None])
  #plt.ylim([None,np.max(s4_all[good_samples])])
  t2y = plt.gca().yaxis.get_offset_text()
  t2y.set_size(ft_12)
  t2x = plt.gca().xaxis.get_offset_text()
  t2x.set_size(ft_12)
  plt.xlabel('Log(s3)',fontsize=ft_12,labelpad=pd, fontweight='bold')
  #plt.ylabel('Log(s4)',fontsize=ft_12,labelpad=pd, fontweight='bold')
  plt.minorticks_on()
  plt.tick_params(which='major',length=ltw,width=lw,direction='in',labelsize=ft_1,pad=pd,axis='x',bottom=True,top=True,left=True,right=True)
  plt.tick_params(which='minor',length=ltwm,width=lw,direction='in',labelsize=ft_1,pad=pd,axis='x',bottom=True,top=True,left=True,right=True)
  plt.tick_params(which='major',length=ltw,width=lw,direction='in',labelsize=ft_1,pad=pd,axis='y',bottom=True,top=True,left=True,right=True)
  plt.tick_params(which='minor',length=ltwm,width=lw,direction='in',labelsize=ft_1,pad=pd,axis='y',bottom=True,top=True,left=True,right=True)
  for axis in ['top','bottom','left','right']:
    plt.gca().spines[axis].set_linewidth(lw)
  
  plt.subplot(3,3,7)
  for stat in range (0,3):
     plt.plot(T_age,np.log10(Mass_present[stat,:]),linewidth=lw,c=stat_c[stat])
  plt.plot(T_age,Initial_SFH[:-3],linewidth=lw,c='k')
  plt.xlabel('Log($t_{age}$)',fontsize=ft_12,labelpad=pd, fontweight='bold')
  plt.ylabel('Log($M_{present}$$M_{\odot}$)',fontsize=ft_12,labelpad=pd, fontweight='bold')
  plt.minorticks_on()
  plt.tick_params(which='major',length=ltw,width=lw,direction='in',labelsize=ft_1,pad=pd,axis='x',bottom=True,top=True,left=True,right=True)
  plt.tick_params(which='minor',length=ltwm,width=lw,direction='in',labelsize=ft_1,pad=pd,axis='x',bottom=True,top=True,left=True,right=True)
  plt.tick_params(which='major',length=ltw,width=lw,direction='in',labelsize=ft_1,pad=pd,axis='y',bottom=True,top=True,left=True,right=True)
  plt.tick_params(which='minor',length=ltwm,width=lw,direction='in',labelsize=ft_1,pad=pd,axis='y',bottom=True,top=True,left=True,right=True)
  for axis in ['top','bottom','left','right']:
    plt.gca().spines[axis].set_linewidth(lw)

 
  fig.set_size_inches(140,80)
  fig.suptitle('min $\chi^2$='+str(np.round(nchisq.min(),3))+"|Steps="+str(chain_len)+"|max Tau="+str(np.round(np.max(tau),2)),fontsize=ft_22,fontweight='bold')
  fig.subplots_adjust(left=0.06, bottom=0.04, right=0.98, top=0.95,wspace=0.20, hspace=0.20)
  fig.savefig(path_plots+'sfr_obs/'+str(e_fold_tau)+'sigma_'+str(fraction)+'.png')
  plt.close()
 
 
  ALL_PAR = np.c_[R1_all,R2_all,R3_all,TF_all]
  ALL_S = np.c_[s1_all,s2_all,s3_all,s4_all]
  f1_SFH = nflat_SFH[:,:-3]
  f2_SFH = np.c_[ALL_PAR,nflat_SFH[:,-3:]]
  Median_PAR = np.c_[R1_median,R2_median,R3_median,TF_median]
  MLE_PAR = np.c_[R1_mle,R2_mle,R3_mle,TF_mle]
  Median_S =  np.c_[s1_median,s2_median,s3_median,s4_median]
  MLE_S = np.c_[s1_mle,s2_mle,s3_mle,s4_mle]
  Mode_PAR = np.c_[R1_mode,R2_mode,R3_mode,TF_mode]

  mle_f1 = np.append(new_sfr[0,:-3],np.log10(nsa_mass - np.sum(10**(new_sfr[0,:-3]))))
  mle_f2 = np.concatenate((MLE_PAR[0],new_sfr[0,[7,8,9]]))
  median_f1 = np.append(new_sfr[1,:-3],np.log10(nsa_mass - np.sum(10**(new_sfr[1,:-3]))))
  median_f2 = np.concatenate((Median_PAR[0],new_sfr[1,[7,8,9]]))
  mode_f1 = np.append(new_sfr[2,:-3],np.log10(nsa_mass - np.sum(10**(new_sfr[2,:-3]))))
  mode_f2 = np.concatenate((Mode_PAR[0],new_sfr[2,[7,8,9]]))
  data_f2 = np.array([R1_fsim,R2_fsim,R3_fsim,TF_fsim,0.5,0.05,0.0])
  #Corner plots of all parameters
  labels_f1 = ['Log($M_{0.01}$)','Log($M_{0.04}$)','Log($M_{0.12}$)','Log($M_{0.40}$)','Log($M_{1.31}$)','Log($M_{4.28}$)','Log($M_{7.74}$)','Log($M_{14}$)']
  labels_f1_lin = ['$M_{0.01}$','$M_{0.04}$','$M_{0.12}$','$M_{0.40}$','$M_{1.31}$','$M_{4.28}$','$M_{7.74}$','$M_{14}$']
  labels_f2 = ['R1','R2','R3','T$_{f}$ Gyr','$E_{y}$','$E_{o}$','[Fe/H]']
  fig = corner.corner(f1_SFH,plot_datapoints=False,bins=bins,labels=labels_f1,label_kwargs=dict(fontsize=24),truth_color='r',plot_density=True)
  ndim = len(labels_f1)
  axes = np.array(fig.axes).reshape((ndim, ndim))
 # Loop over the diagonal
  p=0
  for p in range(ndim):
        ax = axes[p, p]
        ax.axvline(mle_f1[p], color="r")
        ax.axvline(median_f1[p], color="g")
        ax.axvline(mode_f1[p], color="b")
  for ax in fig.get_axes():
        ax.tick_params(axis='both', labelsize=15)
        
  fig.savefig(path_plots+'corner/'+str(e_fold_tau)+'sigma_'+str(fraction)+'_c1.png')
  plt.close()


  fig = corner.corner(f2_SFH,plot_datapoints=False,bins=bins,labels=labels_f2,label_kwargs=dict(fontsize=24),truth_color='r',plot_density=True)
  ndim = len(labels_f2)
  axes = np.array(fig.axes).reshape((ndim, ndim))
 # Loop over the diagonal
  p=0
  for p in range(ndim):
        ax = axes[p, p]
        ax.axvline(mle_f2[p], color="r")
        ax.axvline(median_f2[p], color="g")
        ax.axvline(data_f2[p],color="k")
        ax.axvline(mode_f2[p], color="b")
  for ax in fig.get_axes():
        ax.tick_params(axis='both', labelsize=15)
  fig.savefig(path_plots+'corner/'+str(e_fold_tau)+'sigma_'+str(fraction)+'_c2.png')
  plt.close()


  fig = corner.corner(10**(f1_SFH),plot_datapoints=False,bins=bins,labels=labels_f1_lin,label_kwargs=dict(fontsize=24),truth_color='r',plot_density=True)
  ndim = len(labels_f1)
  axes = np.array(fig.axes).reshape((ndim, ndim))
 # Loop over the diagonal
  p=0
  for p in range(ndim):
        ax = axes[p, p]
        ax.axvline(10**(mle_f1[p]), color="r")
        ax.axvline(10**(median_f1[p]), color="g")
        ax.axvline(10**(mode_f1[p]), color="b")
  for ax in fig.get_axes():
        ax.tick_params(axis='both', labelsize=15)
        
  fig.savefig(path_plots+'corner/'+str(e_fold_tau)+'sigma_'+str(fraction)+'_c3.png')
  plt.close()
