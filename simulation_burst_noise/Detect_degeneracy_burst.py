
import numpy as np
from matplotlib import pyplot as plt
from astropy.io import fits
from scipy import stats
import math
import os
#os.chdir('/Users/nikhil/code/SDSS21') #Comment on cluster
from plot_functions_norm2 import make_mods
from simulation_model import make_models,get_Mbc03,get_dt
import site
import sys
site.addsitedir('/home/naj222') #Uncomment on cluster
import pandas as pd
import emcee
import corner
from itertools import combinations
import random
import extinction
from random import randint

path_input = '/mnt/gpfs2_4m/scratch/naj222/dir1/input/'                #Uncomment on cluster
path_output = '/mnt/gpfs2_4m/scratch/naj222/dir1/output/simulation_burst_noise/'#Uncomment on cluster
path_plots = '/mnt/gpfs2_4m/scratch/naj222/dir1/plots/simulation_burst_noise/'      #Uncomment on cluster

#path_input='/Users/nikhil/Data/emcee_test/Input/'    #Comment on cluster
#path_output='/Users/nikhil/Data/emcee_test/Output/'  #Comment on cluster
#path_plots='/Users/nikhil/Documents4/SFH_Paper_plots/'          #Comment on cluster
#specify path to text file that contains MaNGA IDs of galaxies with AGN cut and axis ratio cut



t_index = np.array([5,9,13,17,21,25,27,29]) #New Scheme
COLOR_4SSP = fits.open(path_input+'ssp/ssp_bc03_4.color.fits')
DATA_4SSP = COLOR_4SSP[0].data[:,t_index,:]

hdu = fits.open(path_input+'Logcube/manga-7495-12703-LOGCUBE.fits')
Waveh = hdu['WAVE'].data

with open(path_input+'/sample_txt/TOT.txt') as f:
    Line0 = [line.rstrip('\n') for line in open(path_input+'/sample_txt/TOT.txt')]
IZ =np.zeros(np.shape(Line0))
q=0
for q in range (0,np.shape(Line0)[0]):
  ID = Line0[q]
  drpall = fits.open(path_input+'drpall/drpall-v2_3_1.fits')
  tbdata = drpall[1].data
  indx = np.where(tbdata['mangaid'] == Line0[q])
  IZ[q] = tbdata['nsa_z'][indx][0]
  
L_solar = 3.839*math.pow(10,33)
fAB = 3.631*math.pow(10,-20)
c = 2.99792*math.pow(10,10)
#Read photometric filters
mFUV = pd.read_csv(path_input+'filters/FUV_Prime.txt', comment='#', header=None, delim_whitespace=True)
mNUV = pd.read_csv(path_input+'filters/NUV_Prime.txt', comment='#', header=None, delim_whitespace=True)
mW1  = pd.read_csv(path_input+'filters/W1_Prime.txt', comment='#', header=None, delim_whitespace=True)
mW2  = pd.read_csv(path_input+'filters/W2_Prime.txt', comment='#', header=None, delim_whitespace=True)
mM2  = pd.read_csv(path_input+'filters/M2_Prime.txt', comment='#', header=None, delim_whitespace=True)
mU  = pd.read_csv(path_input+'filters/U_Prime.txt', comment='#', header=None, delim_whitespace=True)
mG  = pd.read_csv(path_input+'filters/G_Prime.txt', comment='#', header=None, delim_whitespace=True)
mR  = pd.read_csv(path_input+'filters/R_Prime.txt', comment='#', header=None, delim_whitespace=True)
mI  = pd.read_csv(path_input+'filters/I_Prime.txt', comment='#', header=None, delim_whitespace=True)
mZ  = pd.read_csv(path_input+'filters/Z_Prime.txt', comment='#', header=None, delim_whitespace=True)
#Read Data of photometric filters
RFUV = np.array(mFUV)
RNUV = np.array(mNUV)
RW1 = np.array(mW1)
RW2 = np.array(mW2)
RM2 = np.array(mM2)
RU = np.array(mU)
RG = np.array(mG)
RR = np.array(mR)
RI = np.array(mI)
RZ = np.array(mZ)
#Read Fluxes
RfFUV = RFUV[:,1]
RfNUV = RNUV[:,1]
RfW1 = RW1[:,1]
RfW2 = RW2[:,1]
RfM2 = RM2[:,1]
RfU = RU[:,1]
RfG = RG[:,1]
RfR = RR[:,1]
RfI = RI[:,1]
RfZ = RZ[:,1]
#Read Wavelengths
RwFUV =RFUV[:,0]
RwNUV = RNUV[:,0]
RwW1 = RW1[:,0]
RwW2 = RW2[:,0]
RwM2 = RM2[:,0]
RwU = RU[:,0]
RwG = RG[:,0]
RwR = RR[:,0]
RwI = RI[:,0]
RwZ = RZ[:,0]

#Read Lick indices
mf = pd.read_csv(path_input+'filters/spec1.txt', comment='#', header=None, delim_whitespace=True)
Indices = np.array(mf[2])
H_Blue = np.array(mf[9])
L_Blue = np.array(mf[8])
H_Red = np.array(mf[13])
L_Red = np.array(mf[12])
H_Index = np.array(mf[5])
L_Index = np.array(mf[4])
Unit_indx = np.array(mf[16])
KJ = np.where(Unit_indx == 'mag')
KL = Unit_indx*0
KL[KJ] = 1
Rmid = (H_Red + L_Red)/2
Bmid = (H_Blue + L_Blue)/2
Imid = (H_Index + L_Index)/2
Delt_In = H_Index - L_Index
Delt_R = H_Red - L_Red
Delt_B = H_Blue - L_Blue
#Initiate Arrays for lick
Ages_array = np.array([0.0030,0.0035,0.0054,0.0063,0.0099,0.0114,0.0181,0.0206,0.0331,0.0374,0.0605,0.0676,0.1103,0.1224,0.2012,0.2213,0.3670,0.4003,0.6692,0.7238,1.2205,1.3090,2.2258,2.3673,4.0592,4.2810,7.4026,7.7417,13.500,14.000])
IK_prime = Ages_array
EBV = np.linspace(0,1.5,30)
W = np.array([-2.2490,-1.6464,-0.6392,-0.3300,0.0932,0.5595])
red_shift = np.sort(IZ)
nsa_mass = 12137400000.0
Simulation_Z = 0.0
e_folding_time = np.array([0.1,0.2,0.3,0.9,2.7])
fraction_array = [0,1,2,3,4,5,6,7,8,9,10]
f=0
for f in range (0,len(fraction_array)):
 fraction = fraction_array[f]
 e=0
 for e in range (0,len(e_folding_time)):
  e_fold_tau = 1
  qe_fold_tau = e_folding_time[e]
  t_age = Ages_array[t_index]


  #Calculate delta_T (time widths)
  Ages = t_age
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
  time1=comb_all[4,0]
  time2=comb_all[4,1]
  time3=comb_all[4,2]
  #for discreet bins
  t1p = np.where(Beg_Age<time1)
  t2p = np.where((Beg_Age>time1) & (Beg_Age<time2))
  t3p = np.where((Beg_Age>time2) & (Beg_Age<time3))
  t4p = np.where(Beg_Age>time3)
  
  #new
  dt = get_dt(t_age)
  bc03_M = get_Mbc03(Simulation_Z,DATA_4SSP)
  quench_t = np.where(t_age<=1.5)[0]
  #fine SFH
  t_fine = np.linspace(t_age[0],t_age[-1],100)
  quench_t_fine = np.where(t_fine<=t_age[quench_t][-1])[0]
  dt_fine = get_dt(t_fine)
  bc03_M_fine = np.interp(t_fine,t_age,bc03_M)
  old_sfr_fine = np.exp(-(14-t_fine)/e_fold_tau)
  burst_fine = np.exp(-(14-t_fine)/qe_fold_tau)
  burst_sfr_fine = np.zeros(np.shape(burst_fine))
  burst_sfr_fine[quench_t_fine] = 0.5*burst_fine[-len(quench_t_fine):]
  Simulated_SFH_fine = burst_sfr_fine + old_sfr_fine
  Simulated_mass_prime_fine = Simulated_SFH_fine*(dt_fine*bc03_M_fine)
  Simulated_mass_fine = (Simulated_mass_prime_fine/np.sum(Simulated_mass_prime_fine))*nsa_mass
  Simulated_total_mass_fine = Simulated_mass_fine/bc03_M_fine
  Simulated_Log_SFH_fine = np.log10(Simulated_mass_fine/(bc03_M_fine*dt_fine))


  burst_interp = np.interp(t_age[quench_t],t_fine[quench_t_fine],burst_sfr_fine[quench_t_fine])
  old_sfr = np.exp(-(14-t_age)/e_fold_tau)
  new_sfr = np.zeros(np.shape(old_sfr))
  new_sfr[quench_t] = burst_interp
  new_SFH = new_sfr + old_sfr
  Simulated_SFH = new_SFH
  Simulated_mass_prime = Simulated_SFH*(dt*bc03_M)
  Simulated_mass = (Simulated_mass_prime/np.sum(Simulated_mass_prime))*nsa_mass
  Log_Mass = np.log10(Simulated_mass)
  Log_Mass = np.log10(Simulated_mass)
  alpha_bestfit1 = 10**(Log_Mass)/bc03_M
  Simulated_total_mass = Simulated_mass/bc03_M
  
  #calculate R1,R2,R3 for simulations
  R1_sim = np.log10((np.sum(Simulated_total_mass[t1p])/np.sum(dt[t1p]))/(np.sum(Simulated_total_mass[t2p])/np.sum(dt[t2p])))
  R2_sim = np.log10((np.sum(Simulated_total_mass[t3p])/np.sum(dt[t3p]))/(np.sum(Simulated_total_mass[t4p])/np.sum(dt[t4p])))
  R3_sim = np.log10((np.sum(Simulated_total_mass[np.c_[t1p,t2p][0]])/np.sum(dt[np.c_[t1p,t2p][0]]))/(np.sum(Simulated_total_mass[np.c_[t3p,t4p][0]])/np.sum(dt[np.c_[t3p,t4p][0]])))
  cumulative_sm = np.cumsum(Simulated_mass)
  TF_sim = np.interp(0.5,cumulative_sm/nsa_mass,t_age)
  
  reader = emcee.backends.HDFBackend(path_output+'_'+str(qe_fold_tau)+'_sigma_'+str(fraction))
  tau = reader.get_autocorr_time(tol=0)
  chain_len = len(reader.get_chain())
  nburnin= int(2*tau.max())
  thin = int(0.5*tau.min())
  #get flat samples of parameters and ln_probability
  nflat_samples = reader.get_chain(discard=nburnin,thin=thin,flat=True) #Chain thinning
  nflat_lnprob = reader.get_log_prob(discard=nburnin,thin=thin,flat=True)
  nchisq = (nflat_lnprob/(-0.5))/39
  chi_limit = 0.02
  good_samples = np.where((nchisq>=nchisq.min())&(nchisq<nchisq.min()+chi_limit))[0]
  mle_samples = np.where(nchisq==nchisq.min())
  if len(good_samples)>20:
      sample_len = 20
      idx = np.argsort(nchisq[good_samples])[:20]
  else:
      sample_len = len(good_samples)
      idx = good_samples
  idx[-1] = mle_samples[0][0]
  CSP = np.zeros((6900,sample_len+1))
  SFR = np.zeros((6900,sample_len+1))
  SFR_bestfit = np.zeros((6900,sample_len+1))
  Residue = np.zeros((6900,sample_len+1))
  alpha_bf = np.zeros((sample_len+1,len(alpha_bestfit1)))
  Z_sample = np.zeros(sample_len+1)
  Ey_sample = np.zeros(sample_len+1)
  Eo_sample = np.zeros(sample_len+1)
  R1_sample = np.zeros(sample_len+1)
  R2_sample = np.zeros(sample_len+1)
  R3_sample = np.zeros(sample_len+1)
  CM_MASS = np.zeros((sample_len+1,len(alpha_bestfit1)))
  SM_MASS = np.zeros((sample_len+1,len(alpha_bestfit1)))
  TF_sample =np.zeros(sample_len+1)
  sample=0
  for sample in range (0,sample_len):
      new_sfr = nflat_samples[idx[sample],:]
      Z_sample[sample] = new_sfr[-1]
      Ey_sample[sample] = new_sfr[-3]
      Eo_sample[sample] = new_sfr[-2]
      bc03_M_sample = get_Mbc03(new_sfr[-1],DATA_4SSP)
      remaining_mass = nsa_mass - np.sum(10**new_sfr[:-3])
      SM = np.append(10**(new_sfr[:-3]),remaining_mass)
      SM_MASS[sample,:] = SM
      SMT = SM/bc03_M_sample
      alpha_bf[sample,:] = SM/bc03_M_sample
      R1_sample[sample] = np.log10((np.sum(SMT[t1p])/np.sum(delta_T[t1p]))/(np.sum(SMT[t2p])/np.sum(delta_T[t2p])))
      R2_sample[sample] = np.log10((np.sum(SMT[t3p])/np.sum(delta_T[t3p]))/(np.sum(SMT[t4p])/np.sum(delta_T[t4p])))
      R3_sample[sample] = np.log10((np.sum(SMT[np.c_[t1p,t2p][0]])/np.sum(delta_T[np.c_[t1p,t2p][0]]))/(np.sum(SMT[np.c_[t3p,t4p][0]])/np.sum(delta_T[np.c_[t3p,t4p][0]])))
      cumulative_mass = np.cumsum(SM)
      CM_MASS[sample,:] = cumulative_mass
      TF_sample[sample] = np.interp(0.5,cumulative_mass/nsa_mass,t_age)
  alpha_bf[-1,:] = alpha_bestfit1
  Z_sample[-1] = 0.0
  Ey_sample[-1] = 0.5
  Eo_sample[-1] = 0.05
  for metal_mu in range (0,len(Z_sample)):
    required_Z = Z_sample[metal_mu]
    m1 = np.where(required_Z>W)[0][-1]
    m2 = np.where(required_Z<W)[0][0]
    delta_metal = W[m2] - W[m1]
    Ext_y = Ey_sample[metal_mu]
    Ext_o = Eo_sample[metal_mu]
    Av_y = 3.1 * Ext_y
    Av_o = 3.1 * Ext_o
    Frame1 = pd.read_csv(path_input+'templates/bc2003_hr_'+str(W[m1])+'.44', comment='#', header=None, delim_whitespace=True)
    Frame2 = pd.read_csv(path_input+'templates/bc2003_hr_'+str(W[m2])+'.44', comment='#', header=None, delim_whitespace=True)
    Frame1A = np.array(Frame1)
    Frame2A = np.array(Frame2)
    ll = Frame1A[:,0]
    Dat_1 = Frame1A[:,1:]
    Dat_2 = Frame2A[:,1:]
    Data_original = np.zeros(np.shape(Dat_1))
    for time in range (0,np.shape(Dat_1)[1]):
        for lmd in range (0,len(Dat_1)):
            Data_original[lmd,time] = np.interp(required_Z,[W[m1],W[m2]],[Dat_1[lmd,time],Dat_2[lmd,time]])
    Data = Data_original[:,t_index]
    ll_shift = ll*(1+red_shift[60])
    Alambda_red_y = extinction.calzetti00(ll, Av_y, 3.1)
    Alambda_red_o = extinction.calzetti00(ll, Av_o, 3.1)
    Data_fl = np.zeros(np.shape(Data))
    Data_O = np.zeros(np.shape(Data))
    w=0
    for w in range (0,1):
        Data_O[:,w] = extinction.apply(Alambda_red_y,Data[:,w])

    w=0
    for w in range (1,np.shape(Data)[1]):
        Data_O[:,w] = extinction.apply(Alambda_red_o,Data[:,w])

    #Color computation in observed frame
    lmd_shift =ll_shift
    Shifted_Data = Data_O/(1+red_shift[60])
    CSP[:,metal_mu] = np.dot(Shifted_Data,alpha_bf[metal_mu,:])
    SFR[:,metal_mu] = np.log10(CSP[:,metal_mu]/np.sum(CSP[:,metal_mu]))
    
  ft_12 =180
  ft_22 =180
  ft_1 = 180
  lw=15
  ltw = 60
  ltwm = 35
  pad_space=15
  
  number_of_colors = sample_len

  c_sample = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])for i in range(number_of_colors)]
  

  SFR_bf = SFR[:,-1]
  SFR_bestfit = np.repeat(SFR_bf[np.newaxis,],sample_len+1,axis=0)
  Residue = -2.5*(SFR_bestfit.T - SFR)
  l_idx = np.where((ll>1350)&(ll<3750))[0]
  csp_min= SFR[l_idx,:].min()
  csp_max= SFR[l_idx,:].max()
  res_min = Residue[l_idx,:].min()
  res_max = Residue[l_idx,:].max()
  

  fig, (ax1,ax2) = plt.subplots(nrows=2, sharex=True)
  for bf_sample in range (0,sample_len-1):
      ax1.plot(ll,SFR[:,bf_sample],c=c_sample[bf_sample],linewidth=lw)
  ax1.plot(ll,SFR[:,-1],c='k',label='Input SFH',linewidth=lw)
  ax1.axvline(1528,linewidth=lw,c='violet',label='Galex FUV $\lambda_{e}$')
  ax1.axvline(1928,linewidth=lw,c='blue',label='Swift uvw2 $\lambda_{c}$')
  ax1.axvline(2246,linewidth=lw,c='green',label='Swift uvm2 $\lambda_{c}$')
  ax1.axvline(2271,linewidth=lw,c='yellow',label='Galex NUV $\lambda_{e}$')
  ax1.axvline(2600,linewidth=lw,c='red',label='Swift uvw1 $\lambda_{c}$')
  ax1.set_ylabel('Log[f$_{\\lambda}$($L_{\odot}$/$\AA^{-1}$)]',fontsize=ft_12+40,labelpad=pad_space+5)
  ax1.set_xlim([1350,3750])
  ax1.set_ylim([csp_min,csp_max])
  ax1.legend(loc='lower right',fontsize=ft_22-20,frameon=False)
  ax1.minorticks_on()
  ax1.tick_params(which='major',length=ltw,width=lw,direction='in',labelsize=ft_1,pad=pad_space,axis='x',bottom=True,top=True,left=True,right=True)
  ax1.tick_params(which='minor',length=ltwm,width=lw,direction='in',labelsize=ft_1,pad=pad_space,axis='x',bottom=True,top=True,left=True,right=True)
  ax1.tick_params(which='major',length=ltw,width=lw,direction='in',labelsize=ft_1,pad=pad_space,axis='y',bottom=True,top=True,left=True,right=True)
  ax1.tick_params(which='minor',length=ltwm,width=lw,direction='in',labelsize=ft_1,pad=pad_space,axis='y',bottom=True,top=True,left=True,right=True)
  for axis in ['top','bottom','left','right']:
    ax1.spines[axis].set_linewidth(lw)
    
  
  for s in range (0,sample_len-1):
      ax2.plot(ll,Residue[:,s],c=c_sample[s],linewidth=lw-2)
  ax2.set_xlabel('$\\lambda$($\AA$)',fontsize=ft_12+40,labelpad=pad_space+5)
  ax2.set_ylabel('Residuals (mags)',fontsize=ft_12+40,labelpad=pad_space+5)
  ax2.axhline(y=0,linewidth=lw,c='grey')
  ax2.set_ylim([res_min,res_max])
  ax2.set_xlim([1350,3750])
  ax2.minorticks_on()
  ax2.tick_params(which='major',length=ltw,width=lw,direction='in',labelsize=ft_1,pad=pad_space,axis='x',bottom=True,top=True,left=True,right=True)
  ax2.tick_params(which='minor',length=ltwm,width=lw,direction='in',labelsize=ft_1,pad=pad_space,axis='x',bottom=True,top=True,left=True,right=True)
  ax2.tick_params(which='major',length=ltw,width=lw,direction='in',labelsize=ft_1,pad=pad_space,axis='y',bottom=True,top=True,left=True,right=True)
  ax2.tick_params(which='minor',length=ltwm,width=lw,direction='in',labelsize=ft_1,pad=pad_space,axis='y',bottom=True,top=True,left=True,right=True)
  for axis in ['top','bottom','left','right']:
    ax2.spines[axis].set_linewidth(lw)

  
  fig.set_size_inches(180,100)
  fig.subplots_adjust(left=0.10, bottom=0.10, right=0.95, top=0.95, hspace=0.0)
  fig.savefig(path_plots+'spectra/'+str(qe_fold_tau)+'_'+str(fraction)+'spectra.png')
  plt.close()


  
  
  fig = plt.figure()
  ax1 = fig.add_subplot(221)
  for s in range (0,sample_len-1):
      ax1.plot(np.log10(t_age),np.log10(alpha_bf[s,:]/delta_T),c=c_sample[s],linewidth=lw-5)
  ax1.plot(np.log10(t_age),np.log10(alpha_bestfit1/delta_T),c='k',label='Input SFH',linewidth=lw+4)
  ax1.axvline(np.log10(Beg_Age[t1p][-1]),c='k',linewidth=lw-2)
  ax1.axvline(np.log10(Beg_Age[t2p][-1]),c='k',linewidth=lw-2)
  ax1.axvline(np.log10(Beg_Age[t3p][-1]),c='k',linewidth=lw-2)
  ax1.legend(loc='lower left',fontsize=ft_22-20,frameon=False)
  ax1.set_xlabel('Log($t_{look back}$ Gyr)',fontsize=ft_22)
  ax1.set_ylabel('Log(SFR $M_{\odot}$/Gyr)',fontsize=ft_22)
  ax1.minorticks_on()
  ax1.tick_params(which='major',length=ltw,width=lw,direction='in',labelsize=ft_1,pad=pad_space,axis='x',bottom=True,top=True,left=True,right=True)
  ax1.tick_params(which='minor',length=ltwm,width=lw,direction='in',labelsize=ft_1,pad=pad_space,axis='x',bottom=True,top=True,left=True,right=True)
  ax1.tick_params(which='major',length=ltw,width=lw,direction='in',labelsize=ft_1,pad=pad_space,axis='y',bottom=True,top=True,left=True,right=True)
  ax1.tick_params(which='minor',length=ltwm,width=lw,direction='in',labelsize=ft_1,pad=pad_space,axis='y',bottom=True,top=True,left=True,right=True)
  for axis in ['top','bottom','left','right']:
      ax1.spines[axis].set_linewidth(lw)

  ax2 = fig.add_subplot(222)
  ax2.hist(R1_sample[:-1],bins='auto',edgecolor='r',facecolor='None',linewidth=lw-2)
  ax2.axvline(R1_sim,linewidth=lw+4,c='k',label='Input R1')
  ax2.legend(loc='upper left',fontsize=ft_22-20,frameon=False)
  #ax2.set_xlim([-7,0])
  ax2.set_xlabel('R1',fontsize=ft_22)
  ax2.minorticks_on()
  ax2.tick_params(which='major',length=ltw,width=lw,direction='in',labelsize=ft_1,pad=pad_space,axis='x',bottom=True,top=True,left=True,right=True)
  ax2.tick_params(which='minor',length=ltwm,width=lw,direction='in',labelsize=ft_1,pad=pad_space,axis='x',bottom=True,top=True,left=True,right=True)
  ax2.tick_params(which='major',length=ltw,width=lw,direction='in',labelsize=ft_1,pad=pad_space,axis='y',bottom=True,top=True,left=True,right=True)
  ax2.tick_params(which='minor',length=ltwm,width=lw,direction='in',labelsize=ft_1,pad=pad_space,axis='y',bottom=True,top=True,left=True,right=True)
  for axis in ['top','bottom','left','right']:
      ax2.spines[axis].set_linewidth(lw)
  
  fig.set_size_inches(140,80)
  fig.subplots_adjust(left=0.10, bottom=0.06, right=0.98, top=0.95,wspace=0.20, hspace=0.35)
  fig.savefig(path_plots+'spectra/'+str(qe_fold_tau)+'_'+str(fraction)+'spectra_pars.png')
  plt.close()

