import numpy as np
from matplotlib import pyplot as plt
from astropy.io import fits
from scipy import stats
import pandas as pd
import math
import os
os.chdir('/Users/nikhil/code/mcc/code/norm2_all_lick_bin/') #Comment on cluster
from plot_functions_norm2 import make_models
import site
import sys
site.addsitedir('/home/naj222') #Uncomment on cluster
import emcee
import corner
from itertools import combinations

#path_input = '/mnt/gpfs3_amd/scratch/naj222/dir1/input/'                #Uncomment on cluster
#path_output = '/mnt/gpfs3_amd/scratch/naj222/dir1/output/norm2_all_lick_bin/'#Uncomment on cluster
#path_plots = '/mnt/gpfs3_amd/scratch/naj222/dir1/plots/norm2_all_lick_bin/'#Uncomment on cluster

path_input='/Users/nikhil/Data/emcee_test/Input/'    #Comment on cluster
path_output='/Users/nikhil/Data/emcee_test/Output/'  #Comment on cluster
path_plots='/Users/nikhil/Data/emcee_test/'          #Comment on cluster
#specify path to text file that contains MaNGA IDs of galaxies with AGN cut and axis ratio cut
with open(path_input+'sample_txt/TOT.txt') as f:
     Line0 = [line.rstrip('\n') for line in open(path_input+'sample_txt/TOT.txt')]
     
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
Z_all= np.array([-2.2490,-1.6464,-0.6392,-0.3300,0.0932,0.5595])
bin_array = [0,1,2]
bin_color = ['r','g','b']
bin_marker = ['*','o',',']
bin_label = ['center','mid','out']
i=21
for i in range (21,22):#np.shape(Line0)[0]):
  bin_index =1
  fig = plt.figure()
  for bin_index in range (1,2):#len(bin_array)):
    #get galaxy info from drpall
    drpall = fits.open(path_input+'drpall/drpall-v2_3_1.fits')
    tbdata = drpall[1].data
    indx = np.where(tbdata['mangaid'] == Line0[i])
    nsa_mass = tbdata['nsa_elpetro_mass'][indx][0]
    zeta = tbdata['nsa_z'][indx][0] #Redshift
    #Open the pre computed SSP integrated observables for the specific redshift.
    DATA = fits.open(path_input+'models/bc03_ssp_'+str(zeta)+'_.fits')
    delta_lambda_lick = DATA[11].data
    #Open the data file
    #hdu = fits.open("/Volumes/Nikhil/MPL-7_Files/SwiM_binned/SwiM_"+str(Line0[i])+".fits")
    hdu = fits.open(path_input+'swim_files/SwiM_'+str(Line0[i])+'.fits')
    N_bin=hdu[28].data[bin_index]
    b_array = np.array([1,1,1,1.27,3,3,3,1.27])
    Nanomags_covar = 1 + b_array*np.log10(N_bin)
    Lick_list = [3,4,8,13,14,21,22,23,24,25] #fe4383,fe5270,fe5335,HdA
    m1_list,m2_list = [0,0,4,4,3,3],[3,1,6,5,6,7]  #w2-u,w1-u,g-i,g-r,u-i,u-z
    mags_info = np.array(['W2','W1','M2','u','g','r','i','z'])
    #Read mags from data
    Nanomags_data = hdu[22].data[:,bin_index]
    Mags_data = 22.5-2.5*np.log10(Nanomags_data)
    Nanomags_error = np.sqrt(hdu[23].data[:,bin_index])*Nanomags_covar
    Mags_error = (Nanomags_error/Nanomags_data)
    #if Mags_error.max()<0.03:
    #    Max_error = 0.03
    #else:
    #    Max_error = Mags_error.max()
    #Mags_error = np.clip(Mags_error,0.03,Max_error)
    data_color = Mags_data[m1_list] - Mags_data[m2_list]
    #Read Lick indices and d4000 from data
    Dn4000_data = hdu[18].data[43][bin_index]
    Dn4000_error = hdu[19].data[43][bin_index]*(1+1.156*np.log10(N_bin))*2
    Lick_indices_data = hdu[18].data[0:42,:][:,bin_index]
    Lick_indices_error = 2*hdu[19].data[0:42][:,bin_index]*(1+1.156*np.log10(N_bin))*2
    Lick_mag = -2.5*np.log10((delta_lambda_lick[mag_index]-Lick_indices_data[mag_index])/delta_lambda_lick[mag_index])
    Lick_ang = Lick_indices_data[~mag_index]
    #Initialize array

    reader = emcee.backends.HDFBackend(path_output+str(Line0[i])+'_'+str(bin_array[bin_index]))
    tau = reader.get_autocorr_time(tol=0)
    chain_len = len(reader.get_chain())
    pars = ['Log($M_{0.01}$)','Log($M_{0.03}$)','Log($M_{0.12}$)','Log($M_{0.4}$)','Log($M_{1.2}$)','Log($M_{4}$)','Log($M_{7.7}$)','$E_{y}$','$E_{o}$','[Fe/H]']
    dof_mcmc = 40
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
    comb_all = np.array(list(combinations(Beg_Age[1:-2]+0.001,3)))
    time1=comb_all[4,0]
    time2=comb_all[4,1]
    time3=comb_all[4,2]
    t1p = np.where(Beg_Age<time1)
    t2p = np.where((Beg_Age>time1) & (Beg_Age<time2))
    t3p = np.where((Beg_Age>time2) & (Beg_Age<time3))
    t4p = np.where(Beg_Age>time3)

    #tau = reader.get_autocorr_time(quiet=True)
    if np.isnan(tau.max()):
       max_tau = 100
    else:
       max_tau = tau.max()
    nburnin= int(2*max_tau)
    thin = int(0.5*max_tau)
    #get flat samples of parameters and ln_probability
    nflat_samples = reader.get_chain(discard=nburnin,thin=thin,flat=True) #Chain thinning
    nflat_lnprob = reader.get_log_prob(discard=nburnin,thin=thin,flat=True)
    nchisq = (nflat_lnprob/(-0.5))/dof_mcmc
    Rem_mass = np.zeros(np.shape(nflat_lnprob))
    R1_all = np.zeros(np.shape(nflat_lnprob))
    R2_all = np.zeros(np.shape(nflat_lnprob))
    R3_all = np.zeros(np.shape(nflat_lnprob))
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
    nflat_SFH = np.c_[nflat_samples[:,:-3],Rem_mass,nflat_samples[:,-3:]]
    good_samples_1 = np.where((nchisq>=nchisq.min())&(nchisq<nchisq.min()+1))[0]
    good_samples_2 = np.where((nchisq>=nchisq.min())&(nchisq<nchisq.min()+3))[0]
    if len(good_samples_1) < 20:
        good_samples = good_samples_2
    else:
        good_samples = good_samples_1
    flat_samples = nflat_samples[good_samples,:]
    flat_lnprob=nflat_lnprob[good_samples]
    flat_chi =nchisq[good_samples]
    #Get best fit parameters
    new_sfr = np.zeros((3,np.shape(flat_samples)[1]))
    best_index = np.where(flat_chi==flat_chi.min())[0][0]
    new_sfr[0,:] = flat_samples[best_index,:] #mle
    new_sfr[1,:] = np.median(flat_samples,axis=0) #median
    for hist in range (0,np.shape(flat_samples)[1]):
        histogram,bin_histogram = np.histogram(nflat_samples[:,hist],bins=bins)
        new_sfr[2,hist] = bin_histogram[np.argmax(histogram)]
    statistics = ['MLE','Median','Mode']
    stat_c = ['r','g','b']
    sigma_logsfr = np.zeros(flat_samples.shape[1])
    upper_limits = np.zeros(flat_samples.shape[1])
    lower_limits = np.zeros(flat_samples.shape[1])
    par = 0
    for par in range (0,flat_samples.shape[1]):
     sigma_logsfr[par] = (np.sqrt(np.sum((np.median(flat_samples[:,par]) - flat_samples[:,par])**2)/len(flat_samples)))
     if np.max(flat_samples[:,par]) < (np.median(flat_samples[:,par]) + sigma_logsfr[par]):
        upper_limits[par] = 1
     if np.min(flat_samples[:,par]) > (np.median(flat_samples[:,par]) - sigma_logsfr[par]):
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
        ND4[stat],NL,NM = make_models(new_sfr[stat],DATA,t_index,nsa_mass,DATA_4SSP)
        NALICK_mag[:,stat] = (-2.5*np.log10((delta_lambda_lick[mag_index]-NL[mag_index])/delta_lambda_lick[mag_index]))*1.5
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
    R1_mle = R1_all[good_samples][best_index]
    R2_mle = R2_all[good_samples][best_index]
    R3_mle = R3_all[good_samples][best_index]
    R1_median = np.median(R1_all[good_samples])
    R2_median = np.median(R2_all[good_samples])
    R3_median = np.median(R3_all[good_samples])
    #define fontsize, padding etc for plots
    ft_12 =100
    ft_22 =100
    ft_1 = 100
    lw=10
    ltw = 49
    ltwm = 25
    pd=3
    
    
    ALL_R = np.c_[R1_all,R2_all,R3_all]
    f1_SFH = nflat_SFH[:,:-3]
    f2_SFH = np.c_[ALL_R,nflat_SFH[:,-3:]]
    Median_R = np.c_[R1_median,R2_median,R3_median]
    MLE_R = np.c_[R1_mle,R2_mle,R3_mle]

    mle_f1 = np.append(new_sfr[0,:-3],np.log10(nsa_mass - np.sum(10**(new_sfr[0,:-3]))))
    mle_f2 = np.concatenate((MLE_R[0],new_sfr[0,[7,8,9]]))
    median_f1 = np.append(new_sfr[1,:-3],np.log10(nsa_mass - np.sum(10**(new_sfr[1,:-3]))))
    median_f2 = np.concatenate((Median_R[0],new_sfr[1,[7,8,9]]))
    mode_f1 = np.append(new_sfr[2,:-3],np.log10(nsa_mass - np.sum(10**(new_sfr[2,:-3]))))
    mode_f2 = new_sfr[2,[7,8,9]]
    #Corner plots of all parameters
    labels_f1 = ['Log($M_{0.01}$)','Log($M_{0.03}$)','Log($M_{0.06}$)','Log($M_{0.4}$)','Log($M_{1.2}$)','Log($M_{4}$)','Log($M_{7.7}$)','Log($M_{14}$)']
    labels_f1_lin = ['$M_{0.01}$','$M_{0.03}$','$M_{0.06}$','$M_{0.4}$','$M_{1.2}$','$M_{4}$','$M_{7.7}$','$M_{14}$']
    labels_f2 = ['R1','R2','R3','$E_{y}$','$E_{o}$','[Fe/H]']
    fig_c1 = corner.corner(f1_SFH,plot_datapoints=False,bins=bins,labels=labels_f1,label_kwargs=dict(fontsize=24),truth_color='r',plot_density=True,quantiles=(0.16, 0.84))
    ndim = len(labels_f1)
    axes = np.array(fig_c1.axes).reshape((ndim, ndim))
 #   Loop over the diagonal
    p=0
    for p in range(ndim):
        ax = axes[p, p]
        ax.axvline(mle_f1[p], color="r")
        ##ax.axvline(median_f1[p], color="g")
        ###ax.axvline(mode_f1[p], color="b")
    for ax in fig_c1.get_axes():
        ax.tick_params(axis='both', labelsize=15)
        
    fig_c1.savefig(path_plots+'corner/'+str(Line0[i])+'_'+str(bin_index)+'_c1.png')
    plt.close()


    fig_c2 = corner.corner(f2_SFH,plot_datapoints=False,bins=bins,labels=labels_f2,label_kwargs=dict(fontsize=24),truth_color='r',plot_density=True,quantiles=(0.16, 0.84))
    ndim = len(labels_f2)
    axes = np.array(fig_c2.axes).reshape((ndim, ndim))
    # Loop over the diagonal
    p=0
    for p in range(ndim):
        ax = axes[p, p]
        ax.axvline(mle_f2[p], color="r")
        #ax.axvline(median_f2[p], color="g")
        #ax.axvline(mode_f2[p], color="b")
    for ax in fig_c2.get_axes():
        ax.tick_params(axis='both', labelsize=15)
    fig_c2.savefig(path_plots+'corner/'+str(Line0[i])+'_'+str(bin_index)+'_c2.png')
    plt.close()


    fig_c3 = corner.corner(10**(f1_SFH),plot_datapoints=False,bins=bins,labels=labels_f1_lin,label_kwargs=dict(fontsize=24),truth_color='r',plot_density=True,quantiles=(0.16, 0.84))
    ndim = len(labels_f1)
    axes = np.array(fig_c3.axes).reshape((ndim, ndim))
    # Loop over the diagonal
    p=0
    for p in range(ndim):
        ax = axes[p, p]
        ax.axvline(10**(mle_f1[p]), color="r")
        #ax.axvline(10**(median_f1[p]), color="g")
        #ax.axvline(10**(mode_f1[p]), color="b")
    for ax in fig_c3.get_axes():
        ax.tick_params(axis='both', labelsize=15)
        
    fig_c3.savefig(path_plots+'corner/'+str(Line0[i])+'_'+str(bin_index)+'_c3.png')
    plt.close()
    

    ax = fig.add_subplot(311)
    ax.plot(diff_lick_ang[:,0]/Lick_indices_error[~mag_index],c=bin_color[bin_index],linewidth=lw)
    num=0
    for num in range (0,len(num_ticks_ang)):
     ax.axvline(x=num_ticks_ang[num],linewidth=lw-4,c='grey')
    ax.axhline(0,linewidth=lw-4,c='grey')
    ax.set_xticks(num_ticks_ang)
    ax.set_xticklabels(Indices_ang,rotation=20,fontsize=ft_12-80)
    #ax.legend(loc='upper right',fontsize=ft_22,frameon=False)
    ax.set_ylabel('$\\Delta$Lick/$\sigma_{Lick}$ [Ang]',fontsize=ft_12,labelpad=pd, fontweight='bold')
    #ax.set_ylim([0,5])
    ax.legend(loc='upper right',fontsize=ft_22,frameon=False)
    ax.minorticks_on()
    ax.tick_params(which='major',length=ltw,width=lw,direction='in',labelsize=ft_1,pad=pd,axis='x',bottom=True,top=True,left=True,right=True)
    ax.tick_params(which='minor',length=ltwm,width=lw,direction='in',labelsize=ft_1,pad=pd,axis='x',bottom=True,top=True,left=True,right=True)
    ax.tick_params(which='major',length=ltw,width=lw,direction='in',labelsize=ft_1,pad=pd,axis='y',bottom=True,top=True,left=True,right=True)
    ax.tick_params(which='minor',length=ltwm,width=lw,direction='in',labelsize=ft_1,pad=pd,axis='y',bottom=True,top=True,left=True,right=True)
    for axis in ['top','bottom','left','right']:
      ax.spines[axis].set_linewidth(lw)

    ax = fig.add_subplot(334)
    ax.plot(T_age,SFH[0,:],linewidth=lw,c=bin_color[bin_index],label=bin_label[bin_index])
    ax.set_xlabel('Log($t_{age}$)',fontsize=ft_12,labelpad=pd, fontweight='bold')
    ax.set_ylabel('Log(SFR $M_{\odot}/Gyr$)',fontsize=ft_12,labelpad=pd, fontweight='bold')
    ax.axvline(np.log10(Beg_Age[t1p][-1]),c='k',linewidth=lw-2)
    ax.axvline(np.log10(Beg_Age[t2p][-1]),c='k',linewidth=lw-2)
    ax.axvline(np.log10(Beg_Age[t3p][-1]),c='k',linewidth=lw-2)
    ax.minorticks_on()
    ax.tick_params(which='major',length=ltw,width=lw,direction='in',labelsize=ft_1,pad=pd,axis='x',bottom=True,top=True,left=True,right=True)
    ax.tick_params(which='minor',length=ltwm,width=lw,direction='in',labelsize=ft_1,pad=pd,axis='x',bottom=True,top=True,left=True,right=True)
    ax.tick_params(which='major',length=ltw,width=lw,direction='in',labelsize=ft_1,pad=pd,axis='y',bottom=True,top=True,left=True,right=True)
    ax.tick_params(which='minor',length=ltwm,width=lw,direction='in',labelsize=ft_1,pad=pd,axis='y',bottom=True,top=True,left=True,right=True)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(lw)
        
    ax = fig.add_subplot(335)
    ax.plot(T_age,np.log10(Mass_present[0,:]),linewidth=lw,c=bin_color[bin_index])
    ax.set_xlabel('Log($t_{age}$)',fontsize=ft_12,labelpad=pd, fontweight='bold')
    ax.set_ylabel('$M_{present}$$M_{\odot}$',fontsize=ft_12,labelpad=pd, fontweight='bold')
    ax.minorticks_on()
    ax.tick_params(which='major',length=ltw,width=lw,direction='in',labelsize=ft_1,pad=pd,axis='x',bottom=True,top=True,left=True,right=True)
    ax.tick_params(which='minor',length=ltwm,width=lw,direction='in',labelsize=ft_1,pad=pd,axis='x',bottom=True,top=True,left=True,right=True)
    ax.tick_params(which='major',length=ltw,width=lw,direction='in',labelsize=ft_1,pad=pd,axis='y',bottom=True,top=True,left=True,right=True)
    ax.tick_params(which='minor',length=ltwm,width=lw,direction='in',labelsize=ft_1,pad=pd,axis='y',bottom=True,top=True,left=True,right=True)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(lw)
    
    ax = fig.add_subplot(336)
    ax.plot(diff_colors[:,0],c=bin_color[bin_index],linewidth=lw)
    num=0
    for num in range (0,len(num_ticks_color)):
      ax.axvline(x=num_ticks_color[num],linewidth=lw-4,c='grey')
    ax.axhline(0,linewidth=lw-4,c='grey')
    ax.set_xticks(num_ticks_color)
    ax.set_xticklabels(Indices_color,rotation=20,fontsize=ft_12-80)
    ax.set_ylabel('$\\Delta$Color mags',fontsize=ft_12,labelpad=pd, fontweight='bold')
    ax.minorticks_on()
    ax.tick_params(which='major',length=ltw,width=lw,direction='in',labelsize=ft_1,pad=pd,axis='x',bottom=True,top=True,left=True,right=True)
    ax.tick_params(which='minor',length=ltwm,width=lw,direction='in',labelsize=ft_1,pad=pd,axis='x',bottom=True,top=True,left=True,right=True)
    ax.tick_params(which='major',length=ltw,width=lw,direction='in',labelsize=ft_1,pad=pd,axis='y',bottom=True,top=True,left=True,right=True)
    ax.tick_params(which='minor',length=ltwm,width=lw,direction='in',labelsize=ft_1,pad=pd,axis='y',bottom=True,top=True,left=True,right=True)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(lw)
     
    ax = fig.add_subplot(337)
    ax.scatter(Dn4000_data,Lick_indices_data[21],c=bin_color[bin_index],s=4000,marker=bin_marker[bin_index],label=bin_label[bin_index])
    ax.legend(loc='upper right',fontsize=ft_22,frameon=False)
    ax.set_ylabel('H$\delta$ EW',fontsize=ft_12,labelpad=pd, fontweight='bold')
    ax.set_xlabel('Dn4000',fontsize=ft_12,labelpad=pd, fontweight='bold')
    ax.set_xlim([1,2.3])
    ax.set_ylim([-5,10])
    ax.legend(loc='upper right',fontsize=ft_22,frameon=False)
    ax.minorticks_on()
    ax.tick_params(which='major',length=ltw,width=lw,direction='in',labelsize=ft_1,pad=pd,axis='x',bottom=True,top=True,left=True,right=True)
    ax.tick_params(which='minor',length=ltwm,width=lw,direction='in',labelsize=ft_1,pad=pd,axis='x',bottom=True,top=True,left=True,right=True)
    ax.tick_params(which='major',length=ltw,width=lw,direction='in',labelsize=ft_1,pad=pd,axis='y',bottom=True,top=True,left=True,right=True)
    ax.tick_params(which='minor',length=ltwm,width=lw,direction='in',labelsize=ft_1,pad=pd,axis='y',bottom=True,top=True,left=True,right=True)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(lw)

    
    ax = fig.add_subplot(338)
    ax.scatter(R1_mle,R2_mle,c=bin_color[bin_index],s=4000,marker=bin_marker[bin_index])
    ax.set_ylabel('R2',fontsize=ft_12,labelpad=pd, fontweight='bold')
    ax.set_xlabel('R1',fontsize=ft_12,labelpad=pd, fontweight='bold')
    ax.axhline(0,linewidth=lw,c='k')
    ax.axvline(0,linewidth=lw,c='k')
    ax.minorticks_on()
    ax.tick_params(which='major',length=ltw,width=lw,direction='in',labelsize=ft_1,pad=pd,axis='x',bottom=True,top=True,left=True,right=True)
    ax.tick_params(which='minor',length=ltwm,width=lw,direction='in',labelsize=ft_1,pad=pd,axis='x',bottom=True,top=True,left=True,right=True)
    ax.tick_params(which='major',length=ltw,width=lw,direction='in',labelsize=ft_1,pad=pd,axis='y',bottom=True,top=True,left=True,right=True)
    ax.tick_params(which='minor',length=ltwm,width=lw,direction='in',labelsize=ft_1,pad=pd,axis='y',bottom=True,top=True,left=True,right=True)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(lw)
 
 
  fig.set_size_inches(140,80)
  fig.suptitle('min $\chi^2$='+str(np.round(nchisq.min(),3))+"|Steps="+str(chain_len)+"|max Tau="+str(np.round(np.max(tau),2)),fontsize=ft_22,fontweight='bold')
  fig.subplots_adjust(left=0.06, bottom=0.04, right=0.98, top=0.95,wspace=0.20, hspace=0.20)
  fig.savefig(path_plots+'sfr_obs/'+str(Line0[i])+'.png')
  plt.close()
 

    
