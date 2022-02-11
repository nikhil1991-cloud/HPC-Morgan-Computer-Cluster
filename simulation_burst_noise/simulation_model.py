import numpy as np

def make_models(Initial_theta,t_age,CI,CR,CB,D4,Photo,PhotoAB,L_R,L_B,L_I,D_I,EBV_I,Metal_I,D_lmd,nsa_mass,DATA_4SSP):
    Given_Z = Initial_theta[-1]
    Given_E_young = Initial_theta[-3]
    Given_E_old = Initial_theta[-2]
    Given_M = 10**(Initial_theta[:-3])
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
    #calculate bc03 mass
    Z_all= np.array([-2.2490,-1.6464,-0.6392,-0.3300,0.0932,0.5595])
    Best_Metal = Given_Z
    IN_H = np.where(Z_all>Best_Metal)[0][0]
    IN_L = np.where(Z_all<Best_Metal)[0][-1]
    M2 = DATA_4SSP[IN_H,:,6]
    M1 = DATA_4SSP[IN_L,:,6]
    bc03_M = (M1*(Z_all[IN_H]-Best_Metal) + M2*(Best_Metal-Z_all[IN_L]))/(Z_all[IN_H]-Z_all[IN_L])
    #Specify old young age
    young_t = 1
    old_t = 1
    LR = L_R
    LB = L_B
    LI = L_I
    DI = D_I
    EBV = EBV_I
    Metal = Metal_I
    Delta_Ebv = EBV[3] - EBV[2]
    SD4 = D4
    SPR = CR
    SPB = CB
    SPI = CI
    SMg = Photo
    Delta_lmd = D_lmd
    SMgAB = PhotoAB#AB Mag normalization
    #Get indices for upper and lower data points near the predicted metallicity
    metal_high = np.where(Metal>Given_Z)[0][0]
    metal_low = np.where(Metal<Given_Z)[0][-1]
    Delta_metal = Metal[metal_high] - Metal[metal_low]
    #Get indices for upper and lower data points near the predicted yound age E(B-V)
    EBV_I_young = Given_E_young/Delta_Ebv
    EBV_low_young = int(np.floor(EBV_I_young))
    EBV_high_young = int(np.floor(EBV_I_young+1))
    #Get indices for upper and lower data points near the predicted old age E(B-V)
    EBV_I_old = Given_E_old/Delta_Ebv
    EBV_low_old = int(np.floor(EBV_I_old))
    EBV_high_old = int(np.floor(EBV_I_old+1))
    alpha_bestfit = Given_M/bc03_M
    #Interpolate for metallicity
    SD4_1 = ((SD4[metal_low,:,:,:]*(Metal[metal_high]-Given_Z)) + (SD4[metal_high,:,:,:]*(Given_Z-Metal[metal_low])))/(Delta_metal)
    SPR_1 = ((SPR[metal_low,:,:,:]*(Metal[metal_high]-Given_Z)) + (SPR[metal_high,:,:,:]*(Given_Z-Metal[metal_low])))/(Delta_metal)
    SPB_1 = ((SPB[metal_low,:,:,:]*(Metal[metal_high]-Given_Z)) + (SPB[metal_high,:,:,:]*(Given_Z-Metal[metal_low])))/(Delta_metal)
    SPI_1 = ((SPI[metal_low,:,:,:]*(Metal[metal_high]-Given_Z)) + (SPI[metal_high,:,:,:]*(Given_Z-Metal[metal_low])))/(Delta_metal)
    SMg_1 = ((SMg[metal_low,:,:,:]*(Metal[metal_high]-Given_Z)) + (SMg[metal_high,:,:,:]*(Given_Z-Metal[metal_low])))/(Delta_metal)
    SMgAB_1 = ((SMgAB[metal_low,:,:]*(Metal[metal_high]-Given_Z)) + (SMgAB[metal_high,:,:]*(Given_Z-Metal[metal_low])))/(Delta_metal)
    #Interpolate for extinction at young ages
    SD4_young = ((SD4_1[:,EBV_low_young,:young_t]*(EBV[EBV_high_young]-Given_E_young)) + (SD4_1[:,EBV_high_young,:young_t]*(Given_E_young-EBV[EBV_low_young])))/(Delta_Ebv)
    SPR_young = (((SPR_1[:,EBV_low_young,:young_t]*(EBV[EBV_high_young]-Given_E_young)) + (SPR_1[:,EBV_high_young,:young_t]*(Given_E_young-EBV[EBV_low_young])))/(Delta_Ebv))
    SPB_young = (((SPB_1[:,EBV_low_young,:young_t]*(EBV[EBV_high_young]-Given_E_young)) + (SPB_1[:,EBV_high_young,:young_t]*(Given_E_young-EBV[EBV_low_young])))/(Delta_Ebv))
    SPI_young = (((SPI_1[:,EBV_low_young,:young_t]*(EBV[EBV_high_young]-Given_E_young)) + (SPI_1[:,EBV_high_young,:young_t]*(Given_E_young-EBV[EBV_low_young])))/(Delta_Ebv))
    SMg_young = (((SMg_1[:,EBV_low_young,:young_t]*(EBV[EBV_high_young]-Given_E_young)) + (SMg_1[:,EBV_high_young,:young_t]*(Given_E_young-EBV[EBV_low_young])))/(Delta_Ebv))
    SMgAB_young = ((SMgAB_1[:,EBV_low_young]*(EBV[EBV_high_young]-Given_E_young)) + (SMgAB_1[:,EBV_high_young]*(Given_E_young-EBV[EBV_low_young])))/(Delta_Ebv)
    #Interpolate for extinction at old ages
    SD4_old = ((SD4_1[:,EBV_low_old,old_t:]*(EBV[EBV_high_old]-Given_E_old)) + (SD4_1[:,EBV_high_old,old_t:]*(Given_E_old-EBV[EBV_low_old])))/(Delta_Ebv)
    SPR_old = (((SPR_1[:,EBV_low_old,old_t:]*(EBV[EBV_high_old]-Given_E_old)) + (SPR_1[:,EBV_high_old,old_t:]*(Given_E_old-EBV[EBV_low_old])))/(Delta_Ebv))
    SPB_old = (((SPB_1[:,EBV_low_old,old_t:]*(EBV[EBV_high_old]-Given_E_old)) + (SPB_1[:,EBV_high_old,old_t:]*(Given_E_old-EBV[EBV_low_old])))/(Delta_Ebv))
    SPI_old = (((SPI_1[:,EBV_low_old,old_t:]*(EBV[EBV_high_old]-Given_E_old)) + (SPI_1[:,EBV_high_old,old_t:]*(Given_E_old-EBV[EBV_low_old])))/(Delta_Ebv))
    SMg_old = (((SMg_1[:,EBV_low_old,old_t:]*(EBV[EBV_high_old]-Given_E_old)) + (SMg_1[:,EBV_high_old,old_t:]*(Given_E_old-EBV[EBV_low_old])))/(Delta_Ebv))
    SMgAB_old = ((SMgAB_1[:,EBV_low_old]*(EBV[EBV_high_old]-Given_E_old)) + (SMgAB_1[:,EBV_high_old]*(Given_E_old-EBV[EBV_low_old])))/(Delta_Ebv)
    #Concatenate young and old ages
    SD4_2 = np.concatenate((SD4_young,SD4_old),axis=1)
    SPR_2 = np.concatenate((SPR_young,SPR_old),axis=1)
    SPB_2 = np.concatenate((SPB_young,SPB_old),axis=1)
    SPI_2 = np.concatenate((SPI_young,SPI_old),axis=1)
    SMg_2 = np.concatenate((SMg_young,SMg_old),axis=1)
    SMgAB_2 = SMgAB_old
    SMg_2[[0,1],:] = SMg_2[[1,0],:]
    SMgAB_2[[0,1]] = SMgAB_2[[1,0]]
    #Calculate Observables for CSP
    D4000_Model = (np.sum(SD4_2[0,:]*alpha_bestfit))/(np.sum(SD4_2[1,:]*alpha_bestfit))
    CR_T = np.sum(SPR_2*alpha_bestfit,axis=1)
    CB_T = np.sum(SPB_2*alpha_bestfit,axis=1)
    S0_T = (CR_T - CB_T)/(LR-LB)
    C0_T = S0_T*(LI-LB) + CB_T
    CI_T = np.sum(SPI_2*alpha_bestfit,axis=1)
    LICK_Model = (DI - CI_T/C0_T)
    Photo = -2.5*np.log10(np.sum(SMg_2*alpha_bestfit,axis=1)/(SMgAB_2))
    MAGS_Model = Photo
    return D4000_Model,LICK_Model,MAGS_Model


def get_Mbc03(Metal,DATA_4SSP):
    Z_all= np.array([-2.2490,-1.6464,-0.6392,-0.3300,0.0932,0.5595])
    Best_Metal = Metal
    IN_H = np.where(Z_all>Best_Metal)[0][0]
    IN_L = np.where(Z_all<Best_Metal)[0][-1]
    M2 = DATA_4SSP[IN_H,:,6]
    M1 = DATA_4SSP[IN_L,:,6]
    bc03_M = (M1*(Z_all[IN_H]-Best_Metal) + M2*(Best_Metal-Z_all[IN_L]))/(Z_all[IN_H]-Z_all[IN_L])
    return bc03_M

def get_dt(t_age):
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
    return delta_T
