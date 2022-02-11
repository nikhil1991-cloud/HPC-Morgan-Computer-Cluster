import numpy as np


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
