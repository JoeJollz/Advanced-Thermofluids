# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 15:21:59 2024

@author: jrjol
## testing
"""
import numpy as np
from scipy.optimize import fsolve
from numpy.polynomial import polynomial
from sympy import var, Eq, solve, real_roots
import math as m
import matplotlib.pyplot as plt
import pygad

#%%
## IAPWS 2007
def WaterSat_H_PT(P,T):  # liquid saturdated water enthalpy, know P and T. 
# TO DO how to get the temperature just from the saturated pressure of the water.  
    '''
    
    Parameters
    ----------
    P : float or int
        Pressure of the saturated liquid water in kPa
    T : float or int
        Temperature of the saturated liquid water in deg Cel

    Returns
    -------
    H : enthalpy (kJ/kg)

    '''
    
    R = 0.461526 # kJ/kgK
    
    I = [0,0,0,0,0,0,0,0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 8, 8, 21, 23, 29, 30, 31, 32]
    J = [-2, -1, 0, 1, 2, 3, 4, 5, -9, -7, -1, 0, 1, 3, -3, 0, 1, 3, 17, -4, 0, 6, -5, -2, 10, -8, -11, -6, -29, -31, -38, -39, -40, -41]
    n = [0.14632971213167, -0.84548187169114, -0.37563603672040e+1, \
        0.33855169168385e+1,  -0.95791963387872, 0.15772038513228, \
        -0.16616417199501e-1, 0.81214629983568e-3, 0.28319080123804e-3, \
        -0.60706301565874e-3, -0.18990068218419e-1, -0.32529748770505e-1, \
        -0.21841717175414e-1, -0.52838357969930e-4, -0.47184321073267e-3, \
        -0.30001780793026e-3, 0.47661393906987e-4, -0.44141845330846e-5, \
        -0.72694996297594e-15, -0.31679644845054e-4, -0.28270797985312e-5, \
        -0.85205128120103e-9, -0.224252981908000e-5, -0.65171222895601e-6, \
        -0.14341729937924e-12, -0.40516996860117e-6, -0.12734301741641e-8, \
        -0.17424871230634e-9, -0.68762131295531e-18, 0.14478307828521e-19, \
        0.26335781662795e-22, -0.11947622640071e-22, 0.18228094581404e-23, \
        -0.93537087292458e-25]
    
    T = T+273.15 # conversion from celcius to kelvin
    # specific enthalpy for range 273.15K<T<623.15K and  0MPa<P<100MPa
    enthalpy = 0 # kJ/kg
    sum_1 = 0
    pi = P/(16.53*10**3) # kPa/kPa
    tau = 1386/T   # kelvin/kelvin
    
    for i in range(0,34):
        sum_1 +=n[i]*(7.1-pi)**I[i]*J[i]*(tau-1.222)**(J[i]-1)
    
    H = sum_1*R*T*tau

    return H
test = WaterSat_H_PT(1, 6.97)

#%%
# IAPWS 2007 - Section 6.1
def SteamSat_H_PT(P, T):  ## saturated water gas enthalpy equation from P and T
    '''
    Parameters
    ----------
    P : float or int
        Pressure of the saturated liquid water in kPa
    T : float or int
        Temperature of the saturated liquid water in deg Cel

    Returns
    -------
    H : enthalpy (kJ/kg)
    '''
    
    
    
    JO = [0, 1, -5, -4, -3, -2, -1, 2, 3]
    nO = [-0.96927686500217e+1, 0.10086655968018e+2, -0.56087911283020e-2, \
         0.71452738081455e-1, -0.40710498223928, 0.14240819171444e+1, \
        -0.43839511319450e+1, -0.28408632460772, 0.21268463753307e-1]
    
    I = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 5, 6, 6, 6, 7, \
         7, 7, 8, 8, 9, 10, 10, 10, 16, 16, 18, 20, 20, 20, 21, 22, 23, 24, 24, \
             24]
    J = [0, 1, 2, 3, 6, 1, 2, 4, 7, 36, 0, 1, 3, 6, 35, 1, 2, 3, 7, 3, 16, 35,\
         0, 11, 25, 8, 36, 13, 4, 10, 14, 29, 50, 57, 20, 35, 48, 21, 53, 39, \
        26, 40, 58
        ]
    n = [ -0.17731742473213e-2, -0.17834862292358e-1, -0.45996013696365e-1,\
         -0.57581259083432e-1, -0.50325278727930e-1, -0.33032641670203e-4,\
        -0.18948987516315e-3, -0.39392777243355e-2, -0.43797295650573e-1, \
        -0.26674547914087e-4, 0.20481737692309e-7, 0.43870667284435e-6, \
        -0.32277677238570e-4, -0.15033924542148e-2, -0.40668253562649e-1, \
        -0.78847309559367e-9, 0.12790717852285e-7, 0.48225372718507e-6, \
        0.22922076337661e-5, -0.16714766451061e-10, -0.21171472321355e-2, \
        -0.23895741934104e+2, -0.59059564324270e-17, -0.12621808899101e-5, \
        -0.38946842435739e-1, 0.11256211360459e-10, -0.82311340897998e+1, \
        0.19809712802088e-7, 0.10406965210174e-18, -0.10234747095929e-12, \
        -0.10018179379511e-8, -0.80882908646985e-10, 0.10693031879409, \
        -0.33662250574171, 0.89185845355421e-24, 0.30629316876232e-12, \
        -0.42002467698208e-5, -0.59056029685639e-25, 0.37826947613457e-5, \
        -0.12768608934681e-14, 0.73087610595061e-28, 0.55414715350778e-16, \
        -0.94369707241210e-6]
    
    R = 0.461526 # kJ/kgK
    
    T = T+ 273.15 # deg cel to kelvin
    
    pi = P / 1000 # kPa/kPa
    tau = 540/T # Kelvin/kelvin
    
    sum_1 = 0
    
    yO = 0#np.log(pi)
    for i in range(0,9):
        sum_1 += nO[i]*JO[i]*tau**(JO[i]-1)
    yO = sum_1
    sum_1 = 0
    for i in range(0, 43):
        sum_1 += n[i]*pi**I[i]*J[i]*(tau-0.5)**(J[i]-1)
    yr = sum_1
    
    H = tau*R*T*(yO+yr)
    
    return H

test_gas = SteamSat_H_PT(0.611, 0.01)
    

#%%
# IAPWS 2007 - Section 6.2 
# Metastable-Vapor Region- super heated steam values

def H_vap_PT(P, T): # Correct
    '''
    Calculates the enthalpy for superheated steam. Do not use this function and 
    input a pressure with a temperature lower then the corresponding Tsat, the
    results will be incorrect and invalid. 

    Parameters
    ----------
    P : float
        Pressure of the vapor in kPa.
    T : float
        Temperature of the vapor (deg Cel) (ensure it is greater than the Tsat for the given
                                  pressure).

    Returns
    -------
    H : float
        enthalpy of superheated steam (kJ/kg).

    '''
    
    ### ERROR CHECK ###
    Tsat = Sat_T_P(P)
    if T< Tsat:
         print('ERROR (superheated steam): Temperature inputted is less than the saturated temperature for\
  the given pressure. Reevaluate your pressure and input temperature')
         return 

    
    
    ### CALCULATION BEGINS HERE ### 
    
    JO = [0, 1, -5, -4, -3, -2, -1, 2, 3]
    nO = [-0.96937268393049e+1, 0.10087275970006e+2, -0.56087911283020e-2, \
             0.71452738081455e-1, -0.40710498223928, 0.14240819171444e+1, \
            -0.43839511319450e+1, -0.28408632460772, 0.21268463753307e-1
            ]
        
    I = [1, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5]
    J = [0, 2, 5, 11, 1, 7, 16, 4, 16, 7, 10, 9, 10]
    n = [-0.73362260186506, -0.88223831943146e-1, -0.72334555213245e-1, \
         -0.40813178534455e-2, 0.20097803380207e-2, -0.53045921898642e-1, -0.76190409086970e-2, \
        -0.63498037657313e-2, -0.86043093028588e-1, 0.75321581522770e-2, -0.79238375446139e-2, \
        -0.22888160778447e-3, -0.26456501482810e-2]
        
        
    R = 0.461526 # kJ/kgK
    
    T = T+ 273.15 # deg cel to kelvin
    
    pi = P / 1000 # kPa/kPa
    tau = 540/T # Kelvin/kelvin
    
    sum_1 = 0
    
    yO = 0#np.log(pi)
    for i in range(0,9):
        sum_1 += nO[i]*JO[i]*tau**(JO[i]-1)
    yO = sum_1
    sum_1 = 0
    for i in range(0, 13):
        sum_1 += n[i]*pi**I[i]*J[i]*(tau-0.5)**(J[i]-1)
    yr = sum_1
    
    H = tau*R*T*(yO+yr)
    
    return H
    
#test_superheated = H_vap_PT(10, 50)



#%%
# IAPWS 2007 - Section 8.2
def Sat_T_P(P):
    '''

    Parameters
    ----------
    P : Float or int
        Saturation pressure of water (kPa)

    Returns
    -------
    Float
        Returns the saturation temperature in deg cel for water.

    '''
    
    ## suitable within the pressure range 0.611kPa < P < 22000 kPa
    
    n = [ 0.11670521452767e+4, -0.72421316703206e+6, -0.17073846940092e+2, \
        0.12020824702470e+5, -0.32325550322333e+7, 0.14915108613530e+2, \
        -0.48232657361591e+4, 0.40511340542057e+6, -0.23855557567849, \
        0.65017534844798e+3
        ]
    Beta = (P/1000)**(1/4)
    
    E = Beta**2 + n[2]*Beta + n[5]
    F = n[0]*Beta**2 + n[3]*Beta + n[6]
    G = n[1]*Beta**2 + n[4]*Beta + n[7]
    
    D = 2*G / (-F-(F**2-4*E*G)**(1/2))
    
    Ts = (n[9]+D-((n[9]+D)**2-4*(n[8]+n[9]*D))**(1/2))/2
    
    return Ts - 273.15 
     
Test_Temp_sat = Sat_T_P(100) 
test_superheated = H_vap_PT(10, 95)

#%%
# IAPWS - Section 6.3.1 (backwards equation for T(p,h)).
# Suitabel for superheated vapor with pressures less than 4MPa.

def Superheated_T_pH(P, H):
    '''
    Calculates the temperture of superheated vapor from pressure and enthalpy
    input.

    Parameters
    ----------
    P : float
        Pressure of the superheated vapor in kPa.
    H : float
        Enthalpy of the superheated vapor in kJ/kg.

    Returns
    -------
    Temperature : float
        Temperature of the vapor in deg Cel.

    '''
    
    eta = H/2000
    pi = P / 1000
    
    I = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, \
         3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 7]
    J = [0, 1, 2, 3, 7, 20, 0, 1, 2, 3, 7, 9, 11, 18, 44, 0, 2, 7, 36, 38, 40,\
         42, 44, 24, 44, 12, 32, 44, 32, 36, 42, 34, 44, 28]
    n = [ 0.10898952318288e+4, 0.84951654495535e+3, -0.10781748091826e+3, \
        0.33153654801263e+2, -0.74232016790248e+1, 0.11765048724356e+2, \
        0.18445749355790e+1, -0.41792700549624e+1, 0.62478196935812e+1, \
        -0.17344563108114e+2, -0.20058176862096e+3, 0.27196065473796e+3, \
        -0.45511318285818e+3, 0.30919688604755e+4, 0.25226640357872e+6, \
        -0.61707422868339e-2, -0.31078046629583, 0.11670873077107e+2, \
        0.12812798404046e+9, -0.98554909623276e+9,  0.28224546973002e+10, \
        -0.35948971410703e+10, 0.17227349913197e+10, -0.13551334240775e+5, \
        0.12848734664650e+8, 0.13865724283226e+1, 0.23598832556514e+6, \
        -0.13105236545054e+8,  0.73999835474766e+4, -0.55196697030060e+6, \
        0.37154085996233e+7, 0.19127729239660e+5, -0.41535164835634e+6, \
        -0.62459855192507e+2]
        
    sum1 = 0
    for i in range(0, 34):
        sum1 += n[i]*pi**I[i]*(eta-2.1)**J[i]
    
    Temperature = sum1 - 273.15
    return Temperature


#%%

X_initial = 55 # initial concentration guess 
tol = 10e-3
X_old = X_initial
X_new = 70

A = [ -2.00755, 0.16976, -3.133362e-3, 1.97668e-5]
B = [124.937, -7.71649, 0.152286, -7.95090e-4]
C = 7.05
D = -1596.49
E = -104095.5


## IAPWS 2007
R = 0.461526 # kJ/kgK


X_initial = 55 # initial concentration guess 
tol = 10e-3
X_old = X_initial
X_new = 70
#t = 70

def rT__(X, t, A, B):
    #print(X)
    sum_1 = 0
    sum_2 = 0
    for i in range(0,4):
        sum_1 += B[i]*X**i
        sum_2 += A[i]*X**i
    rT = (t - sum_1)/(sum_2)+273.16
    return rT

def P__(rT, C, D, E):
    ans = C+D/rT + E/rT**2  
    P = np.exp(ans)    # this inverse log needs to be double checked!!
    return P

def rt__(P, C, D, E): # Takes in he saturation pressure, produces the sat T.
    '''
    refrigeration temperature in deg C, from the saturation pressure, for the 
    LiBr-Water solution. 
    

    Parameters
    ----------
    P : flaot
        The saturation pressure in kPa.
    C : List or Array
        constants.
    D : float
        constants.
    E : float
        constants.

    Returns
    -------
    rt : float
        Refrigeration temperature (deg Cel).

    '''

    rt = (-2*E)/(D+(D**2-4*E*(C-np.log10(P)))**(0.5))-273.16
    
    return rt

def t__(X, rt, B, A):
    '''
    returns the LiBr-H20 solution temperature, taking in the concentration of 
    LiBr in % terms, and the rt, which can be calculated using the above function
    for the refrigerant temperature, at a given pressure. 

    Parameters
    ----------
    X : float
        Concentration of LiBr in % terms.
    rt : float
        Refrigeration temperature in deg C
    B : Array or List
        Constants
    A : Array or List.
        Constants

    Returns
    -------
    t : float
        Temperature of the solution in deg C.

    '''
    sum_1 = 0
    sum_2 = 0
    for i in range(0,4):
        sum_1 +=B[i]*X**i
        sum_2 += A[i]*X**i
    t = sum_1 + rt*sum_2
    return t
#%%
'''
Lithium bromide - H2O equations
Faruque W. 2020.

'''

def X__(t, rt, B, A):
    coeff = np.zeros(4)
    coeff[0] = B[0]+rt*A[0]-t
    coeff[1] = B[1] + rt * A[1] 
    coeff[2] = B[2] +rt * A[2]
    coeff[3] = B[3] +rt * A[3]
    
    f = lambda x: coeff[3]*x**3+coeff[2]*x**2 + coeff[1]*x+coeff[0]
    s =fsolve(f, [100, 50, 0])
    
    return [num for num in s if 40 <= num <= 70][0]
    

A_t1 = [-2024.33, 163.309, -4.88161, 6.302948e-2, -2.913705e-4]
B_t1 = [18.2829, -1.1691757, 3.248041e-2, -4.034184e-4, 1.8520569e-6]
C_t1 = [-3.7008214e-2, 2.8877666e-3, -8.1313015e-5, 9.9116628e-7, -4.4441207e-9]


def H_Xt(X, t, A_t1, B_t1, C_t1):  # CORRECT saturated liquid solution LiBr, slight inaccuarcy present
    '''
    t in deg C
    X in %
    returns Enthalpy estimate (kJ/kg)
    
    '''

    sum_1 = 0
    sum_2 = 0
    sum_3 = 0
    for i in range(0,5):
        sum_1 += A_t1[i]*X**i
        sum_2 += B_t1[i]*X**i
        sum_3 += C_t1[i]*X**i
    ans = sum_1 + t*sum_2 + t**2*sum_3
    return ans

hf = WaterSat_H_PT(0.676, Sat_T_P(0.676))
hg = SteamSat_H_PT(0.676, Sat_T_P(0.676))
vapor_quality_15 = 0.005
h15_steam_compo = vapor_quality_15*(hg-hf)+hf
H_test = H_Xt(62.16, 44.96, A_t1, B_t1, C_t1) + h15_steam_compo  

# Table 3 - entropy constants
Entropy_constants = {
    0 : [5.127558e-1, -1.393954e-2, 2.924145e-5, 9.035697e-7],
    1 : [1.226780e-2, -9.156820e-5, 1.820453e-8, -7.991806e-10],
    2 : [-1.364895e-5, 1.068904e-7, -1.381109e-9, 1.529784e-11],
    3 : [1.021501e-8, 0, 0, 0]
    
    }


def S_Xt(X, t, Entropy_constants):
    sum_1 = 0
    for i in range(0,4):
        for j in range(0,4):
            sum_1 += Entropy_constants[i][j]*X**j*t**i
    return sum_1

S = S_Xt(65, 45.8, Entropy_constants)

ans = H_Xt(62.16, 89.36, A_t1, B_t1, C_t1)

# mass flow 10

rt1 = rt__(0.676, C, D, E)
t1 = t__(50, rt1, B, A)
X_old = 5
X_new = 50
while abs(X_new-X_old)>10e-3:
    
    X_old = X_new
    t = t__(X_old, rt1, B, A)
    X_new = X__(t, rt1, B, A)
    
    
#X_test = X__(t10, rt10, B, A)
#S10 = S_Xt(56.48, t10, Entropy_constants)
H10 = H_Xt(56.48, 32.72, A_t1, B_t1, C_t1) # this function works, need to ge the correct inputs

# mass flow 11

S11 = S_Xt(55, 43.95, Entropy_constants)


# mass flow 13
rt13 = rt__(10, C, D, E)
t13 = t__(65, rt13, B, A)
S13 = S_Xt(65, t13, Entropy_constants)
H13 = H_Xt(65, t13, A_t1, B_t1, C_t1)




# P = 10
# rt = rt__(P, C, D, E)
# counter = 0
# while abs(X_new-X_old)> tol:
#     X_old = X_new
#     counter +=1
#     print(counter)
#     # rT = rT__(X_old, t, A, B)
#     # print('not stuck on rt')
#     # P = P__(rT, C, D, E)
#     # print('not stuck on P')
#     # rt = rt__(P, C, D, E)
#     # print('Not stuck on rt')
#     t = t__(X_old, rt, B, A)
#     # print('Not stuck on t')
#     X_new = X__(t, rt, B, A)
#     ### Something strange with the concentration. Why does it not iterate, or can we simply select the conc and P. 
    
    
#     # start from P
#     # get rt from P
#     # inital guess X
#     # iterator start
#     # calc t from X, rt
#     # calc X from t
#     # iterator end


# import coolprop.coolprop as cp

# H = cp.PropsSI('H','T',300.0,'P',101325.0,'water')
    
Ts = Sat_T_P(7.406)
enthalpy_test= WaterSat_H_PT(7.406, Ts)


T_outside = 32
Qcooling = 10.7
Qe = Qcooling
P_high = 7.406 # kPa
P_low = 0.676 # kPa
C_low = 56.48 #%
C_high = 62.16 #%
Qpump = 0.02

P_lows = []
P_highs = []
COPs = []
Qgs = []

for P_low in np.arange(0.68, 4.3, 0.025):
    
    for P_high in np.arange(4.8, 9.0, 0.025):
        P_lows.append(P_low)
        P_highs.append(P_high)
        t17 = Sat_T_P(P_high) # sat T of liquid water
        #T = Sat_T_P(4.8)
        h17 = WaterSat_H_PT(P_high, t17)  # sat liquid water high pressure
        
        
        h18 = h17   # expansion value, H18 is at the low pressure state now.
        hf = WaterSat_H_PT(P_low, Sat_T_P(P_low))
        hg = SteamSat_H_PT(P_low, Sat_T_P(P_low)) 
        vapor_quality_18 = (h18-hf)/(hg-hf)
        
        t19 = Sat_T_P(P_low)
        h19 = SteamSat_H_PT(P_low, t19) # sat vapor low pressure
        
        m19 = Qe/(h19-h18) #kg/s 
        m18 = m17 = m16 = m19
        t18 = t19 
        
        rt10 = rt__(P_low, C, D, E)
        t10 = t__(C_low, rt10, B, A)
        h10 = H_Xt(C_low, t10, A_t1, B_t1, C_t1)
        
        m15 = -m19/(1-(C_high/C_low))
        m14 = m13 = m15
        
        m10 = C_high/C_low*m15
        m11 = m12 = m10 
        
        h11 = (m10*h10+Qpump)/m11 
        
        # for h15 we assume a vapor quality of 0.005, after the expansion valve. hence 
        # h15 = h_LiBr_solution + h15_vapor (0.005)
        hf = WaterSat_H_PT(P_low, Sat_T_P(P_low))
        hg = SteamSat_H_PT(P_low, Sat_T_P(P_low))
        vapor_quality_15 = 0.005
        h15_steam_compo = vapor_quality_15*(hg-hf)+hf
        rt15 = rt__(P_low, C, D, E)
        t15 = t__(C_high, rt15, B, A)
        h15 = H_Xt(C_high, t15, A_t1, B_t1, C_t1) + h15_steam_compo 
        h14 = h15
        
        rt13 = rt__(P_high, C, D, E)
        t13 = t__(C_high, rt13, B, A)
        h13 = H_Xt(C_high, t13, A_t1, B_t1, C_t1)
        
        # Heat exhanger balance
        # Stream 1 energy balance 
        Q_he = m13*h13 - m14*h14
        # Stream 2 energy balance
        h12 = (m11*h11 + Q_he)/m12
        
        # Solving for stream 16
        # firstly working on the condensor, we know there is a difference of saturated vapor 
        # to saturated liquid, as stream 16 is already working in the superheated region. 
        
        Tsat = Sat_T_P(P_high)  # T for stream 16 always has to be greater than Tsat.
        h16 = H_vap_PT(P_high,  76.76)
        Qc = m16*h16 - m17*h17 # correct
        Qg = m13*h13 + m16*h16 - m12*h12  # incorrect
        COP = Qe/Qg
        COPs.append(COP)
        Qgs.append(Qg)

## now increment the T a little more to see how the COP and Qg are effected. Do this
# several times. When satisfied, then change either P_low, P_high, C_low, C_high.

#NOTE : the temperature of saturated liquid water under high pressure must not be less then
# the external temperature of moroccan air :) 
P_lows = np.array(P_lows)
P_highs = np.array(P_highs)
COPs = np.array(COPs)

# Reshape COPs to be a 2D array for plotting
num_P_lows = len(np.unique(P_lows))
num_P_highs = len(np.unique(P_highs))
COPs = COPs.reshape(num_P_highs, num_P_lows)

# Create the heatmap
plt.imshow(COPs, extent=[min(P_lows), max(P_lows), min(P_highs), max(P_highs)],
           aspect='auto', origin='lower')
plt.colorbar(label='Coefficient of Performance (COP)')
plt.xlabel('Lower system pressure (kPa)')
plt.ylabel('Upper system pressure (kPa)')
plt.title('Heatmap of COP vs Varying System Pressures')
plt.show()


T_outside = 32
Qcooling = 10.7
Qe = Qcooling
P_high = 4.8 # kPa
P_low = 0.68 # kPa
Qpump = 0.02 # kW

C_low = 56.48 #%
C_high = 62.16 #%
Qpump = 0.02

C_lows = []
C_highs = []
C_COPs = []
Qgs = []

for C_low in np.arange(40, 55, 1):
    
    for C_high in np.arange(55, 70, 1):
        C_lows.append(C_low)
        C_highs.append(C_high)
        t17 = Sat_T_P(P_high) # sat T of liquid water
        #T = Sat_T_P(4.8)
        h17 = WaterSat_H_PT(P_high, t17)  # sat liquid water high pressure
        
        
        h18 = h17   # expansion value, H18 is at the low pressure state now.
        hf = WaterSat_H_PT(P_low, Sat_T_P(P_low))
        hg = SteamSat_H_PT(P_low, Sat_T_P(P_low)) 
        vapor_quality_18 = (h18-hf)/(hg-hf)
        
        t19 = Sat_T_P(P_low)
        h19 = SteamSat_H_PT(P_low, t19) # sat vapor low pressure
        
        m19 = Qe/(h19-h18) #kg/s 
        m18 = m17 = m16 = m19
        t18 = t19 
        
        rt10 = rt__(P_low, C, D, E)
        t10 = t__(C_low, rt10, B, A)
        h10 = H_Xt(C_low, t10, A_t1, B_t1, C_t1)
        
        m15 = -m19/(1-(C_high/C_low))
        m14 = m13 = m15
        
        m10 = C_high/C_low*m15
        m11 = m12 = m10 
        
        h11 = (m10*h10+Qpump)/m11 
        
        # for h15 we assume a vapor quality of 0.005, after the expansion valve. hence 
        # h15 = h_LiBr_solution + h15_vapor (0.005)
        hf = WaterSat_H_PT(P_low, Sat_T_P(P_low))
        hg = SteamSat_H_PT(P_low, Sat_T_P(P_low))
        vapor_quality_15 = 0.005
        h15_steam_compo = vapor_quality_15*(hg-hf)+hf
        rt15 = rt__(P_low, C, D, E)
        t15 = t__(C_high, rt15, B, A)
        h15 = H_Xt(C_high, t15, A_t1, B_t1, C_t1) + h15_steam_compo 
        h14 = h15
        
        rt13 = rt__(P_high, C, D, E)
        t13 = t__(C_high, rt13, B, A)
        h13 = H_Xt(C_high, t13, A_t1, B_t1, C_t1)
        
        # Heat exhanger balance
        # Stream 1 energy balance 
        Q_he = m13*h13 - m14*h14
        # Stream 2 energy balance
        h12 = (m11*h11 + Q_he)/m12
        
        # Solving for stream 16
        # firstly working on the condensor, we know there is a difference of saturated vapor 
        # to saturated liquid, as stream 16 is already working in the superheated region. 
        
        Tsat = Sat_T_P(P_high)  # T for stream 16 always has to be greater than Tsat.
        h16 = H_vap_PT(P_high,  76.76)
        Qc = m16*h16 - m17*h17 # correct
        Qg = m13*h13 + m16*h16 - m12*h12  # incorrect
        COP = Qe/Qg
        C_COPs.append(COP)
        Qgs.append(Qg)

C_lows = np.array(C_lows)
C_highs = np.array(C_highs)
C_COPs = np.array(C_COPs)

# Reshape COPs to be a 2D array for plotting
num_C_lows = len(np.unique(C_lows))
num_C_highs = len(np.unique(C_highs))
C_COPs = C_COPs.reshape(num_C_highs, num_C_lows)

# Create the heatmap
plt.imshow(C_COPs, extent=[min(C_lows), max(C_lows), min(C_highs), max(C_highs)],
           aspect='auto', origin='lower')
plt.colorbar(label='Coefficient of Performance (COP)')
plt.xlabel('Lower LiBr Concentration (%)')
plt.ylabel('Upper LiBr Concentration (%)')
plt.title('Heatmap of COP vs Varying LiBr Concentrations')
plt.show()


desired_output = 1
def LiBr_cycle_fitness(ga_instance, solution_LiBr, solution_idx_LiBr):
    P_low = (( solution_LiBr[0] - 0) /(100 - 0)) * (4.8 - 0.676)+0.676
    if P_low <0.676:
        return -np.inf
    P_high = ((solution_LiBr[1] - 0) / (100 - 0)) * (10 - 4.8) + 4.8
   # rescaled_number = ((original_number - original_min) / (original_max - original_min)) * (target_max - target_min) + target_min
    if  P_high< (P_low+0.5) or P_high> 10 or P_high<4.8:
        return -np.inf
    C_low = solution_LiBr[2]
    if C_low < 40:
        return -np.inf
    C_high = solution_LiBr[3]
    if C_high<C_low or C_high > 70:
        return -np.inf
    

    
    t6 = Sat_T_P(P_high) # sat T of liquid water # t6 has just flowed out of the condensor, which is wrt to outside air temp
    # if t6>T_outside then the condenser would not work effectively, and would start to act like a evaportator. 
    # t5 the inflow stream calculated later in this code is designed to operate at a superheated vapor for P_high,
    # so t5>T_outside always. 
    if t6<T_outside:
        return -np.inf
    #T = Sat_T_P(4.8)
    h6 = WaterSat_H_PT(P_high, t6)  # sat liquid water high pressure
    
    
    h7 = h6   # expansion value, H18 is at the low pressure state now.
    hf = WaterSat_H_PT(P_low, Sat_T_P(P_low))
    hg = SteamSat_H_PT(P_low, Sat_T_P(P_low)) 
    vapor_quality_7 = (h7-hf)/(hg-hf)
    
    t8 = Sat_T_P(P_low)
    h8 = SteamSat_H_PT(P_low, t8) # sat vapor low pressure
    
    m8 = Qe/(h8-h7) #kg/s 
    m7 = m6 = m5 = m8
    t7 = t8 
    
    rt9 = rt__(P_low, C, D, E)
    t9 = t__(C_low, rt9, B, A)
    h9 = H_Xt(C_low, t9, A_t1, B_t1, C_t1)
    
    m14 = -m8/(1-(C_high/C_low))
    m13 = m12 = m14
    
    m9 = C_high/C_low*m14
    m10 = m11 = m9 
    
    h10 = (m9*h9+Qpump)/m10 
    
    # for h15 we assume a vapor quality of 0.005, after the expansion valve. hence 
    # h15 = h_LiBr_solution + h15_vapor (0.005)
    hf = WaterSat_H_PT(P_low, Sat_T_P(P_low))
    hg = SteamSat_H_PT(P_low, Sat_T_P(P_low))
    vapor_quality_14 = 0.005
    h14_steam_compo = vapor_quality_14*(hg-hf)+hf
    rt14 = rt__(P_low, C, D, E)
    t14 = t__(C_high, rt14, B, A)
    h14 = H_Xt(C_high, t14, A_t1, B_t1, C_t1) + h14_steam_compo 
    h13 = h14
    
    rt12 = rt__(P_high, C, D, E)
    t12 = t__(C_high, rt12, B, A)
    h12 = H_Xt(C_high, t12, A_t1, B_t1, C_t1)
    
    # Heat exhanger balance
    # Stream 1 energy balance 
    Q_he = m12*h12 - m13*h13
    # Stream 2 energy balance
    h11 = (m10*h10 + Q_he)/m11
    
    # Solving for stream 16
    # firstly working on the condensor, we know there is a difference of saturated vapor 
    # to saturated liquid, as stream 16 is already working in the superheated region. 
    
    Tsat = Sat_T_P(P_high)  # T for stream 16 always has to be greater than Tsat.
    h5 = H_vap_PT(P_high,  Tsat+2)
    Qc = m5*h5 - m6*h6 # correct
    Qg = m12*h12 + m5*h5 - m11*h11  # incorrect
    COP = Qe/Qg
    fitness = COP
    
    return fitness

num_generations = 500
num_parents_mating = 4

fitness_function = LiBr_cycle_fitness

sol_per_pop = 80
num_genes = 4

init_range_low = 0
init_range_high = 70

parent_selection_type = "sss"
keep_parents = 1

crossover_type = "single_point"

mutation_type = "random"
mutation_percent_genes = 10
def on_gen(ga_instance):
    print("Generation : ", ga_instance.generations_completed)
    print("Fitness of the best solution :", ga_instance.best_solution()[1])
    
ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_function,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       init_range_low=init_range_low,
                       init_range_high=init_range_high,
                       parent_selection_type=parent_selection_type,
                       keep_parents=keep_parents,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       mutation_percent_genes=mutation_percent_genes)

ga_instance.run()

ga_instance.plot_fitness()

solution_LiBr, solution_fitness, solution_idx_LiBr = ga_instance.best_solution()

optimal_P_low = round((( solution_LiBr[0] - 0) /(100 - 0)) * (4.8 - 0.676)+0.676,2)
optimal_P_high = round(((solution_LiBr[1] - 0) / (100 - 0)) * (10 - 4.8) + 4.8,2)
T_superheated = Sat_T_P(optimal_P_high)
optimal_C_low = round(solution_LiBr[2],2)
optimal_C_high = round(solution_LiBr[3],2)

print(f"Parameters of the best solution : Lower system pressure {optimal_P_low}kPa ;\
       Upper system pressure {optimal_P_high}kPa ; Lower LiBr concentration {optimal_C_low}% ; \
           Upper LiBr concentration {optimal_C_high}%")
print(f"Fitness value of the best solution = {solution_fitness}")
print(f"Index of the best solution : {solution_idx_LiBr}")

# Therefore to inputting all the optimal parameters to return temps and enthalpies at all point ##
#input solution here into a function

def Final_LiBr_values(optimal_P_low, optimal_P_high, optimal_C_low, optimal_C_high):
    
    LiBr_system = {}
    
    t6 = Sat_T_P(optimal_P_high) # sat T of liquid water
    #T = Sat_T_P(4.8)
    h6 = WaterSat_H_PT(optimal_P_high, t6)  # sat liquid water high pressure
    
    
    h7 = h6   # expansion value, H18 is at the low pressure state now.
    hf = WaterSat_H_PT(optimal_P_low, Sat_T_P(optimal_P_low))
    hg = SteamSat_H_PT(optimal_P_low, Sat_T_P(optimal_P_low)) 
    vapor_quality_7 = (h7-hf)/(hg-hf)
    
    t8 = Sat_T_P(optimal_P_low)
    h8 = SteamSat_H_PT(optimal_P_low, t8) # sat vapor low pressure
    
    m8 = Qe/(h8-h7) #kg/s 
    m7 = m6 = m5 = m8
    t7 = t8 
    
    rt9 = rt__(optimal_P_low, C, D, E)
    t9 = t__(optimal_C_low, rt9, B, A)
    h9 = H_Xt(optimal_C_low, t9, A_t1, B_t1, C_t1)
    
    m14 = -m8/(1-(optimal_C_high/optimal_C_low))
    m13 = m12 = m14
    
    m9 = optimal_C_high/optimal_C_low*m14
    m10 = m11 = m9 
    
    h10 = (m9*h9+Qpump)/m10 
    #t10 = t__(optimal_C_low, rt__(optimal_C_low, C, D, E), B, A)
    t10 = t9
    
    # for h15 we assume a vapor quality of 0.005, after the expansion valve. hence 
    # h15 = h_LiBr_solution + h15_vapor (0.005)
    hf = WaterSat_H_PT(optimal_P_low, Sat_T_P(optimal_P_low))
    hg = SteamSat_H_PT(optimal_P_low, Sat_T_P(optimal_P_low))
    vapor_quality_14 = 0.005
    h14_steam_compo = vapor_quality_14*(hg-hf)+hf
    rt14 = rt__(optimal_P_low, C, D, E)
    t14 = t__(optimal_C_high, rt14, B, A)
    h14 = H_Xt(optimal_C_high, t14, A_t1, B_t1, C_t1) + h14_steam_compo 
    
    h13 = h14
    #t14 = t__(optimal_C_high, rt__(optimal_C_high, C, D, E), B, A)
    # temp estimator for stream 16, lazily replicating a backwards calc.
    t13 = 20
    P13 = P__(rT__(optimal_C_high, t13, A, B), C, D, E)
    while abs(P13-optimal_P_low) < 0.5:
        t13 += 0.01
        P13 = P__(rT__(optimal_C_high, t13, A, B), C, D, E)
        
    
    
    rt12 = rt__(optimal_P_high, C, D, E)
    t12 = t__(optimal_C_high, rt12, B, A)
    h12 = H_Xt(optimal_C_high, t12, A_t1, B_t1, C_t1)
    
    # Heat exhanger balance
    # Stream 1 energy balance 
    Q_he = m12*h12 - m13*h13
    # Stream 2 energy balance
    h11 = (m10*h10 + Q_he)/m11
    t11 = t__(optimal_C_low, rt__(optimal_C_low, C, D, E), B, A)
    
    
    # Solving for stream 16
    # firstly working on the condensor, we know there is a difference of saturated vapor 
    # to saturated liquid, as stream 16 is already working in the superheated region. 
    
    Tsat = Sat_T_P(optimal_P_high)  # T for stream 16 always has to be greater than Tsat.
    h5 = H_vap_PT(optimal_P_high,  Tsat+2) # was originally 76
    Qc = m5*h5 - m6*h6 # correct
    Qg = m12*h12 + m5*h5 - m11*h11  # incorrect
    COP = Qe/Qg
    fitness = COP
    # C_COPs.append(COP)
    # Qgs.append(Qg)
    
    LiBr_system['Stream 5 - Temperature (Deg Cel)'] = Tsat+2
    LiBr_system['Stream 5 - Pressure (kPa)'] = optimal_P_high
    LiBr_system['Stream 5 - Enthalpy (kJ/kg)'] = h5
    LiBr_system['Stream 5 - Mass flow (kg/s)'] = m5
    
    LiBr_system['Stream 6 - Temperature (Deg Cel)'] = t6
    LiBr_system['Stream 6 - Pressure (kPa)'] = optimal_P_high
    LiBr_system['Stream 6 - Enthalpy (kJ/kg)'] = h6
    LiBr_system['Stream 6 - Mass flow (kg/s)'] = m6
    
    LiBr_system['Stream 7 - Temperature (Deg Cel)'] = t7
    LiBr_system['Stream 7 - Pressure (kPa)'] = optimal_P_low
    LiBr_system['Stream 7 - Enthalpy (kJ/kg)'] = h7
    LiBr_system['Stream 7 - Mass flow (kg/s)'] = m7
    
    LiBr_system['Stream 8 - Temperature (Deg Cel)'] = t8
    LiBr_system['Stream 8 - Pressure (kPa)'] = optimal_P_low
    LiBr_system['Stream 8 - Enthalpy (kJ/kg)'] = h8
    LiBr_system['Stream 8 - Mass flow (kg/s)'] = m8
    
    LiBr_system['Stream 9 - Temperature (Deg Cel)'] = t9
    LiBr_system['Stream 9 - Pressure (kPa)'] = optimal_P_low
    LiBr_system['Stream 9 - Enthalpy (kJ/kg)'] = h9
    LiBr_system['Stream 9 - LiBr concentration (%)'] = optimal_C_low
    LiBr_system['Stream 9 - Mass flow (kg/s)'] = m9
    
    LiBr_system['Stream 10 - Temperature (Deg Cel)'] = t10
    LiBr_system['Stream 10 - Pressure (kPa)'] = optimal_P_high
    LiBr_system['Stream 10 - Enthalpy (kJ/kg)'] = h10
    LiBr_system['Stream 10 - LiBr concentration (%)'] = optimal_C_low
    LiBr_system['Stream 10 - Mass flow (kg/s)'] = m10
    
    LiBr_system['Stream 11 - Temperature (Deg Cel)'] = t11
    LiBr_system['Stream 11 - Pressure (kPa)'] = optimal_P_high
    LiBr_system['Stream 11 - Enthalpy (kJ/kg)'] = h11
    LiBr_system['Stream 11 - LiBr concentration (%)'] = optimal_C_low
    LiBr_system['Stream 11 - Mass flow (kg/s)'] = m11
    
    LiBr_system['Stream 12 - Temperature (Deg Cel)'] = t12
    LiBr_system['Stream 12 - Pressure (kPa)'] = optimal_P_high
    LiBr_system['Stream 12 - Enthalpy (kJ/kg)'] = h12
    LiBr_system['Stream 12 - LiBr concentration (%)'] = optimal_C_high
    LiBr_system['Stream 12 - Mass flow (kg/s)'] = m12
    
    LiBr_system['Stream 13 - Temperature (Deg Cel)'] = t13
    LiBr_system['Stream 13 - Pressure (kPa)'] = optimal_P_high
    LiBr_system['Stream 13 - Enthalpy (kJ/kg)'] = h13
    LiBr_system['Stream 13 - LiBr concentration (%)'] = optimal_C_high
    LiBr_system['Stream 13 - Mass flow (kg/s)'] = m13
    
    LiBr_system['Stream 14 - Temperature (Deg Cel)'] = t14
    LiBr_system['Stream 14 - Pressure (kPa)'] = optimal_P_low
    LiBr_system['Stream 14 - Enthalpy (kJ/kg)'] = h14
    LiBr_system['Stream 14 - LiBr concentration (%)'] = optimal_C_high
    LiBr_system['Stream 14 - Mass flow (kg/s)'] = m14
    
    LiBr_system['Q generator (kW)'] = Qg
    
    
    x_coord1 = [h5, h6, h7, h8, h9, h10, h11]  # Added h1 to close the loop
    y_coord1 = [optimal_P_high, optimal_P_high, optimal_P_low, optimal_P_low, optimal_P_low, optimal_P_high, optimal_P_high] # Added optimal_P_low_solar to close the loop
    labels1 = ['5', '6', '7', '8', '9', '10', '11']
    
    
    x_coord2 = [h9, h10, h11, h12, h13, h14]
    y_coord2 = [optimal_P_low, optimal_P_high, optimal_P_high, optimal_P_high, optimal_P_high, optimal_P_low]
    labels2 = ['', '', '', '12', '13', '14']
    
    
    # Calculate saturation curves
    Sat_liquid_curve = []
    Sat_vapor_curve = []
    Plot_pressure = np.linspace(0.68, optimal_P_high*1.5, 100)
    
    for pressure in Plot_pressure:
        Sat_liquid_curve.append(WaterSat_H_PT(pressure, Sat_T_P(pressure)))
        Sat_vapor_curve.append(SteamSat_H_PT(pressure, Sat_T_P(pressure)))
    
    # Plot saturation curves
    plt.plot(Sat_vapor_curve, Plot_pressure, label='Saturated Vapor Curve')
    plt.plot(Sat_liquid_curve, Plot_pressure, label='Saturated Liquid Curve')
    
    # Plot coordinates with smaller marker size
    plt.plot(x_coord1, y_coord1, 'ro', markersize=4) # Adjust markersize as needed
    
    # Add labels to coordinates with slight offsets
    # offset = 0.3
    # for label, x, y in zip(labels1, x_coord1, y_coord1):
    #     plt.text(x , y + offset, label)
    plt.text(h9-100, optimal_P_low -0.2, '9')
    plt.text(h14-50, optimal_P_low -0.35, '14')
    plt.text(h7+15, optimal_P_low +0.25, '7')
    plt.text(h10-120, optimal_P_high -0.1 , '10')
    plt.text(h13-120, optimal_P_high +0.3, '13')
    plt.text(h11-10, optimal_P_high +0.3, '11')
    plt.text(h12+40, optimal_P_high +0.3, '12')
    plt.text(h6+20, optimal_P_high -0.3, '6')
    plt.text(h5, optimal_P_high +0.3, '5')
    plt.text(h8+20, optimal_P_low -0.15, '8')
    
    plt.plot(x_coord1, y_coord1, 'k-')
    
    # Plot coordinates with smaller marker size
    plt.plot(x_coord2, y_coord2, 'ro', markersize=4) # Adjust markersize as needed
    plt.plot([h8, h5], [optimal_P_low, optimal_P_high], 'k--')
    
    # Add labels to coordinates with slight offsets
    # offset = -0.3
    # for label, x, y in zip(labels2, x_coord2, y_coord2):
    #     plt.text(x , y + offset, label)
    
    plt.plot(x_coord2, y_coord2, 'k-')

            
    # Join coordinates with lines
    
    # Set plot labels and title
    plt.title('LiBr H2O cycle - PH diagram')
    plt.ylabel('Pressure (kPa)')
    plt.xlabel('Enthalpy (kJ/kg)')
    
    # Add legend
    plt.legend()
    
    # Show plot
    plt.show()
    
    
    
    #fitness = 1.0 /np.abs(fitness - desired_output)
    
    ## PH diagram
    #loop over the saturated t and P, get the h values in a seperate array
    #plot P vs H
    #plot each stream point on the PH diagram
    
    
    return LiBr_system

LiBr_system = Final_LiBr_values(optimal_P_low, optimal_P_high, optimal_C_low, optimal_C_high)

T_to_beat = max(LiBr_system['Stream 11 - Temperature (Deg Cel)'], LiBr_system['Stream 5 - Temperature (Deg Cel)'])

#%%
### Solar water cycle

def Solar_cycle_fitness(ga_instance_2, solution_solar, sol_ind_solar):
    P_low = (solution_solar[0]-0)/(10-0)*(4.8-0.68)+0.68
    if P_low<0.68 or P_low>2:
        return -np.inf
    P_high = (solution_solar[1]-0) /(10-0)*(9.5-4.9)+4.9
    if P_high >9.5 or P_high<4.9:
        return -np.inf

    
    mass_flow = (solution_solar[2]-0)/(10-0)*(0.1-0.001)+0.001
    if mass_flow >1 or mass_flow<0.0001:
        return -np.inf
    # mass_flow = solution_solar[2]
    # if mass_flow<0:
    #     return -np.inf
    
    m1 = m4 = m3 = m2 = mass_flow
    
    Qg = LiBr_system['Q generator (kW)'] #kW this value must be equal to the Qgenerator for the LiBr cycle.
    
    # ensure energy transfer from stream 2 into the generator of the LiBr cycle
    Ttarget = max(LiBr_system['Stream 12 - Temperature (Deg Cel)'], LiBr_system['Stream 16 - Temperature (Deg Cel)'])
    superheat_added = 5
    t2 = Ttarget+superheat_added
    h2 = H_vap_PT(P_high, t2)
    
    t1 = Sat_T_P(P_low) # saturated vapor
    h1 = SteamSat_H_PT(P_low, t1)
    
    print('h1 : ', h1)
    print('h2: ', h2)
    print('mass flow: ', m1)
    
    t3 = Sat_T_P(P_high)
    h3 = WaterSat_H_PT(P_high, t3)
    
    h4 = h3
    hf = WaterSat_H_PT(P_low, Sat_T_P(P_low))
    hg = SteamSat_H_PT(P_low, Sat_T_P(P_low)) 
    vapor_quality_4 = (h4-hf)/(hg-hf)
    
    print('h1: ', h1)
    print('h4: ', h4)
    print('mass flow: ', m1)
    
    Qsolar = m1*h1 - m4*h4
    print('Qsolar required: ', Qsolar)
    
    # condensor energy balance
    #h2 = (Qg +m3*h3)/m2
    # check if h2 is greater then the Tsat of the P_high, to ensure superheated vapor
    # is present after the comp. 
    h_sat_vapor = SteamSat_H_PT(P_high, Sat_T_P(P_high))
    if h2 <= h_sat_vapor:
        print('Stream 2 is NOT a superheated vapor, re-evaluate calculations')
        print('Current enthalpy: ', h2, 'Sat enthalpy vapor: ', h_sat_vapor)
        return -np.inf
    
    Q_comp = m2*h2 - m1*h1
    if Q_comp<0:
        print('Error compressor negative')
        return -np.inf
    print('Compressor input energy: ', Q_comp)
    
   # fitness = Q_comp/Qsolar # hence we are trying to make Qsolar small. 
    # TO DO: may need to add on some constraints to this fitness, e.g. mass flow rate, energy intake via compressor.
    fitness = 1/Qsolar
    
    return fitness

num_generations = 500
num_parents_mating = 4

fitness_function = Solar_cycle_fitness

sol_per_pop = 80
num_genes = 4

init_range_low = 0
init_range_high = 10

parent_selection_type = "sss"
keep_parents = 1

crossover_type = "single_point"

mutation_type = "random"
mutation_percent_genes = 10
def on_gen(ga_instance):
    print("Generation : ", ga_instance.generations_completed)
    print("Fitness of the best solution :", ga_instance.best_solution()[1])
    
ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_function,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       init_range_low=init_range_low,
                       init_range_high=init_range_high,
                       parent_selection_type=parent_selection_type,
                       keep_parents=keep_parents,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       mutation_percent_genes=mutation_percent_genes)

ga_instance.run()

ga_instance.plot_fitness()

solution_solar, solution_fitness_solar, solution_idx_solar = ga_instance.best_solution()

optimal_P_low_solar = round((( solution_solar[0] - 0) /(10 - 0)) * (4.8 - 0.68)+0.68,2)
optimal_P_high_solar = round(((solution_solar[1] - 0) / (10 - 0)) * (9.5 - 4.9) + 4.9,2)
T_superheated_solar = Sat_T_P(optimal_P_high)
optimal_mass_flow = round(((solution_solar[2]-0)/(10-0)*(0.1-0.001)+0.001),7)
#optimal_mass_flow = round(solution_solar[2],2)
print('---------- SOLAR CYCLE PARAMETERS ------------------------------------')
print(f"Parameters of the best solution : Lower system pressure {optimal_P_low_solar}kPa ;\
       Upper system pressure {optimal_P_high}kPa ; Mass Flow {optimal_mass_flow} kg/s")
print(f"Fitness value of the best solution = {solution_fitness_solar}")
print('----------------------------------------------------------------------')

def Final_Solar_values(optimal_P_low_solar, optimal_P_high_solar, T_superheated_solar, optimal_mass_flow):
    Solar_system = {}
    
    
    m1 = m4 = m3 = m2 = optimal_mass_flow
    
    Qg = LiBr_system['Q generator (kW)'] #kW this value must be equal to the Qgenerator for the LiBr cycle.
    
    # ensure energy transfer from stream 2 into the generator of the LiBr cycle
    Ttarget = max(LiBr_system['Stream 12 - Temperature (Deg Cel)'], LiBr_system['Stream 16 - Temperature (Deg Cel)'])
    superheat_added = 20
    t2 = Ttarget+superheat_added
    h2 = H_vap_PT(optimal_P_high_solar, t2)
    
    t1 = Sat_T_P(optimal_P_low_solar) # saturated vapor
    h1 = SteamSat_H_PT(optimal_P_low_solar, t1)
        
    t3 = Sat_T_P(optimal_P_high_solar)
    h3 = WaterSat_H_PT(optimal_P_high_solar, t3)
    
    h4 = h3
    t4 = t1
    hf = WaterSat_H_PT(optimal_P_low_solar, Sat_T_P(optimal_P_low_solar))
    hg = SteamSat_H_PT(optimal_P_low_solar, Sat_T_P(optimal_P_low_solar)) 
    vapor_quality_4 = (h4-hf)/(hg-hf)

    Qsolar = m1*h1 - m4*h4
    
    # condensor energy balance
    #h2 = (Qg +m3*h3)/m2
    # check if h2 is greater then the Tsat of the P_high, to ensure superheated vapor
    # is present after the comp. 
    h_sat_vapor = SteamSat_H_PT(P_high, Sat_T_P(P_high))
    if h2 <= h_sat_vapor:
        print('Stream 2 is NOT a superheated vapor, re-evaluate calculations')
        print('Current enthalpy: ', h2, 'Sat enthalpy vapor: ', h_sat_vapor)
        return -np.inf
    
    Q_comp = m2*h2 - m1*h1
    if Q_comp<0:
        print('Error compressor negative')
        return -np.inf
    
    Solar_system['Stream 1 - Temperature (Deg Cel)'] = t1
    Solar_system['Stream 1 - Pressure (kPa)'] = optimal_P_low_solar
    Solar_system['Stream 1 - Enthalpy (kJ/kg)'] = h1
    
    Solar_system['Stream 2 - Temperature (Deg Cel)'] = t2
    Solar_system['Stream 2 - Pressure (kPa)'] = optimal_P_high_solar
    Solar_system['Stream 2 - Enthalpy (kJ/kg)'] = h2
    
    Solar_system['Stream 3 - Temperature (Deg Cel)'] = t3
    Solar_system['Stream 3 - Pressure (kPa)'] = optimal_P_high_solar
    Solar_system['Stream 3 - Enthalpy (kJ/kg)'] = h3
    
    Solar_system['Stream 4 - Temperature (Deg Cel)'] = t4
    Solar_system['Stream 4 - Pressure (kPa)'] = optimal_P_low_solar
    Solar_system['Stream 4 - Enthalpy (kJ/kg)'] = h4
    
    Solar_system['Q solar (kW)'] = Qsolar
    
    return Solar_system

Solar_system = Final_Solar_values(optimal_P_low_solar, optimal_P_high_solar, T_superheated_solar, optimal_mass_flow)


