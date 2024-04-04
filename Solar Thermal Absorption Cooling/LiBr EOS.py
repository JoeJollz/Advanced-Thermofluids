# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 15:21:59 2024

@author: jrjol
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


def X__(t, rt, B, A):
    coeff = np.zeros(4)
    coeff[0] = B[0]+rt*A[0]-t
    coeff[1] = B[1] + rt * A[1] 
    coeff[2] = B[2] +rt * A[2]
    coeff[3] = B[3] +rt * A[3]
    
    f = lambda x: coeff[3]*x**3+coeff[2]*x**2 + coeff[1]*x+coeff[0]
    s =fsolve(f, [100, 50, 0])
    
    return [num for num in s if 40 <= num <= 70][0]
    
    #return s
    # x = var('x')
    # sol = solve(Eq(coeff[3]*x**3 + coeff[2] * x **2 + coeff[1] * x + coeff[0], 0), x)
    # #print(sol)
    # # X = None
    # # for num in sol:
    # #     # Check if the number is real and within the range
    # #     print(num)
    # #     if isinstance(num, (int, float)) and 45 < num <= 70:
    # #         X= num
    # #         print('satisfied')
    # # print('retuning X: ', X)
    # print(sol)
    # # X = [n for n in sol if n.is_real][0]
    
    # # If all numbers are real, check if they are between 40 and 70
    # # if all(isinstance(num, float) for num in sol):
    # #     X = [num for num in sol if 40 <= num <= 70]
    
    # # # If all numbers are imaginary, extract the real components and check if they are between 40 and 70
    # # elif all(isinstance(num, complex) for num in sol):
    # #     X = [num.real for num in sol if 40 <= num.real <= 70]
    
    # # # If there is a mix of real and imaginary numbers, extract the real numbers and check if they are between 40 and 70
    # # else:
    # #     X = [num.real for num in sol if isinstance(num, (float, int)) and 40 <= num <= 70]
    # real_numbers = []

    # # Extract real components if the numbers are complex
    # numbers = [num.real if isinstance(num, complex) else num for num in sol]
    # print(numbers[1])
    
    # # Check for real numbers between 40 and 70
    # real_numbers = [num for num in sol if isinstance(num, (float, int)) and 40 <= num <= 70]
    # #return real_numbers
    # #print(X, 'X is here')
    # return numbers[1].real

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
def fitness_func(ga_instance, solution, solution_idx):
    P_low = (( solution[0] - 0) /(100 - 0)) * (4.8 - 0.676)+0.676
    if P_low <0.676:
        return -np.inf
    P_high = ((solution[1] - 0) / (100 - 0)) * (10 - 4.8) + 4.8
   # rescaled_number = ((original_number - original_min) / (original_max - original_min)) * (target_max - target_min) + target_min
    if  P_high< (P_low+0.5) or P_high> 10 or P_high<4.8:
        return -np.inf
    C_low = solution[2]
    if C_low < 40:
        return -np.inf
    C_high = solution[3]
    if C_high<C_low or C_high > 70:
        return -np.inf
    
    
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
    fitness = COP
    # C_COPs.append(COP)
    # Qgs.append(Qg)
    
    #fitness = 1.0 /np.abs(fitness - desired_output)
    
    return fitness

num_generations = 500
num_parents_mating = 4

fitness_function = fitness_func

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

solution, solution_fitness, solution_idx = ga_instance.best_solution()

optimal_P_low = round((( solution[0] - 0) /(100 - 0)) * (4.8 - 0.676)+0.676,2)
optimal_P_high = round(((solution[1] - 0) / (100 - 0)) * (10 - 4.8) + 4.8,2)
optimal_C_low = round(solution[2],2)
optimal_C_high = round(solution[3],2)

print(f"Parameters of the best solution : Lower system pressure {optimal_P_low}kPa ;\
       Upper system pressure {optimal_P_high}kPa ; Lower LiBr concentration {optimal_C_low}% ; \
           Upper LiBr concentration {optimal_C_high}%")
print(f"Fitness value of the best solution = {solution_fitness}")
print(f"Index of the best solution : {solution_idx}")

