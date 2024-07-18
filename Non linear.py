P_P0=1
T = 1173
Ec = 1
Eh = 12
Eo = 4
# [CH4, CO, CO2, H2, H20]
CH4 = 0
CO =1
CO2 =2
H2=3
H2O=4
from numpy import *

def LnKp(T): 
    LnKp1 = -4.30e-5*T**2 +1.07e-1*T-60.1
    LnKp2 = -1.02e-5*T**2+2.42e-2*T-14.2
    return [LnKp1,LnKp2]

def thingtosolve(N_):
    N=N_**2
    lnkp1,lnkp2 = LnKp(T)
    f = zeros(5)
    #carbon
    f[0]= Ec - (N[CH4]+N[CO]+N[CO2])
    #hydrogon
    f[1]= 4*N[CH4]+2*N[H2] +2*N[H2O] - Eh
    
    #oxygen
    f[2]= N[CO] + 2*N[CO2]+N[H2O] -Eo
    
    Nt = sum(N)
    
    f[3] = 4*log(N[H2])+log(N[CO2])-2*log(N[H2O])-log(N[CH4])+2*log(P_P0*1/Nt)-lnkp1
    f[4] = N[H2O]*N[CO]/(N[CO2]*N[H2]) - exp(lnkp2)
    
            
    

    return f

from scipy.optimize import fsolve,root

initial_guess =array([1,1,1,1,1])
ans=fsolve(thingtosolve,initial_guess)
#ans =root(thingtosolve,initial_guess,method='lm')
print(ans)
print(thingtosolve(ans))