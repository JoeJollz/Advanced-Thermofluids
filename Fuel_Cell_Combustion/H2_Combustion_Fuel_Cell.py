"""

jrj44 

Partial reactions of hydrogen combustion via a Fuel Cell. Extracting work via 
the transfer of electrons from anode to cathod. 

At the anode the H2 becomes oxidised, forming 2H+, the H+ ions pass through 
a polymer membrane, hence the name Polymer Exchange Membrane (PEM) fuel cell.
The electrons do not transfer through this membrane (ideally), and instead 
travel through a circuit. 

A PD is created between the anode and the cathode, and this can be related to
the partial pressures of oxygen at the anode and cathode, the difference 
in Po2(anode) and Po2(cathode) result in a PD to be created. 


"""
P_P0=1
T = 1000
Ec = 3
Eh = 8
Eo = 10
En = 18.8
# [CH4, CO, CO2, H2, H20]
#C3H8 = 0
CO = 0
CO2 = 1
H2 = 2
H20 = 3
O2 = 4
N2 = 5
from numpy import *
import pandas as pd

def LnKp(): 
    LnKp1 = 23.528
    LnKp2 = 23.162
    return [LnKp1,LnKp2]


def thingtosolve(N_):
    N=N_**6
    lnkp1,lnkp2 = LnKp()
    f = zeros(6)
    #carbon
    f[0]= Ec - (1*N[CO]+1*N[CO2])
    #hydrogon
    f[1]= 2*N[H2] +2*N[H20] - Eh
    
    #oxygen
    f[2]= 1*N[CO] + 2*(N[O2]) + 2*N[CO2]+1*N[H20] -Eo
    
    #nitrogen
    f[3] = 2*N[N2] - En
    
    Nt = sum(N)
    
    f[4] = log(N[CO2])-1/2*log(N[O2])-log(N[CO])-1/2*log(P_P0*1/Nt)-lnkp1
    f[5] = log(N[H20])-1*log(N[H2])-1/2*log(N[O2])-1/2*log(P_P0*1/Nt)-lnkp2
    
            
    return f

from scipy.optimize import fsolve,root

initial_guess =array([0.6,0.4,1,1,1,1]) # sqrt of variable

ans=fsolve(thingtosolve,initial_guess, maxfev=100000, xtol=1e-10)

print('-------------')
print('check fsolve: ',ans**6)
#print('compare',ans_broyden**6)
print('Residuals: ' , thingtosolve(ans))

Partial_P_O2 = ans[-2]**6/sum(ans**6)
#Partial_P_O2 = Partial_P_O2**2
print('Partial Pressure O2 (anode) ', Partial_P_O2)

nernest_pot = 8.31*1000/(4*96485)*log(0.21/Partial_P_O2)
print('The Nernest Potenetial is: ', nernest_pot)