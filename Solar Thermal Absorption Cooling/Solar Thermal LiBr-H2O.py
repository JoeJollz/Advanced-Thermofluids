'''
This code includes 2 key sections, the first on Solar irradiance estimate 
(part 1 of the coursework).And the thermo cycles for LiBr-H2O cooling with 
a vapor compression cycle. 

'''

#########################################################################
              ### PART 1 - SOLAR IRRADIANCE ESTIMATE ###
#########################################################################
import numpy as np
import matplotlib.pyplot as plt

Isc = 1367 # W/m2 Solar constant

Months = ['January', 'Feburary', 'March', 'April', 'May', 'June', 'July', \
          'August', 'September', 'October', 'November', 'December']
Days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

# Defining location for Marrakesh Morocco
phi = np.radians(31.63) # latitude
longitude = -8.078 # longitude

# Hottel's Model. PowerFromTheSun Section 2.13
A=0.468 # Elevation of Marrakesh
a0 = 0.4237 - 0.00821*(6-A)**2
a1 = 0.5055 +0.00595*(6.5-A)**2
k = 0.2711 + 0.01858*(2.5-A)**2

def daily_irrad_calc(Day, longitude, phi):
    '''
    
    Parameters
    ----------
    Day : Float
        Takes in the day number.
    longitude : Float
        Takes in longitude as degrees.
    phi : Float
        Takes in the latitude as radians.

    Returns
    -------
    daily_irradiance_Ith : List
        Global Total Solar Irradiance at each iteration.
    daily_irradiance_Idh : List
        Diffuse Solar Irradiance at each iteration.
    daily_irradiance_Ibn : List
        Direct normal solar irradiance stored at each iteration.
    counter_out_of_hours : Float
        Counts the number of iterations for when the |Zenith|>90. This is 
        useful when calculating the Average Irradiance during Sunlight hours.

    '''
    daily_irradiance_Ith = []
    daily_irradiance_Ibn = []
    daily_irradiance_Idh = []
    
    counter_out_of_hours = 0
    
    for std_time in np.linspace(0,24,60): # Looping throught the 24 hours of a day.

        UTC= 1
        LSTM = 15*UTC # degrees
        dec = np.radians(23.45*np.sin(np.radians(360*(284+Day)/365)))  # declination angle Power From The Sun 3.7
        lat = phi
        Lloc = longitude #longitude. Powerfrom the sun: LCT == stand time. 
        
        B = np.radians((Day-1)*360/365)  # Outputs radians. Equation 3.3. Power From The Sun
        EoT = 229.2*(0.000075+0.001868*np.cos(B)-0.032077*np.sin(B)-0.014615*np.cos(2*B)-0.04089*np.sin(2*B)) #equation of time # 3.2 Power From The Sun
        solar_time = (4*(LSTM - Lloc)+EoT)/60+std_time  # currently in mintues, convert to hours # equation 3.1. aka 'ts' for solar time
        w = np.radians(15*(solar_time-12))  # takes hours, and outputs degrees ## equation 3.1 power from the sun

        zenith = np.arccos(np.cos(dec)*np.cos(w)*np.cos(lat)+np.sin(dec)*(np.sin(lat))) # Hottel's model, eq 3.17 power from the sun
        Idh = Io*np.cos(zenith)*(0.2710-0.2939*(a0+a1*np.exp(-k*1/np.cos(zenith)))) # Hottel's model eq 2.15 
        Ibn = Io*(a0+a1*(np.exp(-k/np.cos(zenith)))) # Hottel's model eq 2.12 power from the sun
        Ith = Ibn*np.cos(zenith)+Idh # Global total solar irradiance eq 2.9 power from the sun
        if abs(np.degrees(zenith))>90: # if TRUE, the sun is no longer above the horizon.
            Ith = 0
            Ibn = 0
            Idh = 0
            counter_out_of_hours +=1 # Counter for the sun being below the hoirzon.
        daily_irradiance_Ith.append(Ith)
        daily_irradiance_Ibn.append(Ibn)
        daily_irradiance_Idh.append(Idh)
        
    return daily_irradiance_Ith, daily_irradiance_Idh, daily_irradiance_Ibn, counter_out_of_hours

# Calculate the summer solstice
daily_irradiance_Ith, daily_irradiance_Idh, daily_irradiance_Ibn, counter_out_of_hours = daily_irrad_calc(172, longitude, phi)

avg_check = sum(daily_irradiance_Ith)/60.0
avg_daylight = sum(daily_irradiance_Ith)/(60-counter_out_of_hours)
print(avg_check)
x= np.linspace(0,24,60)
plt.plot(x, daily_irradiance_Ith, marker='', linestyle='-', label = 'Ith')
plt.plot(x, daily_irradiance_Ibn, marker='', linestyle='-', label = 'Ibn')
plt.plot(x, daily_irradiance_Idh, marker='', linestyle='-', label = 'Idh')
plt.axhline(y=avg_check, color='r', linestyle='--', label='Average whole day')
plt.axhline(y=avg_daylight, color='r', linestyle='--', label='Average during daylight')
plt.text(x[-27]-0.5, daily_irradiance_Ith[-27]+20, 'Ith', ha='right')
plt.text(x[-21]+1.2, daily_irradiance_Ibn[-21]+30, 'Ibn', ha='right')
plt.text(x[-29], daily_irradiance_Idh[-29]+20, 'Idh', ha='right')
plt.text(x[-23]-1, avg_check+30, 'Average whole day', ha='right')
plt.text(x[-26], avg_daylight+30, 'Average during daylight', ha='right')
plt.title('Daily Irradiance (June 21st)')
plt.xlabel('Time of day (hour)')
plt.ylabel('Irradiance (W/m2)')
plt.grid(False)
plt.show()


#plotting for the winter solstice
daily_irradiance_Ith, daily_irradiance_Idh, daily_irradiance_Ibn, counter_out_of_hours = daily_irrad_calc(355, longitude, phi)

avg_check = sum(daily_irradiance_Ith)/60.0
avg_daylight = sum(daily_irradiance_Ith)/(60-counter_out_of_hours)
print(avg_check)
x= np.linspace(0,24,60)
plt.plot(x, daily_irradiance_Ith, marker='', linestyle='-', label = 'Ith')
plt.plot(x, daily_irradiance_Ibn, marker='', linestyle='-', label = 'Ibn')
plt.plot(x, daily_irradiance_Idh, marker='', linestyle='-', label = 'Idh')
plt.axhline(y=avg_check, color='r', linestyle='--', label='Average whole day')
plt.axhline(y=avg_daylight, color='r', linestyle='--', label='Average during daylight')
plt.text(x[-31], daily_irradiance_Ith[-33]+20, 'Ith', ha='right')
plt.text(x[-30], daily_irradiance_Ibn[-30]-30, 'Ibn', ha='right')
plt.text(x[-29], daily_irradiance_Idh[-29]-30, 'Idh', ha='right')
plt.text(x[-23]-1, avg_check+25, 'Average whole day', ha='right')
plt.text(x[-26], avg_daylight+25, 'Average during daylight', ha='right')
plt.title('Daily Irradiance (December 21st)')
plt.xlabel('Time of day (hour)')
plt.ylabel('Irradiance (W/m2)')
plt.grid(False)
plt.show()

## Monthly Irradiance Calculation ##
# Simply looping through the days of each month, then storing the relevant information, 
# and then plotting this information. 
daily_irradiance = []
Monthly_irradiance = []
Day = 0
for i, month in enumerate(Months):
    Days_in_month = Days[i]
    monthly = 0
    for j in range(Days_in_month):
        Day+=1
        print(Day)
        daily_irradiance_Ith, daily_irradiance_Idh, daily_irradiance_Ibn, counter_out_of_hours = daily_irrad_calc(Day, longitude, phi)
        print(counter_out_of_hours)
        daily_irradiance.append(sum(daily_irradiance_Ith)/(60-counter_out_of_hours))
        monthly += sum(daily_irradiance_Ith)/(60-counter_out_of_hours)
    Monthly_irradiance.append(monthly/Days_in_month)
    
x = range(1,366)
plt.plot(x, daily_irradiance)
plt.ylim(0, 800)
plt.title('Average Daily Irradiance (during daylight hours) for a year')
plt.xlabel('Day of the year')
plt.ylabel('Average daily Irradiance (W/m2)')
plt.show()

x = Months
plt.plot(x, Monthly_irradiance)
plt.xticks(rotation=45, ha='right')
# Title and labels
plt.title('Monthly Average Irradiance')
plt.xlabel('Month')
plt.ylabel('Average Monthly Irradiance (W/m2)')
plt.ylim(0, 800)
plt.show()
print('--------------------------------------------------------------------------------------------')
print(f'The average solar irradiance (W/m2) for July, during daylight hours, is: {round(Monthly_irradiance[6],2)}')
print('--------------------------------------------------------------------------------------------')


#########################################################################
              ### PART 2.1 - LiBr-H20 cycle design ###
#########################################################################
'''
All of these functions in this code, from the IAPWS have been tested for validity on 
the Water Tables.
'''
#%%
## IAPWS 2007 - Section 5.1
def WaterSat_H_PT(P,T):  
    '''
    Calculates the enthalpy of saturdated liquid water from known Psat and Tsat. 

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

#%%
# IAPWS 2007 - Section 6.1
def SteamSat_H_PT(P, T): 
    '''
    Calculates the Enthalpy of saturated vapor from P and T. 

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

#%%
# IAPWS - Section 6.2
# Metastable-Vapor Region- super heated steam values
def H_vap_PT(P, T): 
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

#%%
# IAPWS 2007 - Section 8.2
def Sat_T_P(P):
    '''
    Calculates the corresponding saturated temperature, for a given saturated pressure. 

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
#%%
# IAPWS - Section 6.3.1 - Backwards calcuation for T(p,H)
def Superheated_T_pH(P, H):
    '''
    Calculates the temperture (deg C) of superheated vapor from pressure and enthalpy
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

#########################################################################
              ### PART 2.2 - Vapor Compression cycle design ###
#########################################################################


# %%
