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
## IAPWS 2007
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


#########################################################################
              ### PART 2.2 - Vapor Compression cycle design ###
#########################################################################

