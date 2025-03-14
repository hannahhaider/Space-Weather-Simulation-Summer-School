#!/usr/bin/env python

import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from helper_functions import *
from physics_functions import *

#-----------------------------------------------------------------------------
# some constants we may need
#-----------------------------------------------------------------------------

mass_o = 16.0 # in AMU
mass_o2 = 32.0 # in AMU
mass_n2 = 28.0 # in AMU

#-----------------------------------------------------------------------------
# Initialize temperature as a function of altitude
#-----------------------------------------------------------------------------

def init_temp(alt_in_km):
    temp_in_k = 200 + 600 * np.tanh( (alt_in_km - 100) / 100.0)
    return temp_in_k

#-----------------------------------------------------------------------------
# Update the old state (temp, densities), dTdt, and dt:
#-----------------------------------------------------------------------------

def update_state(old_state, dTdt, dt):

    new_state = {}

    return new_state

# ----------------------------------------------------------------------
# Main Code
# ----------------------------------------------------------------------

if __name__ == '__main__':  # main code block

    # set f107:
    f107 = 100.0
    f107a = 100.0 #average over 81 days or 3 solar rotations (1 = 27 days)

    SZA = 0.0
    efficiency = 0.3
    
    # boundary conditions for densities: hydrostatic atmosphere, densities at some altitude can be predicted
    n_o_bc = 5.0e17 # /m3
    n_o2_bc = 4.0e18 # /m3
    n_n2_bc = 1.7e19 # /m3

    # read in EUV file (step 1):
    euv_file = 'euv_37.csv' #takes in F10.7 and gives intensity, (the linear coefficients and absorption cross sections)
    euv_info = read_euv_csv_file(euv_file)
    print('short : ', euv_info['short'])
    print('ocross : ', euv_info['ocross'])

    # initialize altitudes (step 2):
    nAlts = 41
    alts = np.linspace(100, 500, num = nAlts)
    print('alts : ', alts)

    # initialize temperature (step 3):
    temp = init_temp(alts)
    print('temp : ', temp)
    
    # compute scale height in km (step 4):
    h_o = calc_scale_height(mass_o, alts, temp)
    h_o2 = calc_scale_height(mass_o2, alts, temp)
    h_n2 = calc_scale_height(mass_n2, alts, temp)
    print('Scale height of o : ', h_o)

    # calculate euvac (step 5):
    intensity_at_inf = EUVAC(euv_info['f74113'], euv_info['afac'], f107, f107a) #intensity as a function of wavelength 

    # calculate mean wavelength:
    wavelength = (euv_info['short'] + euv_info['long'])/2

    # calculate energies (step 8):
    energies = convert_wavelength_to_joules(wavelength)
    
    # plot intensities at infinity:
    plot_spectrum(wavelength, intensity_at_inf, 'intensity_inf.png')

    # Calculate the density of O as a function of alt and temp (step 6):
    density_o = calc_hydrostatic(n_o_bc, h_o, temp, alts)
    density_o2 = calc_hydrostatic(n_o2_bc, h_o2 ,temp, alts)
    density_n2 = calc_hydrostatic(n_n2_bc, h_n2, temp, alts)
    # Need to calculate the densities of N2 and O2...
    
    # plot out to a file:
    plot_value_vs_alt(alts, density_o, 'o_init.png', '[O] (/m3)', is_log = True)

    # Calculate Taus for O (Step 7): how much light each species is absorbing
    tau_o = calc_tau(SZA, density_o, h_o, euv_info['ocross']) #create a for loop for all wavelengths
    tau_n2 = calc_tau(SZA, density_n2, h_n2, euv_info['n2cross'])
    tau_o2 = calc_tau(SZA, density_o2, h_o2, euv_info['o2cross'])
    # Need to calculate tau for N2 and O2, and add together...
    # and do this for all of the wavelengths...
    tau = tau_o + tau_n2 + tau_o2 # + ...
    
    # plot Tau to file: #also create a loop for all wavelengths

    
    plot_value_vs_alt(alts, tau_o[8], 'tau_o.png', 'tau ()')
    plot_value_vs_alt(alts, tau_o2[8], 'tau_o2.png', 'tau()')
    plot_value_vs_alt(alts, tau_n2[8], 'tau_n2.png', 'tau()')
    Qeuv_o = calculate_Qeuv(density_o,
                            intensity_at_inf,
                            tau,
                            euv_info['ocross'],
                            energies,
                            efficiency)

    Qeuv_o2 = calculate_Qeuv(density_o2,
                            intensity_at_inf,
                            tau,
                            euv_info['o2cross'],
                            energies,
                            efficiency)
    
    Qeuv_n2 = calculate_Qeuv(density_n2,
                            intensity_at_inf,
                            tau,
                            euv_info['n2cross'],
                            energies,
                            efficiency)
    
    
    Qeuv = Qeuv_o + Qeuv_o2 + Qeuv_n2 # + ...

    # plot out to a file:
    plot_value_vs_alt(alts, Qeuv_o, 'o_qeuv.png', 'Qeuv - O (W/m3)')
    plot_value_vs_alt(alts, Qeuv_n2, 'n2_qeuv.png', 'Qeuv - N2 (W/m3)')
    plot_value_vs_alt(alts, Qeuv_o2, 'o2_qeuv.png', 'Qeuv - O2 (W/m3)')
    
    # alter this to calculate the real mass density (include N2 and O2):
    
    rho = calc_rho(density_o, mass_o, density_o2, density_n2, mass_o2, mass_n2)
    
    # this provides cp, which could be made to be a function of density ratios:
    cp = calculate_cp()
    
    dTdt = convert_Q_to_dTdt(Qeuv, rho, cp)

    # plot out to a file:
    plot_value_vs_alt(alts, dTdt * 86400, 'dTdt.png', 'dT/dt (K/day)')
    
