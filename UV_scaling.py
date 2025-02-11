import numpy as np
import constants as const
import pandas as pd


#NUV/FUV band scaling, from GALEX vs log RHK relation, Findeisen+(2011)

#color-mass relation
magnitudes = pd.read_csv('color_magnitudes.csv', index_col=0, sep=',')

def FUV(logRhk, B, V):

    """
    Parameters
    ------------------------
    B : apparent B magnitude
    V : apparent V magnitude

    Returns
    ------------------------
    FUV : GALEX magnitude in FUV
    """

    FUV = 12.30 - 3.95*(logRhk + 4.5) \
    -2.94*(logRhk+4.5)**2 + 2.22*(B-V - 0.8) \
    -9.11*(B-V - 0.8)*(logRhk + 4.5) \
    -12.0*(B-V - 0.8)*(logRhk+ 4.5)**2 \
    -17.6*(B-V - 0.8)**2 - 3.9*(B-V - 0.8)**2 * (logRhk + 4.5) \
    -24.3*(B-V - 0.8)**2*(logRhk + 4.5)**2 + V 

    return FUV

def NUV(logRhk, B, V):
    """

    Parameters
    ------------------------
    B : apparent B magnitude
    V : apparent V magnitude

    Returns
    ------------------------
    NUV : GALEX magnitude in NUV
    """

    NUV = 6.19 - 0.87*(logRhk + 4.5) \
    - 0.93*(logRhk + 4.5)**2 \
    + 6.54*(B-V - 0.8) - 2.73*(B-V - 0.8)*(logRhk + 4.5) \
    - 2.89 * (B-V - 0.8) * (logRhk + 4.5)**2 + V

    return NUV
    
def GALmag_to_flux(UV, wav):
    """
    Turns GALEX magntiude to absolute flux

    Parameters
    --------------------------------
    UV: GALEX magntiude in UV band
    wav: wavelength range of GALEX UV Band

    Returns
    ---------------------------------
    flux: absolute flux in erg / cm^2 / s
    """
    
    nu = const.c / wav
    flux = 10**(-0.4 * (UV + 48.60)) * (nu[0] - nu[1])
    return flux

def apparent_mag(absolute_mag, d):

    m = 5 * np.log10(d / const.pc) - 5 + absolute_mag

    return m

def get_flux_uv(logRhk, d, Ms):

    MB = np.interp(Ms, magnitudes['Ms'].values[::-1], magnitudes['M_B'].values[::-1])
    MV = np.interp(Ms, magnitudes['Ms'].values[::-1], magnitudes['M_V'].values[::-1])
    mB = apparent_mag(MB, d)
    mV = apparent_mag(MV, d)

    flux_nuv = GALmag_to_flux(NUV(logRhk, mB, mV), np.array([1780e-8, 2830e-8]))
    flux_fuv = GALmag_to_flux(FUV(logRhk, mB, mV), np.array([1350e-8, 1780e-8]))
    flux = np.array([flux_nuv, flux_fuv])

    return flux


#EUV band scaling, from King+ (2018?)

def get_Leuv(Lx, Rs):
    
    Fx = Lx / (4 * np.pi * Rs**2)
    Feuv = 460 * Fx**(1 - 0.425)
    Leuv = 4 * np.pi * Feuv * Rs**2

    return Leuv 