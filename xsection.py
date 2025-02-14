import os
os.environ['XUVTOP']
import ChiantiPy
import ChiantiPy.core as ch
import ChiantiPy.tools.filters as chfilters
import ChiantiPy.tools.io as chio
import numpy as np
import scipy.integrate as sp_int
import scipy.interpolate as sp_intpl
import scipy.special as sp_special
import pandas as pd

#constants
c = 2.9979e10
mp = 1.67e-24
k_b = 1.3807e-16
m_e = 9.11e-28
e = 4.803e-10
G = 6.67e-8

Zelem_lc = {'h':1, 'he':2, 'li':3, 'be':4, 'b':5, 'c':6, 'n':7,
              'o':8, 'f':9,'ne':10, 'na':11, 'mg':12, 'al':13,
              'si':14, 's':16, 'ar':18, 'ca':20, 'fe':26}

Aelem_lc = {'h':1,'he': 4,'li': 6,'be': 8,'b': 10,'c': 12,'n': 14,
              'o': 16,'f': 18,'ne': 20,'na': 22,'mg': 24,'al': 26,
              'si': 28,'s': 32,'ar': 36,'ca': 40,'fe': 52}




def doppler_shift(w0, velocity):
    return (1 + velocity / c)*w0

def voigt_profile(w, w0, gauss_sigma, lorentz_HWHM, out=None, where=True):
    """Return

    Parameters
    --------------------
    w:

    w0:

    gauss_sigma: standard variation of

    lorentz_HWHM:

    """

    return sp_special.voigt_profile(w-w0, gauss_sigma, lorentz_HWHM, out=out, where=where)


def voigt_xsection(w, w0, f, Gamma, T, mmw, out=None, where=True):
    """
    Compute the absoprtion cross section using the voigt profile for a line

    Parameters
    ------------------------
    w:

    w0: Line center wavelength (may be doppler shifted)

    f:

    Gamma:

    T:

    mass:

    Returns
    --------------------------
    """

    lorentz_HWHM = Gamma / (4*np.pi)
    #fixed from last commit
    gauss_sigma = np.sqrt(k_b*T/(mmw * c**2))*w0
    xsection = np.pi * e**2 / (m_e * c) * f * voigt_profile(w, w0, gauss_sigma, lorentz_HWHM, out=out, where=where)
    return xsection


def He_triplet_xsection(w, absorber_v, T, j):
    He_wavj0 = 1.082909e-4
    He_wavj1 = 1.083025e-4
    He_wavj2 = 1.083033e-4 #cm

    if j == 0:
        absorber_w0 = doppler_shift(c / He_wavj0, absorber_v) #in the frame of the object emitting light
        f = 5.9902e-02
        Gamma = 1.0216e+07
    elif j == 1:
        absorber_w0 = doppler_shift(c / He_wavj1, absorber_v)
        f = 1.7974e-01
        Gamma = 1.0216e+07
    elif j == 2:
        absorber_w0 = doppler_shift(c / He_wavj2, absorber_v)
        f = 2.9958e-01
        Gamma = 1.0216e+07

    xsection = voigt_xsection(w, absorber_w0, f, Gamma, T, 4*mp)
    return xsection

def He_triplet_xsection_combined(w, absorber_v, T):

    return He_triplet_xsection(w, absorber_v, T, 0) + He_triplet_xsection(w, absorber_v, T, 1) + He_triplet_xsection(w, absorber_v, T, 2)


def He_triplet_xsection_atlc(T):
    return He_triplet_xsection(c /  1.083033e-4, 0, T, 2)


class xsection():
    #class to extract cross sections from Chianti
    def __init__(self, species, species_base):
        self.species = species
        self.species_base = species_base

    def getZ(self):
        return Zelem_lc[self.species_base]
    
    def getA(self):
        return Aelem_lc[self.species_base]
    
    def get_xs_species(self, w, absorber_v, T):
        all_lines = pd.read_csv('escape_lines.csv')
        lines = all_lines[all_lines['species']==self.species]
        xs = np.zeros(np.shape(absorber_v)[:-1] + np.shape(w))

        for ix, row in lines.iterrows():
            w0 = doppler_shift(c / row['wavelength'], absorber_v)
            xs += voigt_xsection(w, w0, row['f'], row['A'], T, self.getA()*mp, out=out, where=where)
            
        return xs  
    
    def get_xs_lines(self, line_names, w, absorber_v, T, out=None, where=True):
        all_lines = pd.read_csv('escape_lines.csv')
        lines = all_lines[all_lines['name'].isin(line_names)]
        xs = np.zeros(np.shape(absorber_v)[:-1] + np.shape(w))

        for ix, row in lines.iterrows():
            w0 = doppler_shift(c / row['wavelength'], absorber_v)
            xs += voigt_xsection(w, w0, row['f'], row['A'], T, self.getA()*mp, out=out, where=where)
            
        return xs

    def topN_xs(self, T, N = 10):
        #based off top N f values so may be not be exactly true
        ion = ch.ion(self.species, temperature = np.float64(T), eDensity = 1.0)
        lv1 = np.array(ion.Wgfa['lvl1'])
        lv2 = np.array(ion.Wgfa['lvl2'])
        wvl = np.array(ion.Wgfa['wvl'])
        gf = np.array(ion.Wgfa['gf'])
        A = np.array(ion.Wgfa['avalue'])
        mult=np.array(ion.Elvlc['mult'])

        mask = (lv1 == 1) & (wvl != 0.0)
        ind = np.argsort(gf[mask])[::-1]
        wvl1 = np.abs(wvl[mask][ind]) * 1e-8 #change wavelength from angstom to cgs
        w  = c / wvl1   
        gf1 = gf[mask][ind] #gf is number of degenerate quantum states x oscillator strength
        A1 = A[mask][ind]
        lv1=lv1[mask][ind]
        mult=mult[lv1-1]
        f = gf1/mult
        

        if N < len(ind):
            return voigt_xsection(w[:N], w[:N], f[:N], A1[:N], T, self.getA() * mp), wvl1[:N] #check gf values here
        else:
            return voigt_xsection(w, w, gf1, A1, T, self.getA() * mp), wvl1
