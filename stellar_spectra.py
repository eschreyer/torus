import numpy as np
from astropy.io import fits
import constants as const
import astropy.units as u
import os
import scipy.integrate as sp_int


try:
    _PWINDS_REFSPEC_DIR = os.environ["PWINDS_REFSPEC_DIR"]
except KeyError:
    _PWINDS_REFSPEC_DIR = None
    warn("Environment variable PWINDS_REFSPEC_DIR is not set.")

#these 2 functions are in cgs
def nu2wav(nu, F_nu): 
    
    wl = const.c / nu
    F_wl = F_nu * nu**2 / const.c
    return wl, F_wl
    

def wav2nu(wl, F_wl):

    nu = const.c / wl
    F_nu = F_wl * wl**2 / const.c
    return nu, F_nu

def nearest_index(array, target_value):
    """
    Finds the index of a value in ``array`` that is closest to ``target_value``.

    Parameters
    ----------
    array : ``numpy.array``
        Target array.
    target_value : ``float``
        Target value.

    Returns
    -------
    index : ``int``
        Index of the value in ``array`` that is closest to ``target_value``.
    """
    index = array.searchsorted(target_value)
    index = np.clip(index, 1, len(array) - 1)
    left = array[index - 1]
    right = array[index]
    index -= target_value - left < right - target_value
    return index
    
    
class stellar_spectra:

    def __init__(self, stellar_type, semi_major_axis=1, stellar_radius=None, Lbol=None):

        self.spec = self.standard_spectrum(stellar_type, semi_major_axis, stellar_radius=stellar_radius, Lbol=Lbol)
        nu, F_nu = wav2nu(self.spec['wl'].value, self.spec['F_wl'].value)
        self.spec['nu'], self.spec['F_nu'] = nu / u.s, F_nu * u.erg / u.cm**2

        #pwinds spectra
        
        
    def get_spectrum(self, units=None):

        if units:
            return {k: self.spec[k].to(units[i]).value for i,k in enumerate(self.spec)}
        else:
            return {k: v.value for k,v in self.spec.items()}

    def get_rescaled_spectrum(self, norm_wav, norm_flux, rescale_rng='norm', units=[u.cm, u.erg / u.s / u.cm**3, 1 / u.s, u.erg / u.cm**2]):
        
        spectrum = self.get_spectrum(units=units)
        mask = (spectrum['wl'] > norm_wav[0]) & (spectrum['wl'] < norm_wav[1])
        norm_fac = norm_flux / sp_int.trapezoid(spectrum['F_wl'][mask], spectrum['wl'][mask]) #flux in bin / flux meant to be in bin

        if rescale_rng=='norm':
            return {k: (np.where(mask, self.spec[k]*norm_fac, self.spec[k]).to(units[i]).value if k in ('F_wl','F_nu') else self.spec[k].to(units[i]).value) for i,k in enumerate(self.spec)}
        elif rescale_rng=='all':
            return {k : (self.spec[k].to(units[i]).value*norm_fac if k in ('F_wl', 'F_nu')  else self.spec[k].to(units[i]).value) for i,k in enumerate(self.spec)}
        else:
            rescale_rng_mask = (spectrum['wl'] > rescale_rng[0]) & (spectrum['wl'] < rescale_rng[1])
            return {k: (np.where(rescale_rng_mask, self.spec[k]*norm_fac, self.spec[k]).to(units[i]).value if k in ('F_wl','F_nu') else self.spec[k].to(units[i]).value) for i,k in enumerate(self.spec)}
        
    def modify_spectrum(self, norm_wav, norm_flux, rescale_rng='norm'):

        mask = (self.spec['wl'].value > norm_wav[0]) & (self.spec['wl'].value < norm_wav[1])
        norm_fac = norm_flux / sp_int.trapezoid(self.spec['F_wl'][mask].value, self.spec['wl'][mask].value) #flux in bin / flux meant to be in bin

        if rescale_rng=='norm':
            self.spec = {k: (np.where(mask, self.spec[k]*norm_fac, self.spec[k])) if k in ('F_wl','F_nu') else self.spec[k] for i,k in enumerate(self.spec)}
        elif rescale_rng=='all':
            self.spec = {k : self.spec[k]*norm_fac if k in ('F_wl', 'F_nu')  else self.spec[k] for i,k in enumerate(self.spec)}
        else:
            rescale_rng_mask = (self.spec['wl'].value > rescale_rng[0]) & (self.spec['wl'].value < rescale_rng[1])
            self.spec = {k: (np.where(rescale_rng_mask, self.spec[k]*norm_fac, self.spec[k])) if k in ('F_wl','F_nu') else self.spec[k] for i,k in enumerate(self.spec)}
        
    def get_flux_in_bin(self, wav, units=[u.cm, u.erg / u.s / u.cm**3, 1 / u.s, u.erg / u.cm**2]):

        spectrum = self.get_spectrum(units=units)
        mask = (spectrum['wl'] > wav[0]) & (spectrum['wl'] < wav[1])
        flux = sp_int.trapezoid(spectrum['F_wl'][mask], spectrum['wl'][mask]) 
        return flux

    #copied from pwinds
    def standard_spectrum(self, stellar_type, semi_major_axis,
                      reference_spectra_dir=_PWINDS_REFSPEC_DIR,
                      stellar_radius=None, truncate_wavelength_grid=False,
                      cutoff_thresh=13.6, Lbol=None):

        """
    Construct a dictionary containing an input spectrum for a given spectral
    type. The code scales this to the spectrum received at your planet provided
    a value for the scaled ``semi_major_axis``.

    Spectrum of iota Horologii was kindly provided by Jorge Sanz-Forcada (priv.
    comm.). Spectrum of HD 108147 and HR 8799 were obtained from the
    X-exoplanets database and combined with PHOENIX atmosphere models for the
    NUV. Solar spectrum comes from the Whole Heliosphere Interval (WHI)
    Reference Spectra obtained from the LASP Interactive Solar Irradiance
    Datacenter. All other spectra are from the MUSCLES survey.

    Parameters
    ----------
    stellar_type : ``str``
        Define the stellar type. The available options are:

        - ``'mid-A'`` (based on HR 8799)
        - ``'early-F'`` (based on WASP-17)
        - ``'late-F'`` (based on HD 108147)
        - ``'early-G'`` (based on HD 149026)
        - ``'solar'`` (based on the Sun)
        - ``'young-Sun'`` (based on iota Horologii)
        - ``'late-G'`` (based on TOI-193)
        - ``'active-K'`` (based on epsilon Eridanii)
        - ``'early-K'`` (based on HD 97658)
        - ``'late-k'`` (based on WASP-43)
        - ``'active-M'`` (based on Proxima Centauri)
        - ``'early-M'`` (based on GJ 436)
        - ``'late-M'`` (based on TRAPPIST-1)

    semi_major_axis : ``float``
        Semi-major axis of the planet in units of stellar radii. The code first
        converts the MUSCLES spectrum to what it would be at R_star;
        ``semi_major_axis`` is needed to get the spectrum at the planet.

    reference_spectra_dir : ``str``, optional
        Path to the directory with the MUSCLES data. Default value is defined by
        the environment variable ``$PWINDS_REFSPEC_DIR``.

    stellar_radius : ``float``, optional
        Stellar radius in unit of solar radii. Setting a value for this
        parameter allows the spectrum to be scaled to an arbitrary stellar
        radius instead of the radius of the MUSCLES star. If ``None``, then the
        scaling is performed using the radius of the MUSCLES star. Default is
        ``None``.

    truncate_wavelength_grid : ``bool``, optional
        If ``True``, will only return the spectrum with energy > 13.6 eV. This
        may be useful for computational expediency. If False, returns the whole
        spectrum. Default is ``False``.

    cutoff_thresh : ``float``, optional
        If ``truncate_wavelength_grid`` is set to ``True``, then the truncation
        happens for energies whose value in eV is above this threshold, also in
        eV. Default is ``13.6``.

    Returns
    -------
    spectrum : ``dict``
        Spectrum dictionary with entries for the wavelength and flux, and their
        units.
"""
        muscles_match = {'early-A': None, 'late-A': None, 'early-F': 'wasp-17',
                     'late-F': None, 'early-G': 'hd-149026',
                     'late-G': 'toi-193', 'solar': None, 'young-Sun': None,
                     'active-K': 'v-eps-eri', 'early-K': 'hd97658',
                     'late-K': 'wasp-43', 'active-M': 'gj551',
                     'early-M': 'gj436', 'late-M': 'trappist-1'}
        
        muscles_bolometric = {'early-A': None, 'late-A': None, 'early-F': 10**0.61,
                     'late-F': 10**0.3, 'early-G': 10**0.42,
                     'late-G': 10**-0.15, 'solar': 1, 'young-Sun': None,
                     'active-K': 'v-eps-eri', 'early-K': 10**-0.455,
                     'late-K': 10**-0.83, 'active-M': 'gj551',
                     'early-M': 10**-1.63, 'late-M': 10**-3.26}

        try:
            myspectrum = self.generate_muscles_spectrum(muscles_match[stellar_type],
                                             semi_major_axis,
                                             reference_spectra_dir,
                                             stellar_radius,
                                             truncate_wavelength_grid,
                                             cutoff_thresh,
                                             Lbol)
        except KeyError:
            prefix = reference_spectra_dir

        # Check if prefix has a trailing forward slash
            if prefix[-1] == '/':
                pass
        # If not, add it
            else:
                prefix += '/'

            if stellar_type == 'solar':
                spectrum_array = np.loadtxt(
                    prefix + 'ref_solar_irradiance_whi-2008_ver2.dat', skiprows=142,
                    usecols=(0, 2))
                i1 = nearest_index(spectrum_array[:, 0], 300)
                wavelength = (spectrum_array[:i1, 0] * u.nm).to(u.angstrom).value
                flux = (spectrum_array[:i1, 1] * u.W / u.m ** 2 / u.nm).to(
                    u.erg / u.s / u.cm ** 2 / u.angstrom).value
                r_star_origin = 1.00 * u.solRad
                dist = 1 * u.au
            elif stellar_type == 'young-Sun':
                spectrum_array = np.loadtxt(prefix + 'spec_hr810_1au.dat')
                wavelength = spectrum_array[:, 0]
                flux = spectrum_array[:, 1]
                r_star_origin = 1.00 * u.solRad  # Assumption
                dist = 1 * u.au
            elif stellar_type == 'mid-A':
                spectrum_array = np.loadtxt(prefix + 'spec_hr8799_1au.dat')
                wavelength = spectrum_array[:, 0]
                flux = spectrum_array[:, 1]
                r_star_origin = 1.44 * u.solRad  # From Gaia DR2 for HR 8799
                dist = 1 * u.au
            elif stellar_type == 'late-F':
                spectrum_array = np.loadtxt(prefix + 'spec_hd108147_1au.dat')
                wavelength = spectrum_array[:, 0]
                flux = spectrum_array[:, 1]
                r_star_origin = 1.23 * u.solRad  # From Gaia DR2 for HD 108147
                dist = 1 * u.au
            else:
                raise ValueError('Specified stellar type not recognized')

            if stellar_radius is None:
                r_star = r_star_origin
            else:
                r_star = stellar_radius * u.solRad

            conv = float((dist / r_star) ** 2)  # conversion to
            # spectrum at R_star
            spectrum = {'wavelength': wavelength,
                    'flux_lambda': flux * conv * semi_major_axis ** (-2),
                    'wavelength_unit': u.AA,
                    'flux_unit': u.erg / u.s / u.cm / u.cm / u.AA}
            
            if Lbol is None:
                #put spectrum into cgs
                myspectrum = {'wl': (spectrum['wavelength'] * spectrum['wavelength_unit']).to(u.cm),
                          'F_wl': (spectrum['flux_lambda'] * spectrum['flux_unit']).to(u.erg / u.cm**3 / u.s)}
            else:
                myspectrum = {'wl': (spectrum['wavelength'] * spectrum['wavelength_unit']).to(u.cm),
                          'F_wl': (spectrum['flux_lambda'] * spectrum['flux_unit']).to(u.erg / u.cm**3 / u.s) * Lbol / muscles_bolometric[stellar_type]}


        return myspectrum


    def generate_muscles_spectrum(self, host_star_name, semi_major_axis,
                              reference_spectra_dir=_PWINDS_REFSPEC_DIR,
                              stellar_radius=None,
                              truncate_wavelength_grid=False,
                              cutoff_thresh=13.6, Lbol=None):
        """
    Construct a dictionary containing an input spectrum from a MUSCLES spectrum.
    MUSCLES reports spectra as observed at Earth, the code scales this to the
    spectrum received at your planet provided a value for the scaled
    ``semi-major-axis``.

    Parameters
    ----------
    host_star_name : ``str``
        Name of the MUSCLES stellar spectrum you want to use. Must be one of:
        ['gj176', 'gj436', 'gj551', 'gj581', 'gj667c', 'gj832', 'gj876',
        'gj1214', 'hd40307', 'hd85512', 'hd97658', 'v-eps-eri', 'gj1132',
        'hat-p-12', 'hat-p-26', 'hd-149026', 'l-98-59', 'l-678-39', 'l-980-5',
        'lhs-2686', 'lp-791-18', 'toi-193', 'trappist-1', 'wasp-17', 'wasp-43',
        'wasp-77a', 'wasp-127'].

    semi_major_axis : ``float``
        Semi-major axis of the planet in units of stellar radii. The code first
        converts the MUSCLES spectrum to what it would be at R_star;
        ``semi_major_axis`` is needed to get the spectrum at the planet.

    reference_spectra_dir : ``str``, optional
        Path to the directory with the reference spectra. Default value is
        defined by the environment variable ``$PWINDS_REFSPEC_DIR``.

    stellar_radius : ``float``, optional
        Stellar radius in unit of solar radii. Setting a value for this
        parameter allows the spectrum to be scaled to an arbitrary stellar
        radius instead of the radius of the MUSCLES star. If ``None``, then the
        scaling is performed using the radius of the MUSCLES star. Default is
        ``None``.

    truncate_wavelength_grid : ``bool``, optional
        If ``True``, will only return the spectrum with energy > 13.6 eV. This
        may be useful for computational expediency. If False, returns the whole
        spectrum. Default is ``False``.

    cutoff_thresh : ``float``, optional
        If ``truncate_wavelength_grid`` is set to ``True``, then the truncation
        happens for energies whose value in eV is above this threshold, also in
        eV. Default is ``13.6``.

    Returns
    -------
    spectrum : ``dict``
        Spectrum dictionary with entries for the wavelength and flux, and their
        units.
        """
    # Hard coding some values
    # The stellar radii and distances are taken from NASA Exoplanet Archive.

        thresh = cutoff_thresh * u.eV
        stars = [
            # Old ones
            'gj176', 'gj436', 'gj551', 'gj581', 'gj667c', 
            'gj832', 'gj876', 'gj1214', 'hd40307', 'hd85512', 
            'hd97658', 'v-eps-eri',
            # New ones
            #'gj15a', 'gj163', 'gj649', 'gj674', 'gj676a', 'gj699', 'gj729', 'gj849',
            'gj1132', 'hat-p-12', 'hat-p-26', 'hd-149026', 'l-98-59', 
            'l-678-39', 'l-980-5', 'lhs-2686', 'lp-791-18', 'toi-193', 
            'trappist-1', 'wasp-17', 'wasp-43', 'wasp-77a', 'wasp-127'
            ]
        versions = np.array([
            # Old ones
            'v22', 'v22', 'v22', 'v22', 'v22', 
            'v22', 'v22', 'v22', 'v22', 'v22', 
            'v22', 'v22',
            # New ones
            #'v23', 'v23', 'v23', 'v23', 'v23', 'v23', 'v23', 'v23',
            'v23', 'v24', 'v24', 'v24', 'v24', 
            'v24', 'v23', 'v23', 'v24', 'v24', 
            'v23', 'v24', 'v24', 'v24', 'v24'
            ])
        st_rads = np.array([
            # Old ones
            0.46, 0.449, 0.154, 0.3297020, 0.42,
            0.45, 0.35, 0.22, 0.71, 0.69, 
            0.74, 0.77,
            # New ones
            0.21, 0.7, 0.87, 1.41, 0.3, 
            0.34, 0.22, 0.449, 0.18, 0.95, 
            0.12, 1.49, 0.6, 0.910, 1.33 # 1 L 980-5 radius assumed to be the same as GJ 1214 # LHS 2686 radius assumed to be the same as GJ 436
            ]) * u.solRad
        dists = np.array([
            # Old ones
            9.470450, 9.75321, 1.30119, 6.298100, 7.24396, 
            4.964350, 4.67517, 14.6427, 12.9363, 11.2810, 
            21.5618, 3.20260,
            # New ones
            #3.56244, 15.1353,
            12.613, 142.751, 141.837, 75.8643, 10.6194, 
            9.44181, 13.3731, 12.1893, 26.4927, 80.4373, 
            12.4298888, 405.908, 86.7467, 105.6758, 159.507
            ]) * u.pc
        Lbols = np.array([
            1, 10**-1.63, 1, 1, 1, 
            1, 1, 1, 1, 1, 
            10**-0.455, 1, 
            1, 1, 1, 10**0.42, 1, 
            1, 1, 1, 1, 10**-0.15, 
            10**-3.26, 10**0.61,10**-0.83, 1, 1
            ])

        muscles_dists = {starname: dist for starname, dist in zip(stars, dists)}
        muscles_rstars = {starname: st_rad for starname, st_rad in zip(stars,
                                                                   st_rads)}
        muscles_versions = {starname: versions for starname, versions in zip(stars,
                                                                   versions)}
        muscles_Lbols = {starname: bol for starname, bol in zip(stars, Lbols)}

        # MUSCLES records spectra as observed at earth, so we need to convert it to
        # spectrum at R_star. The user has the option of setting an arbitary stellar
        # radius instead of the MUSCLES star radius to allow for more flexibility.
        # This can be especially useful for slightly evolved stars, whose radius
        # are larger than the MUSCLES stars.
        dist = muscles_dists[host_star_name]
        bol = muscles_Lbols[host_star_name]
        vnumber = muscles_versions[host_star_name]
        if stellar_radius is None:
            rstar = muscles_rstars[host_star_name]
        else:
            rstar = stellar_radius * u.solRad
        conv = float((dist / rstar) ** 2)  # conversion to spectrum at R_star

        # First check if reference_spectra_dir has a trailing forward slash
        if reference_spectra_dir[-1] == '/':
            pass
        # If not, add it
        else:
            reference_spectra_dir += '/'

        # Read the MUSCLES spectrum
        spec = fits.getdata(reference_spectra_dir +
                        f'hlsp_muscles_multi_multi_{host_star_name}_broadband_'
                        f'{vnumber}_adapt-const-res-sed.fits',
                        1)

        if truncate_wavelength_grid:
            mask = spec['WAVELENGTH'] * u.AA < thresh.to(u.AA,
                                                     equivalencies=u.spectral())
        else:
            mask = np.ones(spec.shape, dtype='bool')

        spectrum = {'wavelength': spec['WAVELENGTH'][mask],
                    'flux_lambda': spec['FLUX'][mask] * conv *
                    semi_major_axis ** (-2),
                    'wavelength_unit': u.AA,
                    'flux_unit': u.erg / u.s / u.cm / u.cm / u.AA}

        #put spectrum into cgs
        if Lbol is None:
             myspectrum = {'wl': (spectrum['wavelength'] * spectrum['wavelength_unit']).to(u.cm),
                      'F_wl': (spectrum['flux_lambda'] * spectrum['flux_unit']).to(u.erg / u.cm**3 / u.s)} 
        else:
            myspectrum = {'wl': (spectrum['wavelength'] * spectrum['wavelength_unit']).to(u.cm),
                      'F_wl': (spectrum['flux_lambda'] * spectrum['flux_unit']).to(u.erg / u.cm**3 / u.s)*Lbol/bol} 

        
        return myspectrum