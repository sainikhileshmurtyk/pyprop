"""
Generalized wave propagation with the angular spectrum method using the Jones matrix formulation for accounting for polarization

Author: Nikhilesh Murty, 2020
"""



"""
Loading the packages
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
import cmath

"""
Propagators
"""
class Propagators:
    """       
        Initializing the class of propagation operations
        
        Parameters
        ----------
        fld_in : numpy matrix 
                The input wave field
        wvl : float
                Wavelength of the light used [Units: m]
        num_grid : int 
                Number of grid elements
        grid_spacing : float 
                The distance between two grid points [Units:m]
    """
    
    def __init__(self, wvl, num_grid, grid_spacing, pad_param = 1000):
        """        
        Initializing the class of propagation operations
        
        Parameters
        ----------
        fld_in : numpy matrix 
                The input wave field
        wvl : float
                Wavelength of the light used [Units: m]
        num_grid : int 
                Number of grid elements
        grid_spacing : int 
                The distance between two grid points [Units:m]
        """
        self.wvl = wvl
        self.grid_spacing = grid_spacing
        self.num_grid = num_grid
        self.pad_param = pad_param
        
    def GenerateGrid(self):
        """
        Generates the grid on which the input field and all the calculations can be performed

        Returns
        -------
        xx : numpy matrix
                The x-coordinates of the grid 
        yy : numpy matrix
                The y-coordinates of the grid
        """
        num_grid = self.num_grid
        grid_spacing = self.grid_spacing    
        xx, yy = np.meshgrid(np.linspace(-num_grid/2, num_grid/2, num_grid + 1) * grid_spacing, np.linspace(-num_grid/2, num_grid/2, num_grid + 1) * grid_spacing)
        self.xx = xx
        self.yy = yy
        return xx, yy
    
    def InitSplit(self, fld_in):
        """        
        This function splits the beam into two based on the two orthogonal polarization states

        Parameters
        ----------
        fld_in : numpy matrix
                The input wave field

        Returns
        -------
        fld_x: numpy matrix
                Component of the field along the x-direction
        fld_y: numpy matrix
                Component of the field along the y-direction
        """
        prop_mat = {}
        if type(fld_in) == np.ndarray:
            prop_mat['fld'] = fld_in
        else:
            prop_mat['fld'] = fld_in['fld']
        
        if 'pol' in fld_in:
            prop_mat['pol'] = fld_in['pol']
        elif 'pol' not in fld_in:
            prop_mat['pol'] = np.zeros(prop_mat['fld'].shape, dtype=float)
        else:
            raise ValueError()

        u = prop_mat['fld'] # this is your 2D complex field that you wish to propagate
        E, phi = np.abs(u), np.angle(u)
        E_x = E * np.cos(prop_mat['pol'])
        E_y = E * np.sin(prop_mat['pol'])
        alpha_x = (0) * np.ones(u.shape)
        alpha_y = (np.pi/2) * np.ones(u.shape)
        fld_x = {}
        fld_y = {}
        fld_x['fld'] = E_x * np.exp(1j*phi)
        fld_x['pol'] = alpha_x 
        fld_y['fld'] = E_y * np.exp(1j*phi)
        fld_y['pol'] = alpha_y
        return fld_x, fld_y
        
    
    def AngularSpectrum(self, fld_in, prop_dist):
        """        
        This function returns the propagation of the input wave, fld_in, after a certain distance as defined by prop_dist

        Parameters
        ----------
        fld_in : numpy matrix 
                The input wave field
        prop_dist : float 
                Propagation distance of the wave [Units: m]

        Returns
        -------
        res: numpy matrix
                Propagated beam
        """
        u = fld_in['fld'] # this is your 2D complex field that you wish to propagate
        if 'pol' not in fld_in:
            fld_in['pol'] = np.zeros(u.shape)
        z = prop_dist # this is the distance by which you wish to propagate

        # Padding the input field
        pad_u = np.zeros([u.shape[0] + (2 * self.pad_param), u.shape[1] + (2 * self.pad_param)], dtype=complex)
        pad_u[self.pad_param:self.pad_param + u.shape[0], self.pad_param: self.pad_param + u.shape[1]] = u
        # pad_u = np.pad(u, ((pad_param, pad_param), (pad_param, pad_param)), mode='constant', constant_values = (0,0))


        dx, dy = self.grid_spacing, self.grid_spacing # or whatever

        wavelen = self.wvl # or whatever
        wavenum = 2 * np.pi / wavelen
        wavenum_sq = wavenum * wavenum

        kx = np.fft.fftfreq(pad_u.shape[0], dx / (2 * np.pi))
        ky = np.fft.fftfreq(pad_u.shape[1], dy / (2 * np.pi))

        # this is just for broadcasting, the "indexing" argument prevents NumPy
        # from doing a transpose (default meshgrid behaviour)
        kx, ky = np.meshgrid(kx, ky, indexing = 'ij', sparse = True)

        kz_sq = kx * kx + ky * ky

        # we zero out anything for which this is not true, see many textbooks on
        # optics for an explanation
        mask = wavenum * wavenum > kz_sq

        g = np.zeros((kx.size, ky.size), dtype = np.complex_)
        g[mask] = np.exp(1j * np.sqrt(wavenum_sq - kz_sq[mask]) * z)

        res_pad = np.fft.ifft2(g * np.fft.fft2(pad_u)) # this is the result

        # Removing the padding
        res = {}
        res['fld'] = res_pad[self.pad_param:self.pad_param + u.shape[0], self.pad_param:self.pad_param + u.shape[1]]
        res['pol'] = fld_in['pol']
        return res
    
    def PolAngularSpectrum(self, fld_x, fld_y, prop_dist):
        """
        This function returns the propagation of the input wave, fld_in, after a certain distance as defined by prop_dist

        Parameters
        ----------
        fld_x : numpy matrix
                The input wave field along x-direction
        fld_y : numpy matrix
                The input wave field along the y-direction
        prop_dist : float 
                Propagation distance of the wave [Units: m]
        pol_mat : numpy matrix 
                The matrix that indicates changes in polarization [Units: radians]

        Returns
        -------
        out_fld_x : numpy matrix 
                The input field along the x-directon propagated for a certain distance
        out_fld_y : numpy matrix 
                The input field along the x-directon propagated for a certain distance  
        """
        # Creating new dictionaries
        out_fld_x = {}
        out_fld_y = {}

        # Propagate TE and TM Mat
        out_fld_x = self.AngularSpectrum(fld_x, prop_dist) # Code for propagating along x-direction
        out_fld_y = self.AngularSpectrum(fld_y, prop_dist) # Code for propagating along y-direction
        
        return out_fld_x, out_fld_y
