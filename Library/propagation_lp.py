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
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import cmath
import imageio

"""
Inputs
"""
class Inputs:
    def GaussBeam(x_pos, y_pos, dia, x_shift = 0.0, y_shift = 0.0, amp = 1):
        """        
        Generates a Gaussian input field on a grid

        Parameters
        ----------
        x_pos : numpy matrix
                x-coordinates of the grid
        y_pos : numpy matrix
                y-coordinates of the grid
        dia : float
                diameter of the Gaussian beam
        x_shift : float (default = 0.0)
                Shift in the x-direction from the center
        y_shift : float (default - 0.0)
                Shift in y-direction from the center
        amp : float (default = 1)
                Amplitude of the input beam

        Retuns
        ------
        Gauss: numpy matrix
                A gaussian field output
        """
        y_pos = y_pos - y_shift
        x_pos = x_pos - x_shift
        dia2 = (dia/2) * (dia/2)
        Gauss = {}
        Gauss['fld']=amp * np.exp(-(x_pos*x_pos + y_pos*y_pos)/dia2)
        return Gauss

    def Circ(x_pos, y_pos, diameter, x_shift=0.0, y_shift=0.0, amp=1):
        """
        Generates a Circular input field on a grid

        Parameters
        ----------
        x_pos : numpy matrix
                x-coordinates of the grid
        y_pos : numpy matrix
                y-coordinates of the grid
        diameter : float 
                diameter of the Gaussian beam
        x_shift : float (default = 0.0)
                Shift in the x-direction from the center
        y_shift : float (default - 0.0)
                Shift in y-direction from the center
        amp : float (default = 1)
                Amplitude of the input beam

        Returns
        -------
        circ : numpy matrix
                A circular field
        """
        radius = np.sqrt(x_pos**2+y_pos**2)
        if np.max(radius) < (diameter/2):
            raise Exception('The value of the diameter should be smaller than the propagating image')
        val_circ = np.double(radius<(diameter/2))
        val_circ[radius == (diameter/2)] = amp
        circ = {}
        circ['fld'] = val_circ
        return circ

    def Uniform(x_pos, y_pos):
        """
        Generates a uniform illuminated background

        Parameters
        ----------
        x_pos : numpy matrix
                x-coordinates of the grid
        y_pos : numpy matrix
                y-coordinates of the grid

        Returns
        -------
        val_uniform : numpy matrix 
                Matrix with uniform values
        """
        val_uniform = {}
        val_uniform['fld'] = np.ones(x_pos.shape)
        return val_uniform

    def SuperGaussian(x_mat, y_mat, dia_x, dia_y, Px, Py):
        """
        SuperGaussian(x_mat, y_mat, dia_x, dia_y, Px, Py)

        Parameters
        ----------
        x_mat : numpy matrix
                x-coordinates of the grid
        y_mat : numpy matrix
                y-coordinates of the grid
        dia_x : float
                Diameter of the super-gaussian along x
        dia_y : float
                Diameter of the super-gaussian along y
        Px : int
                Power of the super-gaussian raised along x coordinates
        Px : int
                Power of the super-gaussian raised along y coordinates
        
        Returns
        -------
        SuperGauss : Te super-gaussian input field
        """
        SuperGauss = {}
        SuperGauss['fld'] = np.exp(-((x_mat**2)/(dia_x/2)**2)**Px - ((y_mat**2)/(dia_y**2))**Py)
        return SuperGauss

"""
Phase Masks
"""
class PhaseMasks:
    def AddPhase(inp_fld, phase_mask):
        """
        Add an arbitrary phase mask
        
        Parameters
        ----------
        inp_fld : numpy matrix
                Field
        phase_mask : numpy matrix 
                Phase mask to be added [Units: radians]

        Returns
        -------
        out_fld : numpy matrix 
                Field with the phase added
        """
        inp_fld = inp_fld['fld']
        out_fld = inp_fld * np.exp(-1j * np.pi * phase_mask)
        return out_fld

    def LensApply(inp_fld, wavelen, ref_idx, foc_len, aperture, num_grid, grid_spacing, offset = [0.0, 0.0]):
        """
        Generate a phase mask that replicates that of a lens and apply it to the input field

        Parameters
        ----------
        inp_fld : numpy matrix 
                Input field
        wavelen : numpy matrix 
                Wavelength of the light field [Units: m]
        ref_idx : float 
                Refractive Index of the medium [Units: dimensionless]
        foc_len : float 
                Focal length of the lens [Units: m]
        aperture : float 
                Input aperture [Units: m]
        num_grid : int 
                Number of grid points on which the calculation is performed
        grid_spacing : int 
                Spacing between the grid points mentioned above [Units: m]
        offset : numpy matrix [1 × 2]
                Offset along the center of the lens [Datatype: Numpy array (2), Default: [0.0, 0.0]]

        Returns
        -------
        fld_out : numpy matrix 
                Output field after application of the phase mask
        """
        fld_in = inp_fld['fld']
        xx, yy = np.meshgrid(np.linspace(-num_grid/2, num_grid/2, num_grid + 1) * grid_spacing, np.linspace(-num_grid/2, num_grid/2, num_grid + 1) * grid_spacing)
        wave_num = 2 * np.pi * ref_idx * (1/wavelen)
        xx -= offset[0]
        yy -= offset[1]
        fi = -wave_num * (xx**2+yy**2)/(2*foc_len)
        fld_out = fld_in * np.exp(-1j * fi)
        return fld_out

    def Checkboard(inp_size, cboard_size): 
        """
        Applies a checkboard phase mask to the dimensions of a field

        Parameters
        ----------
        inp_size : int 
                Size of the input matrix [Units: m]
        cboard_size : int 
                Size of the Checkboard [Units: m]

        Returns
        -------
        pad_x : numpy matrix
                Output field with a checkboard phase pattern
        """
        # create a n * n matrix
        x = np.zeros((cboard_size[0], cboard_size[1]), dtype = int) 
        
        # fill with 1 the alternate rows and columns 
        x[1::2, ::2] = 90
        x[::2, 1::2] = 90
        # Padding the input field
        pad_x = np.zeros([inp_size[0], inp_size[1]], dtype=complex)
        pad_x[int((inp_size[0]-cboard_size[0])/2):int((inp_size[0]+cboard_size[0])/2), int((inp_size[1]-cboard_size[1])/2):int((inp_size[1]+cboard_size[1])/2)] = x
        return pad_x
"""
Intensity Masks
"""
class IntensityMasks:
    def YDSEInt(fld_in, stripe_size, separation, intensity = [0,1,1]):
        """
        Applies a Young's double slit intensity mask to an input field

        Parameters
        ----------
        fld_in : numpy matrix 
                Input field
        stripe_size : int 
                Size of the slit [Units: m]
        separation : float 
                Separation between the center line of the two slits [Units: m] 
        intensity : numpy matrix [1 × 3] 
                Intensity of the light that is allowed through the slits

        Returns
        -------
        fld_out : numpy matrix 
                Field with the mask applied
        """
        inp_fld = fld_in['fld']
        inp_size = inp_fld.shape
        int_mat = np.ones(inp_size) * intensity[0]
        if stripe_size[1] > inp_size[1] or stripe_size[0] > inp_size[0]:
            raise Exception('The value of the stripe width or height cannot be larger than the image dimensions')
        if separation < stripe_size[1]:
            raise Exception('The stripe separation is too small to form two independent stripes')
        int_mat[int((0.5 * inp_size[0]) - (0.5 * stripe_size[0])) : np.int((0.5 * inp_size[0]) + (0.5 * stripe_size[0])), int((0.5 * (inp_size[1] - separation)) - (0.5 * stripe_size[1])) : int((0.5 * (inp_size[1] - separation)) + (0.5 * stripe_size[1]))] = intensity[1] * inp_fld[int((0.5 * inp_size[0]) - (0.5 * stripe_size[0])) : np.int((0.5 * inp_size[0]) + (0.5 * stripe_size[0])), int((0.5 * (inp_size[1] - separation)) - (0.5 * stripe_size[1])) : int((0.5 * (inp_size[1] - separation)) + (0.5 * stripe_size[1]))] 
        int_mat[int((0.5 * inp_size[0]) - (0.5 * stripe_size[0])) : np.int((0.5 * inp_size[0]) + (0.5 * stripe_size[0])), int((0.5 * (inp_size[1] + separation)) - (0.5 * stripe_size[1])) : int((0.5 * (inp_size[1] + separation)) + (0.5 * stripe_size[1]))] = intensity[2] * inp_fld[int((0.5 * inp_size[0]) - (0.5 * stripe_size[0])) : np.int((0.5 * inp_size[0]) + (0.5 * stripe_size[0])), int((0.5 * (inp_size[1] - separation)) - (0.5 * stripe_size[1])) : int((0.5 * (inp_size[1] - separation)) + (0.5 * stripe_size[1]))]
        fld_out = {}
        fld_out['fld'] = int_mat
        return fld_out

"""
Polarization Masks
"""
class PolMasks:
    def CheckboardPol(inp_size, cboard_size, pol = np.pi/2):
        """
        Applies a checkboard polarization mask to the dimensions of a field

        Parameters
        ----------
        inp_size : int 
                Size of the input matrix [Units: m]
        cboard_size : int 
                Size of the Checkboard [Units: m]
        pol: float
                Polarization [Units: radians, Default: π/2]

        Returns
        -------
        pad_x : numpy matri
                Output field with a checkboard polarization pattern       
        """

        # create a n * n matrix
        x = np.zeros((cboard_size[0], cboard_size[1]), dtype = float) 
        
        # fill with 1 the alternate rows and columns 
        x[1::2, ::2] = pol
        x[::2, 1::2] = pol
        # Padding the input field
        pad_x = np.zeros([inp_size[0], inp_size[1]], dtype=int)
        pad_x[int((inp_size[0]-cboard_size[0])/2):int((inp_size[0]+cboard_size[0])/2), int((inp_size[1]-cboard_size[1])/2):int((inp_size[1]+cboard_size[1])/2)] = x
        return pad_x

    def StripePol(inp_size, stripe_size, pol = np.pi/2):
        """
        Applies a stripe polarization mask to the dimensions of a field

        Parameters
        ----------
        inp_size : int 
                Size of the input matrix [Units: m]
        stripe_size : int 
                Size of the stripes [Units: m]
        pol: float
                Polarization [Units: radians, Default: π/2]

        Returns
        -------
        pad_x : numpy matrix
                Output field with a stripe polarization pattern   
        """
        out = np.zeros((inp_size[0], inp_size[1]), dtype=float)
        out[0:stripe_size, :] = pol
        return out

    def YDSEPol(inp_size, stripe_size, separation, pol = [0,np.pi/2,np.pi/4]):
        """
        Applies a Young's double slit polarization mask to an input field

        Parameters
        ----------
        inp_size : numpy matrix 
                Input field size
        stripe_size : float
                Size of the slit [Units: m]
        separation : float
                Separation between the center line of the two slits [Units: m] 
        pol : float
                Polarization of the light that is allowed through the slits [Units: radians, Datatype: Numpy array (3), Default: [0, π/2, π/4]]

        Returns
        -------
        fld_out : numpy matrix 
                Field with the mask applied 
        """
        pol_mat = np.ones(inp_size) * pol[0]
        if stripe_size[1] > inp_size[1] or stripe_size[0] > inp_size[0]:
            raise Exception('The value of the stripe width or height cannot be larger than the image dimensions')
        if separation < stripe_size[1]:
            raise Exception('The stripe separation is too small to form two independent stripes')
        pol_mat[int((0.5 * inp_size[0]) - (0.5 * stripe_size[0])) : np.int((0.5 * inp_size[0]) + (0.5 * stripe_size[0])), int((0.5 * (inp_size[1] - separation)) - (0.5 * stripe_size[1])) : int((0.5 * (inp_size[1] - separation)) + (0.5 * stripe_size[1]))] = pol[1]
        pol_mat[int((0.5 * inp_size[0]) - (0.5 * stripe_size[0])) : np.int((0.5 * inp_size[0]) + (0.5 * stripe_size[0])), int((0.5 * (inp_size[1] + separation)) - (0.5 * stripe_size[1])) : int((0.5 * (inp_size[1] + separation)) + (0.5 * stripe_size[1]))] = pol[2]
        return pol_mat

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
    
    def __init__(self, wvl, num_grid, grid_spacing, pad_param = 5000):
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
"""
Polarization Utilities
"""
class PolUtils:
    """
    A collection of relevant polarization utilities the can be used throughout the program
    """
    def AddPolFields(fld1_mat, fld2_mat, min_pol = 0, max_pol = 90, step_pol = 1):
        """        
        Add polarization matrices of same size using the principle of superposition of waves
        
        Parameters
        ----------
        fld1_mat : numpy matrix
                First input matrix of different fields
        fld2_mat : numpy matrix 
                First input matrix of different fields
        min_pol : float 
                Minimum polarization angle [Units: degrees, Default: 0]
        max_pol : float 
                Maximum Polarization angle [Units: degrees, Default: 90]
        step_pol : float
                Steps between two polarization propagation [Units: Degrees, Default: 1]

        Returns
        -------
        new_add_mat : numpy matrix 
                Matrix with both the fields added
        """ 
        if fld1_mat.shape != fld2_mat.shape:
            print('Input fields don\'t match in size')
            return 0
        new_add_mat = np.zeros(fld1_mat.shape, dtype=complex)
        for li in range(min_pol, max_pol, step_pol):
            tmp_add_mat = fld1_mat[:,:,li] + fld2_mat[:,:,li]
            new_add_mat = np.zstack(new_add_mat, tmp_add_mat)
        return new_add_mat

    def PolRotation(fld_x, fld_y, pol_mat):
        """
        Parameters
        ------
        fld_x : numpy matrix
                Field along the x-direction
        fld_y : numpy matrix
                Field along the y-direction
        pol_mat : numpy matrix
                The polarization values by which each matrix is rotated [Units: radians]

        Returns
        -------
        fld_x_out : numpy matrix 
                Output field along x-direction
        fld_y_out : numpy matrix
                Output field along y-direction
        """
        # Obtaining te polar components of the field
        E_x, phi_x = np.abs(fld_x['fld']), np.angle(fld_x['fld'])
        E_y, phi_y = np.abs(fld_y['fld']), np.angle(fld_y['fld'])

        # Finding the projection of those fields
        E_x *= (1/np.cos(pol_mat))
        E_y *= (1/np.cos(pol_mat))

        # Creating and defining the output dictionaries
        fld_x_out = {}
        fld_y_out = {}
        fld_x_out['fld'] = E_x * np.exp(1j*phi_x)
        fld_y_out['fld'] = E_y * np.exp(1j*phi_y)
        fld_x_out['pol'] = fld_x['pol']
        fld_y_out['pol'] = fld_y['pol']
        return fld_x_out, fld_y_out

    
    def SelectPol(fld, pol):
        """
        Select a certain polarization from the propagation matrices

        Parameters
        ----------
        fld: numpy matrix
                Input field of the form dict
        pol: float
                polarization [Units: radians]

        Returns
        -------
        pol_fld : numpy matrix 
                Field along the selected polarization
        """
        fld = fld[:,:,0]
        cond_pol_mat = fld == pol
        cond_pol_mat = np.invert(cond_pol_mat)
        pol_fld = np.ma.masked_array(fld, mask=cond_pol_mat)
        pol_fld = np.ma.filled(pol_fld.astype(complex), fill_value=0)
        return pol_fld
        
    def Analyzer(fld_x, fld_y, pol):
        """
        Find the resultant field fter placing an analyzer after accounting for polarization of all the other pixels

        Parameters
        ----------
        fld_x : numpy matrix
                Input field along x-direction
        fld_y : numpy matrix
                Input field along the y-direction
        pol: float
                Orientation of the analyzer [Units: radians]

        Returns
        -------
        fld_out : numpy matrix 
                Field along the single polarization axis defined by pol
        """
        # Splitting into real and complex parts (in polar coordinates)
        E_x, phi_x = np.abs(fld_x['fld']), np.angle(fld_y['fld'])
        E_y, phi_y = np.abs(fld_y['fld']), np.angle(fld_y['fld'])
        
        # Applying the polarization angle
        E_out_x = np.square(np.cos(pol)) * E_x + np.cos(pol) * np.sin(pol) * E_y
        E_out_y = np.cos(pol) * np.sin(pol) * E_x +  np.square(np.sin(pol)) * E_y
        
        # Creating new dictionaries
        fld_out_x = {}
        fld_out_y = {}

        # Combining the real and complex parts into a field and saving to a dict
        fld_out_x['fld'] = E_out_x * np.exp(1j*phi_x)
        fld_out_y['fld'] = E_out_y * np.exp(1j*phi_y)

        fld_out_x['pol'] = fld_x['pol']
        fld_out_y['pol'] = fld_y['pol']
        return fld_out_x, fld_out_y
    
    def ConvertRad(Degrees):
        """
        Converts degrees to Radians

        Parameters
        ----------
        Degrees : float
                Angle in Degrees

        Returns
        -------
        deg_rad : float
                Angle in radians
        """
        deg_rad = (Degrees/180) * np.pi
        return deg_rad

"""
Detector
"""
class Detector:
    def Camera(fld_x, fld_y):
        """
        Calculates the image that a square law detector like a camera would detect for the given input

        Parameters
        ----------
        fld_x : numpy matrix
                Field along the x-direction
        fld_y : numpy matrix
                Field along the y-direction

        Returns
        -------
        img_out : numpy matrix 
                Output image
        """
        # Splitting into real and imaginary parts in the polar form
        E_x, phi_x = np.abs(fld_x['fld']), np.angle(fld_x['fld'])
        E_y, phi_y = np.abs(fld_y['fld']), np.angle(fld_y['fld'])

        # Finding the resultant image on a square law detector
        img_out = np.square(E_x) + np.square(E_y)
        return img_out

"""
Other Utilities
"""
class Utils:
    def CheckFolder(folder_link):
        """
        Checks if a given folder exists

        Parameters
        ----------
        folder_link : string
                Address of the folder

        Returns
        -------
        None
        """
        if ~folder_link:
            os.mkdir(folder_link)
        return 0
    
    def ShowFig(fld, comp='fld', type='E2'):
        """
        Displays a figure using the matplotlib library for the given input

        Parameters
        ----------
        fld : numpy matrix
                Input field
        comp : string
                Component of the field to be displayed
        type : string
                Choose between E/E2 depending on which the field or the intensity is shown
        """
        fld_in = fld[comp]
        if type == 'E':
            plt.imshow(np.abs(fld_in))
            plt.colorbar()
        elif type == 'E2':
            plt.imshow(np.square(np.abs(fld_in)))
            plt.colorbar()
