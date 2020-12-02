"""
Importing the packages
"""
import numpy as np

"""
Polarization Utilities
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
