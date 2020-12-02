"""
Importing the packages
"""
import numpy as np

"""
Detector
"""
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
