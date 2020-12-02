"""
Importing the packages
"""
import numpy as np
import os
import matplotlib.pyplot as plt

"""
Other Utilities
"""
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
