"""
Script to propagate a defined wavefront for a pre-defined distance and to save the result based on Angular Spectrum

Author: Nikhilesh Murty (2020)
"""
'''
Importing the libraries
'''
import Library.pyprop as prop
import matplotlib.pyplot as plt
import os
import numpy as np

'''
Setting the Input parameters
'''
# Young's double slit parameter
stripe_size = [500,50]
separation = 100
pol = [0,np.pi/2, np.pi/2]

# Important Simulation Parameters
num_grid = 1000
grid_spacing = 5e-6

# Important Physical Parameters
dia_aper = 5e-4
wavelen = 532e-9
prop_dist = [175e-3, 325e-3]

# File saving
directory = os.path.join('Results','Tests','Accounting Geometrical Phase')
filename = '[201201] YDS - Without Polarization Masks (5000 padding and positive product)'
if not os.path.exists(directory):
    os.makedirs(directory)
saveloc = os.path.join(directory, filename) 

'''
Generating the input wave-field
'''
# Setting the Propagator
propagate = prop.Propagation.Propagators(wavelen, num_grid, grid_spacing)

# Generating the grid
xx,yy = propagate.GenerateGrid()

# Generating the input field
fld_in = prop.Inputs.SuperGaussian(xx,yy,4e-3,2e-3,2,2)
# fld_in = prop.Masks.PolMasks.YDSEPol(fld_in, stripe_size, separation, pol)
fld_in = prop.Masks.PhaseMasks.YDSEPhase(fld_in, stripe_size, separation)

'''
Propagating the beam
'''
fld_in_x, fld_in_y = propagate.InitSplit(fld_in)
fld_a_x, fld_a_y = propagate.PolAngularSpectrum(fld_in_x, fld_in_y, prop_dist[0])
fld_out_ana_x, fld_out_ana_y = prop.PolUtils.Analyzer(fld_a_x, fld_a_y, prop.PolUtils.ConvertRad(0)) 
fld_out_x, fld_out_y = propagate.PolAngularSpectrum(fld_out_ana_x, fld_out_ana_y, prop_dist[0])
img_out = prop.Detector.Camera(fld_out_x,fld_out_y)
print('Beam propagated\n')
'''
Plotting the image
'''
plt.imshow(img_out)
# plt.imshow(np.imag(fld_in['fld']))
plt.colorbar()
plt.savefig(saveloc + '.png', dpi=800, transparent=True)
plt.savefig(saveloc + '.pdf')
plt.close()
print('Images saved\n')
