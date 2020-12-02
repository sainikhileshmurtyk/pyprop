"""
Import the packages
"""
import numpy as np


"""
Create Inputs
"""
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
	Gauss['fld'] = amp * np.exp(-(x_pos*x_pos + y_pos*y_pos)/dia2)
	Gauss['pol'] = np.zeros(x_pos.shape)
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
	circ['pol'] = np.zeros(x_pos.shape)
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
	val_uniform['pol'] = np.zeros(x_pos.shape)
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
	SuperGauss['pol'] = np.zeros(x_mat.shape)
	return SuperGauss