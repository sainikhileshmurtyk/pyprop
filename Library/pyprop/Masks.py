"""
Importing the packages
"""
import numpy as np

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

	def YDSEPhase(fld_in, stripe_size, separation, phase = [0, 0.5 * np.pi, 0.5 * np.pi]):
		"""
		Applies a Young's double slit phase mask to an input field

		Parameters
		----------
		fld_in : dict of numpy matrices 
				Input field with field and polarization component
		stripe_size : int 
				Size of the slit [Units: m]
		separation : float 
				Separation between the center line of the two slits [Units: m] 
		intensphaseity : numpy matrix [1 × 3] 
				Intensity of the light that is allowed through the slits

		Returns
		-------
		fld_out : numpy matrix 
				Field with the mask applied
		"""
		fld_out = {}
		inp_fld = fld_in['fld']
		inp_size = inp_fld.shape
		if stripe_size[1] > inp_size[1] or stripe_size[0] > inp_size[0]:
				raise Exception('The value of the stripe width or height cannot be larger than the image dimensions')
		if separation < stripe_size[1]:
				raise Exception('The stripe separation is too small to form two independent stripes')
		phase_mat = np.ones(inp_size) * phase[0]
		phase_mat[int((0.5 * inp_size[0]) - (0.5 * stripe_size[0])) : np.int((0.5 * inp_size[0]) + (0.5 * stripe_size[0])), int((0.5 * (inp_size[1] - separation)) - (0.5 * stripe_size[1])) : int((0.5 * (inp_size[1] - separation)) + (0.5 * stripe_size[1]))] = phase[1]
		phase_mat[int((0.5 * inp_size[0]) - (0.5 * stripe_size[0])) : np.int((0.5 * inp_size[0]) + (0.5 * stripe_size[0])), int((0.5 * (inp_size[1] + separation)) - (0.5 * stripe_size[1])) : int((0.5 * (inp_size[1] + separation)) + (0.5 * stripe_size[1]))] = phase[2]
		out_fld = inp_fld * np.exp(1j * phase_mat)
		fld_out['fld'] = out_fld
		fld_out['pol'] = fld_in['pol']
		return fld_out
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

	def YDSEPol(fld_in, stripe_size, separation, pol = [0,np.pi/2,np.pi/4]):
			"""
			Applies a Young's double slit polarization mask to an input field

			Parameters
			----------
			fld_in : dict 
					Input field matrix with field and polarization components
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
			fld_out = {}
			fld_out['fld'] = fld_in['fld']
			inp_pol = fld_in['pol']
			inp_size = inp_pol.shape
			pol_mat = np.ones(inp_size) * pol[0]
			if stripe_size[1] > inp_size[1] or stripe_size[0] > inp_size[0]:
					raise Exception('The value of the stripe width or height cannot be larger than the image dimensions')
			if separation < stripe_size[1]:
					raise Exception('The stripe separation is too small to form two independent stripes')
			pol_mat[int((0.5 * inp_size[0]) - (0.5 * stripe_size[0])) : np.int((0.5 * inp_size[0]) + (0.5 * stripe_size[0])), int((0.5 * (inp_size[1] - separation)) - (0.5 * stripe_size[1])) : int((0.5 * (inp_size[1] - separation)) + (0.5 * stripe_size[1]))] = pol[1]
			pol_mat[int((0.5 * inp_size[0]) - (0.5 * stripe_size[0])) : np.int((0.5 * inp_size[0]) + (0.5 * stripe_size[0])), int((0.5 * (inp_size[1] + separation)) - (0.5 * stripe_size[1])) : int((0.5 * (inp_size[1] + separation)) + (0.5 * stripe_size[1]))] = pol[2]
			fld_out['pol'] = pol_mat
			return fld_out


