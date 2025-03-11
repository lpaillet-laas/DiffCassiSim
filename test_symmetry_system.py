import os
import numpy as np
import scipy.io
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

import sys
sys.path.append("../")
import diffoptics as do

import time
import yaml

import cProfile
from CASSI_class import *
from DD_CASSI_class import DD_CASSI
from matplotlib import cm

from matplotlib.colors import ListedColormap, LinearSegmentedColormap

# initialize a lens
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_dir = './render_pattern_demo/'

d_R_lens1 = 7.318                                   # 1/2*Height of the perfect lens (in mm)
d_R_prism = 12.0
#d_R_prism = 8.344                                   # 1/2*Height of the prism (in mm)
d_R_lens2 = 12.7                                    # 1/2*Height of the doublet lens (in mm)

d_F = 50.0                                          # Focal length of the lens
d_back_F = 43.2                                     # Back focal length of the lens

perfect_lens_setup = [{'type': 'ThinLens',
                 'params': {
                     'f': d_F,
                 },
                 'd': d_F,
                 'R': d_R_lens1,
                }]

perfect_lens_materials = ['air', 'air']

angle1 = 30.0
angle2 = 2*30.0 - angle1

d_prism_length = d_R_prism*(np.cos(angle1*np.pi/180)*np.tan(angle1*np.pi/180) + np.cos(angle2*np.pi/180)*np.tan(angle2*np.pi/180)).item()      # Length of the prism
prism_setup = [{'type': 'XYPolynomial',
                        'params': { 'J':1,
                                'ai': [0,0,np.tan(angle1*np.pi/180).item()],
                            },
                        'd': 0.0,
                        'R': 12*np.cos(angle1*np.pi/180).item(),
                        'is_square': True
                    },
                    {'type': 'XYPolynomial',
                        'params': { 'J':1,
                                'ai': [0,0,-np.tan(angle2*np.pi/180).item()],
                            },
                        'd': d_R_prism*(np.cos(angle2*np.pi/180)*np.tan(angle2*np.pi/180) + np.cos(angle1*np.pi/180)*np.abs(np.tan(angle1*np.pi/180))).item(),
                        'R': 12*np.cos(angle2*np.pi/180).item(),
                        'is_square': True
                    }]

prism_materials = ['air', 'N-BK7', 'air']
doublet_length = 11.5

doublet_setup = [{'type': 'Aspheric',
                'params': {
                    'c': 1/33.34,
                    'k': 0.0,
                    'ai': None,
                },
                'd' : d_F,
                'R': d_R_lens2
                    },
                {'type': 'Aspheric',
                'params': {
                    'c': - 1/22.28,
                    'k': 0.0,
                    'ai': None,
                },
                'd': 9.,
                'R': d_R_lens2
                },
                {'type': 'Aspheric',
                'params': {
                    'c': - 1/291.070,
                    'k': 0.0,
                    'ai': None,
                },
                'd': 2.5,
                'R': d_R_lens2
                }
                ]


doublet_materials = ['air', 'N-BAF10', 'N-SF10', 'air']

parameters_break = [{'type': 'XYPolynomial',
                        'params': { 'J':1,
                                'ai': [0,0,0],
                            },
                        'd': 10.0,
                        'R': (d_back_F-10.)*np.arctan2(d_R_lens2, d_back_F).item(),
                        'is_square': True
                    },
                    {'type': 'XYPolynomial',
                        'params': { 'J':1,
                                'ai': [0,0,0],
                            },
                        'd': 0.0,
                        'R': (d_back_F-10.)*np.arctan2(d_R_lens2, d_back_F).item(),
                        'is_square': True
                    }]

parameters_break_materials = ['air', 'air', 'air']

angle_misalign_prism = 0.0
angle_prism_y = - 19.680
#angle_prism_y = - 19.678
d_tilt_angle_final_y = angle_prism_y - 19.263

d_shift_value_x = -0.017        #/np.cos((90+d_tilt_angle_final_y)*np.pi/180)  # Shouldn't need this since all coordinates in Zemax are relative
#d_shift_value_x = 0.034*np.sin((d_tilt_angle_final_y)*np.pi/180) + 30

d_shift_value_break_x = 0.012*np.sin((d_tilt_angle_final_y)*np.pi/180).item() + 0.024
d_shift_value_break_z = 0.012*np.cos((d_tilt_angle_final_y)*np.pi/180).item() + 0.2
print("x: ", d_shift_value_x)
d_shift_value_y = 0.0
if __name__ == '__main__':
    #usecase is a string defining the example to run among the following:
    # - 'spot': Plot the spot diagram for the lens group.
    # - 'render_mapping': Render the acquisition and then use the mapping to create the 3D estimation.
    # - 'spot_compare_zemax': Compare the spot diagram with Zemax.
    # - 'mapping': Create the mapping cube for the system.
    # - 'compare_positions_trace': Plot the path of the rays coming from different positions in the object plane for a given wavelength.
    # - 'compare_wavelength_trace': Plot the path of the rays coming from different (or only one) positions in the object plane for several wavelengths.
    # - 'render': Render the acquisition for a given scene.
    # - 'render_lots': Render the acquisition for a given scene with different number of rays.
    # - 'save_pos_render': Save the positions on the detector of traced rays from the scene.
    # - 'save_pos_render_all': Save the positions on the detector of traced rays from the scene, with another method.
    # - 'get_dispersion': Get the dispersion of the system and some tests.
    # - 'psf': Plot the PSF for the lens group.
    # - 'psf_line': Plot the PSF for the lens group for a line of sources.
    # - 'optimize_adam_psf_zemax': Automatically optimize the distance of the sensor and the angle of the system to match Zemax system
    # - 'optimize_psf_zemax': Manually optimize the distance of the sensor and the angle of the system to match Zemax system.
    # - 'compare_psf_zemax': Compare the PSF of the system with Zemax.

    usecase = 'render'
    
    oversample = 4
    x_center_second_surface_prism = d_prism_length*np.sin(angle_prism_y*np.pi/180).item()
    #print(x_center_second_surface_prism)

    d_last_surface_distance = 2*d_F + d_prism_length*np.cos((angle_prism_y)*np.pi/180).item()

    x_center_end_lens = d_prism_length*np.sin(angle_prism_y*np.pi/180).item() + (d_F + doublet_length + d_shift_value_x)*np.sin(d_tilt_angle_final_y*np.pi/180).item()
    d_lens_surface = 2*d_F + d_prism_length*np.cos(angle_prism_y*np.pi/180).item() + (d_F + doublet_length + d_shift_value_x)*np.cos(d_tilt_angle_final_y*np.pi/180).item()


    optimized_lens_shift = -0.13
    list_d_sensor = [2*d_F,
                     d_prism_length + d_F,
                     #2*d_F + d_prism_length + d_F + doublet_length + d_back_F + optimized_lens_shift]
                     d_F + doublet_length + d_back_F + optimized_lens_shift,
                     d_back_F + optimized_lens_shift]
    list_r_last = [d_R_prism, d_R_prism, d_R_prism, d_R_prism]

    list_film_size = [[650, 512] for i in range(4)]
    list_pixel_size = [10e-3]*4
    list_theta_y = [0., angle_prism_y, d_tilt_angle_final_y, d_tilt_angle_final_y]
    list_theta_x = [0., angle_misalign_prism, 0., 0.]
    list_theta_z = [0., 0., 0., 0.]
    list_origin = [None, [0., 0., 2*d_F], [x_center_second_surface_prism, 0., d_last_surface_distance], [x_center_end_lens, 0., d_lens_surface]]
    list_shift = [[0., 0., 0.], [0., 0., 0.], [d_shift_value_x,d_shift_value_y, 0.], [d_shift_value_break_x, 0., d_shift_value_break_z]]
    system_wavelengths = torch.linspace(450, 650, 28*oversample)

    dd_system = DD_CASSI(list_systems_surfaces=[perfect_lens_setup, prism_setup, doublet_setup, parameters_break], list_systems_materials=[perfect_lens_materials, prism_materials, doublet_materials, parameters_break_materials],
                        list_d_sensor=list_d_sensor, list_r_last=list_r_last, list_film_size=list_film_size, list_pixel_size=list_pixel_size,
                        list_theta_y=list_theta_y, list_theta_x=list_theta_x, list_theta_z=list_theta_z,
                        list_origin=list_origin, list_shift=list_shift,
                        wavelengths = system_wavelengths,
                        device=device, save_dir=None)
    
    #dd_system.expand_system_symmetrically(symmetry_ax = "any", ax_position = [0., 0., 200.], ax_normal = [0., 0., 1.])
    dd_system.expand_system_symmetrically(symmetry_ax = "any", ax_position = [50., 0., 0.], ax_normal = [1., 0., 0.])
    #dd_system.expand_system_symmetrically(symmetry_ax = "vertical", ax_position = 50.)

    #dd_system.expand_system_symmetrically(symmetry_ax = "any", ax_position = [-7., 0., 175.], ax_normal = [1., 0., 1.])

    #print(dd_system.systems_surfaces)
    print("Origin: ", dd_system.list_origin)
    print("Shift: ", dd_system.list_shift)



    dd_system.combined_plot_setup()