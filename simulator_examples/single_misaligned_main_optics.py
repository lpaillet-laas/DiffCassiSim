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
from utils import *
import yaml

import cProfile
from CASSI_class import *
from matplotlib import cm

# initialize a lens
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_dir = './render_pattern_demo/'

d_R_lens1 = 7.318                                   # 1/2*Height of the perfect lens (in mm)
d_R_prism = 12.0
#d_R_prism = 8.344                                   # 1/2*Height of the prism (in mm)
d_R_lens2 = 12.7                                    # 1/2*Height of the doublet lens (in mm)
angle = 53.4/2                                      # Angle of the prism

d_x1 = 1.767909                                     # Abscissa of the foot of the first curvature
d_x2 = 4.531195                                     # Abscissa of the foot of the second curvature
d_x3 = 8.651783                                     # Abscissa of the foot of the third curvature

curv_x1 = 0.0                                       # Abscissa of the first curvature midpoint
curv_x2 = 7                                         # Abscissa of the second curvature midpoint
curv_x3 = 9.5                                       # Abscissa of the third curvature midpoint

x1_e = d_x1                                         # Distance between the curvature midpoint and the foot of the first curvature
x2_e = 7-d_x2                                       # Distance between the curvature midpoint and the foot of the second curvature
x3_e = 9.5-d_x3                                     # Distance between the curvature midpoint and the foot of the third curvature
d_length = curv_x3 - curv_x1                        # Length of the lens

d_H = d_R_prism*2                                   # Height of the prism

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
print(d_prism_length)
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

angle_prism_y = - 19.680
d_tilt_angle_final_y = angle_prism_y - 19.383

angle_prism_x = 5.0 #*1/10
prism_edge = torch.tensor([0., 0., 2*d_F], dtype=torch.float32)
prism_center = torch.tensor([0., 0., 2*d_F + d_prism_length/2], dtype=torch.float32) # No rotation for now
prism_center = prism_edge
rotation_matrix_x = torch.tensor([[1, 0, 0],
                                  [0, np.cos(angle_prism_x*np.pi/180), -np.sin(angle_prism_x*np.pi/180)],
                                  [0, np.sin(angle_prism_x*np.pi/180), np.cos(angle_prism_x*np.pi/180)]]).float()
rotation_matrix_y = torch.tensor([[np.cos(angle_prism_y*np.pi/180), 0, np.sin(angle_prism_y*np.pi/180)],
                                    [0, 1, 0],
                                    [-np.sin(angle_prism_y*np.pi/180), 0, np.cos(angle_prism_y*np.pi/180)]]).float()

shift_prism = torch.tensor([0., 0., 0.]) + rotation_matrix_x.T @ rotation_matrix_y.T @ (- (rotation_matrix_x @ (prism_center - prism_edge) + prism_edge - prism_center))
print("Shift: ", shift_prism)

d_shift_value_x = -0.017        #/np.cos((90+d_tilt_angle_final)*np.pi/180)  # Shouldn't need this since all coordinates in Zemax are relative
d_shift_value_y = 0.0
d_shift_value_z = 0.0

d_shift_value_break_x = (0.011*np.sin((d_tilt_angle_final_y)*np.pi/180).item() + 0.024)
d_shift_value_break_z = (0.011*np.cos((d_tilt_angle_final_y)*np.pi/180).item() + 0.2)

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

    usecase = 'spot'


    oversample = 1
    x_center_second_surface_prism = d_prism_length*np.sin(angle_prism_y*np.pi/180).item()
    d_last_surface_distance = 2*d_F + (rotation_matrix_x @ rotation_matrix_y @ torch.tensor([0., 0., d_prism_length]))[-1] + shift_prism[-1]
    origin_last_surface = torch.tensor([0., 0., 2*d_F]) + rotation_matrix_y @ torch.tensor([0., 0., d_prism_length]) + shift_prism

    x_center_end_lens = origin_last_surface[0].item() + (d_F + doublet_length + d_shift_value_x)*np.sin(d_tilt_angle_final_y*np.pi/180).item()
    d_lens_surface = 2*d_F + d_prism_length*np.cos(angle_prism_y*np.pi/180).item() + (d_F + doublet_length + d_shift_value_x)*np.cos(d_tilt_angle_final_y*np.pi/180).item()


    optimized_lens_shift = -0.134
    list_d_sensor = [2*d_F,
                     d_prism_length + d_F,
                     d_F + doublet_length + d_back_F + d_shift_value_break_z + optimized_lens_shift,
                     d_back_F + optimized_lens_shift]
    list_r_last = [d_R_prism, d_R_prism, d_R_prism, d_R_prism]

    list_film_size = [[680, 560] for i in range(4)]  
    list_pixel_size = [10e-3]*4
    list_theta_y = [0., angle_prism_y, d_tilt_angle_final_y, d_tilt_angle_final_y]
    list_theta_x = [0., angle_prism_x, 0., 0.]
    list_theta_z = [0., 0., 0., 0.]
    list_origin = [None, [0., 0., 2*d_F], origin_last_surface.tolist(), [x_center_end_lens, 0., d_lens_surface]]
    list_shift = [[0., 0., 0.], shift_prism.tolist(), [d_shift_value_x,d_shift_value_y, d_shift_value_z], [d_shift_value_break_x, 0., d_shift_value_break_z]]
    list_rotation_order = ['xyz', 'yxz', 'xyz', 'xyz']
    system_wavelengths = torch.linspace(450, 650, 28*oversample)

    lens_group = HSSystem(list_systems_surfaces=[perfect_lens_setup, prism_setup, doublet_setup, parameters_break], list_systems_materials=[perfect_lens_materials, prism_materials, doublet_materials, parameters_break_materials],
                        list_d_sensor=list_d_sensor, list_r_last=list_r_last, list_film_size=list_film_size, list_pixel_size=list_pixel_size,
                        list_theta_y=list_theta_y, list_theta_x=list_theta_x, list_theta_z=list_theta_z,
                        list_origin=list_origin, list_shift=list_shift, list_rotation_order=list_rotation_order,
                        wavelengths = system_wavelengths,
                        device=device, save_dir="../system_comparison_with_zemax/single_prism_misaligned/images/")
    
    prism = lens_group.system[1]

    lens_group.export_system("system_singlemis.yml")
    #lens_group.d_subsystems = torch.tensor([0, 0, 0])
    #lens_group.update_system()
    #lens_group.system = [lens_group.system[0]]
    #lens_group.size_system = 1

    lens_group.combined_plot_setup()

    if usecase in ['psf', 'spot']:   
        lens_group.system[0].d_sensor = d_back_F + d_length + 0.5  # For plotting purposes
        lens_group.system[1].d_sensor = d_back_F + d_length + d_F + d_prism_length + 0.5  # For plotting purposes

    
    if usecase == 'spot':
        """
        Plot the spot diagram for the lens group.
        """
        wavelengths = torch.tensor([450, 520, 650])
        lens_group.plot_spot_less_points(9, 10*1e-3, wavelengths = wavelengths)

    elif usecase == 'render_mapping':
        """
        Render the acquisition and then use the mapping to create the 3D estimation.
        """
        wavelengths = torch.linspace(450, 650, 28).float()
        nb_rays = 1

        texture = np.ones((512, 512, len(wavelengths)))
        #texture = np.zeros((512, 512, len(wavelengths)))
        #texture[25, :, :] = 1
        #texture[:125, 100, :] = 1

        texture = scipy.io.loadmat("/home/lpaillet/Documents/Codes/simca/datasets_reconstruction/mst_datasets/cave_1024_28_train/scene109.mat")['img_expand'][:512,:512].astype('float32')

        texture = torch.from_numpy(texture).float().to(device)

        save_dir="/home/lpaillet/Documents/Codes/article-distorsions-dont-matter/images/"

        plt.figure()
        plt.imshow(torch.load("mask.pt"), cmap='gray')
        plt.axis('off')
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0, wspace = 0)
        plt.margins(0,0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.savefig(save_dir + 'mask_overview.svg', format='svg', bbox_inches = 'tight', pad_inches = 0)

        plt.figure()
        plt.imshow(texture[:,:,0], cmap='Purples')
        plt.axis('off')
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0, wspace = 0)
        plt.margins(0,0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.savefig(save_dir + 'cube1_overview.svg', format='svg', bbox_inches = 'tight', pad_inches = 0)

        plt.figure()
        plt.imshow(texture[:,:,7], cmap='Blues')
        plt.axis('off')
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0, wspace = 0)
        plt.margins(0,0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.savefig(save_dir + 'cube2_overview.svg', format='svg', bbox_inches = 'tight', pad_inches = 0)

        plt.figure()
        plt.imshow(texture[:,:,14], cmap='Greens')
        plt.axis('off')
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0, wspace = 0)
        plt.margins(0,0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.savefig(save_dir + 'cube3_overview.svg', format='svg', bbox_inches = 'tight', pad_inches = 0)

        plt.figure()
        plt.imshow(texture[:,:,21], cmap='Oranges')
        plt.axis('off')
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0, wspace = 0)
        plt.margins(0,0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.savefig(save_dir + 'cube4_overview.svg', format='svg', bbox_inches = 'tight', pad_inches = 0)

        plt.figure()
        plt.imshow(texture[:,:,-1], cmap='Reds')
        plt.axis('off')
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0, wspace = 0)
        plt.margins(0,0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.savefig(save_dir + 'cube5_overview.svg', format='svg', bbox_inches = 'tight', pad_inches = 0)
        plt.show()



        z0 = torch.tensor([lens_group.system[-1].d_sensor*torch.cos(lens_group.system[-1].theta_y*np.pi/180) + lens_group.system[-1].origin[-1] + lens_group.system[-1].shift[-1]]).item()
        
        #mapping_cube = lens_group.get_mapping_scene_detector(wavelengths, shape_scene = [512, 512]).int()
        mapping_cube = torch.load('mapping_cube_singlemis.pt', map_location='cpu').int()
        
        #image = lens_group.render(wavelengths=wavelengths, nb_rays=nb_rays, z0=z0,
        #                texture=texture, numerical_aperture=0.05, plot=False)
        image = torch.load('render_saved_pos_singlemis.pt', map_location='cpu')
        
        image = image.permute(2, 0, 1)
        image = image.unsqueeze(0)

        acq = image.sum(1).flip(1)

        mapping_cube = mapping_cube.permute(2,0,1,3)
        mapping_cube[:, :, :, 0] = image.shape[2] -1  - mapping_cube[:, :, :, 0]
        #mapping_cube[:, :, :, 1] = image.shape[3] -1  - mapping_cube[:, :, :, 1]

        bs, nC, h, w = image.shape

        fake_output = torch.zeros(mapping_cube.shape[:-1]).unsqueeze(0).repeat(bs, 1, 1, 1).float()

        row_indices = mapping_cube[..., 0]
        col_indices = mapping_cube[..., 1]

        for b in range(bs):
            for i in range(nC):
                fake_output[b, i] = acq[b, row_indices[i], col_indices[i]].rot90(2, (0, 1)).flip(0)
            
        fig, axes = plt.subplots(nC//4, 4, figsize=(15, 5))

        # Plot each slice of new_cube in a separate subplot
        for i in range(nC):
            axes[i//4, i%4].imshow(fake_output[0, i,:,:])
            axes[i//4, i%4].set_title(f'Slice {i}')

        plt.figure()
        plt.imshow(acq[0], cmap='gray')
        plt.axis('off')
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0, wspace = 0)
        plt.margins(0,0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.savefig(save_dir + 'acq_overview.svg', format='svg', bbox_inches = 'tight', pad_inches = 0)

        plt.figure()
        plt.imshow(fake_output[0, 0,:,:], cmap = 'Purples')
        plt.axis('off')
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0, wspace = 0)
        plt.margins(0,0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.savefig(save_dir + 'mapping1_overview.svg', format='svg', bbox_inches = 'tight', pad_inches = 0)

        plt.figure()
        plt.imshow(fake_output[0, 7,:,:], cmap = 'Blues')
        plt.axis('off')
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0, wspace = 0)
        plt.margins(0,0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.savefig(save_dir + 'mapping2_overview.svg', format='svg', bbox_inches = 'tight', pad_inches = 0)

        plt.figure()
        plt.imshow(fake_output[0, 14,:,:], cmap = 'Greens')
        plt.axis('off')
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0, wspace = 0)
        plt.margins(0,0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.savefig(save_dir + 'mapping3_overview.svg', format='svg', bbox_inches = 'tight', pad_inches = 0)

        plt.figure()
        plt.imshow(fake_output[0, 21,:,:], cmap = 'Oranges')
        plt.axis('off')
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0, wspace = 0)
        plt.margins(0,0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.savefig(save_dir + 'mapping4_overview.svg', format='svg', bbox_inches = 'tight', pad_inches = 0)

        plt.figure()
        plt.imshow(fake_output[0, -1,:,:], cmap = 'Reds')
        plt.axis('off')
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0, wspace = 0)
        plt.margins(0,0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.savefig(save_dir + 'mapping5_overview.svg', format='svg', bbox_inches = 'tight', pad_inches = 0)

        plt.figure()
        plt.imshow(torch.sum(texture, dim=-1))
        plt.show()

    elif usecase == 'spot_compare_zemax':
        """
        Compare the spot diagram with Zemax.
        """
        lens_group.compare_spot_zemax(path_compare='/home/lpaillet/Documents/Codes/article-distorsions-dont-matter-data/data_zemax/single_prism_misaligned_5deg_delta_beta_c/')
        
    elif usecase == 'mapping':
        """
        Create the mapping cube for the system.
        """
        wavelengths = torch.tensor(system_wavelengths).float().to(device)

        mapping_cube = lens_group.get_mapping_scene_detector(wavelengths, shape_scene = [512, 512])

        torch.save(mapping_cube, 'mapping_cube.pt')

    elif usecase == 'compare_positions_trace':
        """
        Plot the path of the rays coming from different positions in the object plane for a given wavelength.
        """
        N = 2
        nb_ray = 1 + 3*N*(N-1) # Hexapolar number of rays based on N
        print(f"Nb rays: {nb_ray}")
        max_angle = 0.05*180/np.pi
        wavelength = 520.0

        line_pos1, col_pos1 = 0., 0.
        line_pos2, col_pos2 = 0., 2.5
        line_pos3, col_pos3 = 0., -2.5
        list_source_pos = [torch.tensor([col_pos1, line_pos1]), torch.tensor([col_pos2, line_pos2]), torch.tensor([col_pos3, line_pos3])]
        list_source_pos = [list_source_pos[0], list_source_pos[1]]
        colors = ['b-', 'g-', 'r-', 'k-', 'm-']
        
        lens_group.compare_positions_trace(N, list_source_pos, max_angle, wavelength, colors=colors)

    elif usecase == 'compare_wavelength_trace':
        """
        Plot the path of the rays coming from different (or only one) positions in the object plane for several wavelengths.
        """
        wavelengths = [450., 520., 650.]
        colors = ['b-', 'g-', 'r-']
        N = 2
        nb_ray = 1 + 3*N*(N-1) # Hexapolar number of rays based on N
        print(f"Nb rays: {nb_ray}")
        max_angle = 0.05*180/np.pi

        line_pos, col_pos = 0., 0.
        line_pos2, col_pos2 = 0., 2.5
        list_source_pos = [torch.tensor([col_pos, line_pos]), torch.tensor([col_pos2, line_pos2])]

        lens_group.compare_wavelength_trace(N, list_source_pos, max_angle, wavelengths, colors=colors)

    elif usecase == 'render':
        """
        Render the acquisition for a given scene.
        """
        wavelengths = torch.tensor(system_wavelengths).float().to(device)
        wavelengths = torch.tensor([450., 520., 650.]).float().to(device)

        nb_rays = 1

        texture = np.kron([[1, 0] * 10, [0, 1] * 10] * 10, np.ones((25, 25)))[:,:,np.newaxis].repeat(len(wavelengths), axis=-1).astype(np.float32) # Checkboard with each square being 25x25 on a 500x500 size
        texture = np.ones((256, 256, len(wavelengths)), dtype=np.float32)

        #texture = scipy.io.loadmat("/home/lpaillet/Documents/Codes/simca/datasets_reconstruction/mst_datasets/cave_1024_28_train/scene1.mat")['img_expand'][100:356,300:556].astype('float32')
        
        texture = [1 for i in range(4)] + [5] + [4] + [3 for i in range(7)] + [1 for i in range(5)] + [4 for i in range(10)]
        texture = 1000*torch.tensor(texture).repeat_interleave(oversample).unsqueeze(0).unsqueeze(0).repeat(256, 256, 1).float().numpy()
        mask = np.zeros((256, 256), dtype=np.float32)
        mask[:, 128] = 1
        texture = np.multiply(texture, mask[:,:,np.newaxis])
        
        texture = np.ones((512, 512, 3))
        
        texture = torch.from_numpy(texture).float().to(device)

        #for i in range(lens_group.size_system):
        #    lens_group.system[i].film_size = [800, 800]

        z0 = list_d_sensor[-1]
        z0 = torch.tensor([lens_group.system[-1].d_sensor*np.cos(lens_group.system[-1].theta_y*np.pi/180) + lens_group.system[-1].origin[-1] + lens_group.system[-1].shift[-1]]).item()
        print("z0: ", z0)
        #offsets = -d_R + d_x1
        image = lens_group.render(wavelengths=wavelengths, nb_rays=nb_rays, z0=z0,
                        texture=texture, numerical_aperture=0.05, plot=True)
        
        torch.save(image, f'test_n{nb_rays}_ov{int(len(wavelengths)//28)}.pt')

    elif usecase == 'render_lots':
        """
        Render the acquisition for a given scene with different number of rays.
        """
        wavelengths = torch.tensor(system_wavelengths).float().to(device)

        for n in range(1,21):
            nb_rays = 20*n

            texture = np.kron([[1, 0] * 10, [0, 1] * 10] * 10, np.ones((25, 25)))[:,:,np.newaxis].repeat(len(wavelengths), axis=-1).astype(np.float32) # Checkboard with each square being 25x25 on a 500x500 size
            texture = np.ones((256, 256, len(wavelengths)), dtype=np.float32)

            #texture = scipy.io.loadmat("/home/lpaillet/Documents/Codes/simca/datasets_reconstruction/mst_datasets/cave_1024_28_train/scene1.mat")['img_expand'][100:356,300:556].astype('float32')
            
            texture = [1 for i in range(4)] + [5] + [4] + [3 for i in range(7)] + [1 for i in range(5)] + [4 for i in range(10)]
            texture = 1000*torch.tensor(texture).repeat_interleave(oversample).unsqueeze(0).unsqueeze(0).repeat(256, 256, 1).float().numpy()
            mask = np.zeros((256, 256), dtype=np.float32)
            mask[:, 128] = 1
            texture = np.multiply(texture, mask[:,:,np.newaxis])      
            
            texture = torch.from_numpy(texture).float().to(device)


            z0 = list_d_sensor[-1]
            #offsets = -d_R + d_x1
            image = lens_group.render(wavelengths=wavelengths, nb_rays=nb_rays, z0=z0,
                            texture=texture, offsets=[0, 0, 0], numerical_aperture=0.05, plot=False)
            
            torch.save(image, f'test_n{nb_rays}_ov{int(len(wavelengths)//28)}.pt')
    
    elif usecase == 'save_pos_render':
        """
        Save the positions on the detector of traced rays from the scene.
        """
        wavelengths = torch.tensor(system_wavelengths).float().to(device)
        #wavelengths = torch.tensor([450., 520.0, 650.]).float().to(device)
        nb_rays = 20


        texture = np.kron([[1, 0] * 8, [0, 1] * 8] * 8, np.ones((16, 16)))[:,:,np.newaxis].repeat(len(wavelengths), axis=-1).astype(np.float32) # Checkboard with each square being 16x16 on a 256x256 size
        texture = np.ones((int(list_film_size[0][0]), int(list_film_size[0][1]), len(wavelengths)), dtype=np.float32)
        texture = texture[:256,:256,0]
        texture = torch.from_numpy(texture).float().to(device)

        #z0 = list_d_sensor[-1]
        z0 = torch.tensor([lens_group.system[-1].d_sensor*np.cos(lens_group.system[-1].theta_y*np.pi/180) + lens_group.system[-1].origin[-1] + lens_group.system[-1].shift[-1]]).item()
        nb_squares = int(list_film_size[0][0])//32
        nb_squares = 8 
        texture = torch.ones((256, 256, 3))
        
        # big_uv, big_mask = lens_group.save_pos_render(wavelengths=wavelengths, nb_rays=nb_rays, z0=z0,
        #                   texture=texture, numerical_aperture=0.05)
        
        big_uv = torch.load("rays_singlemis.pt", map_location='cpu')
        big_mask = torch.load("rays_valid_singlemis.pt", map_location='cpu')

        texture = np.kron([[1, 0] * nb_squares, [0, 1] * nb_squares] * nb_squares, np.ones((16, 16)))[:,:,np.newaxis].repeat(len(wavelengths), axis=-1).astype(np.float32) # Checkboard with each square being 16x16 on a 256x256 size
        texture = texture[:256,:256,:]

        texture = scipy.io.loadmat("/home/lpaillet/Documents/Codes/simca/datasets_reconstruction/mst_datasets/cave_1024_28_train/scene109.mat")['img_expand'][:512,:512, :].astype('float32')
        texture = torch.from_numpy(texture).float().unsqueeze(0)

        print(texture.shape)
        texture = torch.nn.functional.interpolate(texture, scale_factor=(1, oversample), mode='bilinear', align_corners=True)
        # texture = [1 for i in range(4)] + [5] + [4] + [3 for i in range(7)] + [1 for i in range(5)] + [4 for i in range(10)]
        # texture = 1000*torch.tensor(texture).repeat_interleave(4).unsqueeze(0).unsqueeze(0).repeat(256, 256, 1).float().numpy()
        texture = texture.squeeze(0)
        #mask = np.zeros((512, 512), dtype=np.float32)
        #mask[:, 256] = 1
        mask = torch.load("mask.pt", map_location='cpu')
        texture = np.multiply(texture, mask[:,:,np.newaxis]) 
        # texture = texture[:256,:256,:3]     
        
        texture = torch.from_numpy(texture).float().to(device) if isinstance(texture, np.ndarray) else texture.float().to(device)
        #texture = np.zeros((int(list_film_size[0][0]), int(list_film_size[0][1]), len(wavelengths)), dtype=np.float32)
        #texture[:, texture.shape[0]//2, :] = 1

        airy = torch.load("airy_single.pt", map_location='cpu')

        image = lens_group.render_based_on_saved_pos(big_uv = big_uv, big_mask = big_mask, texture = texture, nb_rays=nb_rays, wavelengths = wavelengths,
                                    z0=z0, plot = True)
        
        airy = airy[::4, ...]
        image = torch.nn.functional.conv2d(image.unsqueeze(0).permute(0, 3, 1, 2), airy.float(), padding = airy.shape[-1]//2, groups=wavelengths.shape[0]).squeeze()
        image = image.permute(1, 2, 0)

        torch.save(image, 'render_saved_pos_singlemis.pt')

    elif usecase == 'get_dispersion':
        """
        Get the dispersion of the system and some tests.
        """
        wavelengths = torch.linspace(450, 650, 3)
        pos_dispersed, pixel_dispersed = lens_group.central_positions_wavelengths(wavelengths)
        rounded_pixel = pixel_dispersed.round().int()
        print(rounded_pixel)
        empty_space = torch.arange((512))
        reduced_empty = empty_space[512//4 + rounded_pixel[0,0]: 512//4 + 512//2 + rounded_pixel[-1,0]]
        print(reduced_empty.shape)
        rounded_pixel = rounded_pixel - rounded_pixel.min(dim=0).values
        print(pos_dispersed, rounded_pixel)
        fake_cube = torch.zeros(256, 3)
        for i in range(3):
            fake_cube[:, i]=reduced_empty[rounded_pixel[i,0]:rounded_pixel[i,0]+256]
        print(fake_cube[:,-1])

    elif usecase == 'psf':
        """
        Plot the PSF for the lens group.
        """
        N = 58
        nb_ray = 1 + 3*N*(N-1) # Hexapolar number of rays based on N
        print(f"Nb rays: {nb_ray}")
        max_angle = 0.05*180/np.pi
        wavelength = 520.0

        line_pos, col_pos = 0., 0.
        source_pos = torch.tensor([col_pos, line_pos])

        ps = lens_group.plot_psf(N, source_pos, max_angle, wavelength, show_rays = False, show_res = False)
    
    elif usecase == 'psf_line':
        """
        Plot the PSF for the lens group for a line of sources.
        """
        N = 58
        nb_ray = 1 + 3*N*(N-1) # Hexapolar number of rays based on N
        print(f"Nb rays: {nb_ray}")
        max_angle = 0.05*180/np.pi
        kernel_size = 11
        wavelengths = system_wavelengths

        line_pos, col_pos = 0., 0.
        source_pos = torch.tensor([col_pos, line_pos])

        d = lens_group.extract_hexapolar_dir(N, source_pos, max_angle)

        psf_line = torch.empty((len(wavelengths), kernel_size, kernel_size))

        for w_id, wavelength in enumerate(tqdm(wavelengths)):
            ps = lens_group.trace_psf_from_point_source(angles = None, x_pos = source_pos[0], y_pos = source_pos[1], z_pos = 0., wavelength = wavelength,
                        show_rays = False, d = d, ignore_invalid = False, show_res = False)
            bins_i, bins_j, centroid = find_bins(ps, list_pixel_size[0], kernel_size)
            hist_ps = torch.histogramdd(ps, bins=(bins_i, bins_j), density=False).hist
            hist_ps /= hist_ps.sum()
            psf_line[w_id, :, :] = hist_ps.reshape(kernel_size, kernel_size)
            #plt.imshow(hist_ps, origin='lower', interpolation='nearest')
            #plt.show()
        torch.save(psf_line, 'psf_line.pt')        
        
    elif usecase == 'optimize_adam_psf_zemax':
        """
        Automatically optimize the distance of the sensor and the angle of the system to match the PSF of Zemax.
        """
        n_iter = 100
        start_dist = lens_group.system[-1].d_sensor
        N = 20
        nb_ray = 1 + 3*N*(N-1) # Hexapolar number of rays based on N
        pixel_size = 5e-3

        source_pos1 = torch.tensor([0., 0.])
        source_pos2 = torch.tensor([2.5, 0.])
        source_pos3 = torch.tensor([0., 2.5])
        source_pos4 = torch.tensor([2.5, 2.5])
        source_pos_list = [source_pos1, source_pos2, source_pos3, source_pos4]
        w_list = [450.0, 520., 650.]

        file_name = "/home/lpaillet/Documents/Codes/article-distorsions-dont-matter-data/data_zemax/AMICI/ray_positions_wavelength_W1_field_F1.h5"

        params = [[source_pos_list[i], w_list[j], extract_positions(file_name.replace('W1', f'W{j+1}').replace('F1', f'F{i+1}'))]
                  for i in range(len(source_pos_list)) for j in range(len(w_list))]
        
        lens_group.system[-1].d_sensor = torch.tensor([lens_group.system[-1].d_sensor], requires_grad=True)
        angle = torch.tensor([20.], requires_grad=True)
        
        optimizer = torch.optim.Adam([lens_group.system[-1].d_sensor, angle], lr=1e-2)
        for i in range(n_iter):
            optimizer.zero_grad()
            loss = 0.0
            for k in range(len(params)):
                d = lens_group.extract_hexapolar_dir(N, params[k][0], angle) 
                ps = lens_group.trace_psf_from_point_source(angles = None, x_pos = params[k][0][0], y_pos = params[k][0][1], z_pos = 0., wavelength = params[k][1],
                        show_rays = False, d = d, ignore_invalid = False, show_res = False).float()
                bins_i, bins_j, centroid = find_bins(ps, pixel_size, 11)
                hist_ps = torch.histogramdd(ps, bins=(bins_i, bins_j), density=False).hist
                hist_ps /= hist_ps.sum()

                ps_zemax = params[k][2]
                ps_zemax = torch.stack(ps_zemax, dim=-1).float()
                bins_i, bins_j, centroid_zemax = find_bins(ps_zemax, pixel_size, 11)
                hist_zemax = torch.histogramdd(ps_zemax, bins=(bins_i, bins_j), density=False).hist
                hist_zemax /= hist_zemax.sum()
                
                #loss += torch.sqrt(torch.mean(((hist_ps - hist_zemax))**2))
                loss += torch.abs(centroid_zemax - centroid).sum()
                #loss += torch.abs(torch.max(ps, dim=0).values - torch.max(ps_zemax, dim=0).values).sum()
            loss /= len(params)
            loss.backward()
            optimizer.step()
            print("Loss: ", loss)
            print("Angle: ", angle.clone().detach())
            print("Dist capteur: ", lens_group.system[-1].d_sensor.clone().detach())
            print("")
            
    elif usecase == 'optimize_psf_zemax':
        """
        Manually optimize the distance of the sensor and the angle of the systems to match the PSF of Zemax.
        """
        start_dist = lens_group.system[-1].d_sensor
        N = 30
        nb_ray = 1 + 3*N*(N-1) # Hexapolar number of rays based on N
        pixel_size = 5e-3

        source_pos1 = torch.tensor([0., 0.])
        source_pos2 = torch.tensor([2.5, 0.])
        source_pos3 = torch.tensor([0., 2.5])
        source_pos4 = torch.tensor([2.5, 2.5])
        source_pos_list = [source_pos1, source_pos2, source_pos3, source_pos4]
        w_list = [450.0, 520., 650.]

        file_name = "/home/lpaillet/Documents/Codes/article-distorsions-dont-matter-data/data_zemax/single_prism_misaligned_5deg_delta_beta_c/ray_positions_wavelength_W1_field_F1.txt"

        depth_list = torch.from_numpy(np.arange(-2., 2., 0.01))
        angle_list = torch.from_numpy(np.arange(5, 50, 2))
        angle_list = torch.tensor([22.5])
        depth_list = torch.from_numpy(np.arange(-0.2, 0.2, 0.001))

        params = [[source_pos_list[i], w_list[j], extract_positions(file_name.replace('W1', f'W{j+1}').replace('F1', f'F{i+1}'))]
                  for i in range(len(source_pos_list)) for j in range(len(w_list))]

        map = lens_group.fit_psf(N, params, depth_list, angle_list, start_dist, pixel_size, kernel_size = 11, show_psf=False)
        torch.save(map, 'map_optim.pt')
        if map.shape[0] > 1 and map.shape[1] > 1:
            fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

            X, Y = np.meshgrid(angle_list, depth_list)

            surf = ax.plot_surface(X, Y, map, cmap=cm.coolwarm, linewidth=0, antialiased=False)

            fig.colorbar(surf, shrink=0.5, aspect=5)
        else:
            if map.shape[0] > 1:
                plt.plot(depth_list, map)
            else:
                plt.plot(angle_list, map)
        plt.show()

    elif usecase == 'compare_psf_zemax':
        """
        Compare the PSF of the system with Zemax.
        """
        N = 40
        nb_ray = 1 + 3*N*(N-1) # Hexapolar number of rays based on N
        print(f"Nb rays: {nb_ray}")
        max_angle = 0.05*180/np.pi
        wavelength = 520.0

        pixel_size = 10e-3

        source_pos1 = torch.tensor([0., 0.])
        source_pos2 = torch.tensor([2.5, 0.])
        source_pos3 = torch.tensor([0., 2.5])
        source_pos4 = torch.tensor([2.5, 2.5])
        source_pos_list = [source_pos1, source_pos2, source_pos3, source_pos4]
        w_list = [450.0, 520., 650.]

        file_name = "/home/lpaillet/Documents/Codes/article-distorsions-dont-matter-data/data_zemax/single_prism_misaligned_5deg_delta_beta_c/ray_positions_wavelength_W1_field_F1.txt"

        params = [[source_pos_list[i], w_list[j], extract_positions(file_name.replace('W1', f'W{j+1}').replace('F1', f'F{i+1}'))]
                  for i in range(len(source_pos_list)) for j in range(len(w_list))]


        lens_group.compare_psf(N, params, max_angle, pixel_size, kernel_size = 11, show_rays = False, show_res = False)
        
    elif usecase == "psf_train_database":
        N = 58
        nb_ray = 1 + 3*N*(N-1) # Hexapolar number of rays based on N
        print(f"Nb rays: {nb_ray}")
        max_angle = 0.05*180/np.pi

        lim = 2.7
        nb_pts = 50
        min_w = 450
        max_w = 650
        n_w = 55
        line_pos_list = np.linspace(-lim, lim, nb_pts)
        col_pos_list = np.linspace(-lim, lim, nb_pts)
        wavelength_list = np.linspace(min_w, max_w, n_w)

        lens_group.create_train_database(N, line_pos_list, col_pos_list, max_angle, wavelength_list, lim)

    elif usecase == "psf_test_database":
        N = 58
        nb_ray = 1 + 3*N*(N-1) # Hexapolar number of rays based on N
        print(f"Nb rays: {nb_ray}")
        max_angle = 0.05*180/np.pi

        database_size = 4000
        lim = 2.56
        line_pos_list = np.random.uniform(-lim, lim, size=database_size)
        col_pos_list = np.random.uniform(-lim, lim, size=database_size)
        wavelength_list = np.random.randint(0, 54, size=database_size)

        lens_group.create_test_database(N, line_pos_list, col_pos_list, max_angle, wavelength_list, lim, index_save = True)

    elif usecase == "psf_test_regular_database":
        N = 58
        nb_ray = 1 + 3*N*(N-1) # Hexapolar number of rays based on N
        print(f"Nb rays: {nb_ray}")
        max_angle = 0.05*180/np.pi

        lim = 2.7
        nb_pts = 50
        min_w = 450
        max_w = 650
        n_w = 55
        line_pos_list = np.linspace(-lim, lim, nb_pts)[:-1] + 2*lim/(nb_pts-1)/2
        col_pos_list = np.linspace(-lim, lim, nb_pts)[:-1] + 2*lim/(nb_pts-1)/2
        wavelength_list = np.linspace(min_w, max_w, n_w)

        lens_group.create_test_regular_database(N, line_pos_list, col_pos_list, max_angle, wavelength_list, 2.56, reduced = True, index_save = True)

    elif usecase == "view_psf_field":
        N = 58
        nb_ray = 1 + 3*N*(N-1) # Hexapolar number of rays based on N
        max_angle = 0.05*180/np.pi
        wavelength_list = [450., 520., 650.]

        depth_list = np.arange(-0.5, 0.51, 0.05)
        depth_list = [0]

        nb_pts = 3
        line_pos_list = np.linspace(2.5, -2.5, nb_pts)
        col_pos_list = np.linspace(2.5, -2.5, nb_pts)

        lens_group.psf_field_view(N, depth_list, line_pos_list, col_pos_list, max_angle, wavelength_list)

    elif usecase == "minimize_psf":
        N = 58
        nb_ray = 1 + 3*N*(N-1) # Hexapolar number of rays based on N
        print(f"Nb rays: {nb_ray}")
        max_angle = 0.05*180/np.pi

        wavelength_list = [450., 520.0, 650.]

        depth_list = np.arange(-0.025, 0.025, 0.005)

        nb_pts = 7

        line_pos_list = np.linspace(2.5, -2.5, nb_pts)
        col_pos_list = np.linspace(2.5, -2.5, nb_pts)

        lens_group.minimize_psf(N, depth_list, line_pos_list, col_pos_list, max_angle, wavelength_list)
    
    elif usecase == "minimize_psf_random":
        N = 58
        nb_ray = 1 + 3*N*(N-1) # Hexapolar number of rays based on N
        print(f"Nb rays: {nb_ray}")
        max_angle = 0.05*180/np.pi

        wavelength_list = [450., 500., 520.0, 600., 650.]

        lim = 2.56
        nb_pts = 49
        depth_list = np.arange(-0.08, 0.02, 0.01)

        n_run = 1

        lens_group.minimize_psf_random(N, depth_list, max_angle, wavelength_list, lim, nb_pts, n_run)

def extract_acq_from_cube(cube_3d, dispersion_pixels, middle_pos, texture_size):
        acq_2d = cube_3d.sum(-1)
        rounded_disp = dispersion_pixels.round().int() # Pixels to indices

        return acq_2d[middle_pos[0] - texture_size[0]//2 + rounded_disp[0,0]: middle_pos[0] + texture_size[0]//2 + rounded_disp[-1,0],
                        middle_pos[1] - texture_size[1]//2 + rounded_disp[0,1]: middle_pos[1] + texture_size[1]//2 + rounded_disp[-1,1]]

def shift_back(inputs, dispersion_pixels, n_lambda = 28): 
    """
        Input [bs, H + disp[0], W + disp[1]]
        Output [bs, n_wav, H, W]
    """
    bs, H, W = inputs.shape
    rounded_disp = dispersion_pixels.round().int() # Pixels to indices
    max_min = rounded_disp.max(dim=0).values - rounded_disp.min(dim=0).values
    output = torch.zeros(bs, n_lambda, H - max_min[0], W - max_min[1])
    abs_shift = rounded_disp - rounded_disp.min(dim=0).values
    for i in range(n_lambda):
        output[:, i, :, :] = inputs[:, abs_shift[i, 0]: H - max_min[0] + abs_shift[i, 0], abs_shift[i, 1]: W - max_min[1] + abs_shift[i, 1]]
    return output

def shift(inputs, dispersion_pixels, n_lambda = 28):
    """
        Input [bs, n_wav, H, W]
        Output [bs, n_wav, H + disp[0], W + disp[1]]
    """
    bs, n_lambda, H, W = inputs.shape
    rounded_disp = dispersion_pixels.round().int() # Pixels to indices
    max_min = rounded_disp.max(dim=0).values - rounded_disp.min(dim=0).values
    output = torch.zeros(bs, n_lambda, H + max_min[0], W + max_min[1])
    abs_shift = rounded_disp - rounded_disp.min(dim=0).values
    for i in range(n_lambda):
        output[:, i, abs_shift[i, 0]: H + abs_shift[i, 0], abs_shift[i, 1]: W + abs_shift[i, 1]] = inputs[:, i, :, :]
    return output

oversample = 10

image_20_4 = torch.load("test_n20_ov4.pt").flip(0,1)
image_180 = torch.load("test_n180_ov4.pt").flip(0,1)
image_20_5 = torch.load("test_n20_ov5.pt").flip(0,1)
image_20_10 = torch.load("test_n20_ov10.pt").flip(0,1)



texture = scipy.io.loadmat("/home/lpaillet/Documents/Codes/simca/datasets_reconstruction/mst_datasets/cave_1024_28_train/scene1.mat")['img_expand'][100:356,300:556]
texture = [1 for i in range(4)] + [5] + [4] + [3 for i in range(7)] + [1 for i in range(5)] + [4 for i in range(10)]
texture = 1000*torch.tensor(texture).repeat_interleave(oversample).float()
# plt.figure()
# plt.imshow(np.sum(texture, axis=-1))
""" image = torch.load("test.pt").flip(0,1)
plt.figure()
plt.title("Panchro acquisition")
plt.imshow(torch.sum(image, dim=-1))

plt.figure()
plt.plot(torch.mean(torch.sum(image, dim=-1)[110:146, 81:172], dim=0))
plt.xlabel("Pixel")
plt.title("Spread spectrum")
plt.figure()
plt.plot(lens_group.wavelengths, texture)
plt.xlabel("Wavelength")
plt.title("True spectrum") """
psf_line = torch.load('psf_line.pt')
convoluted_spectrum = torch.zeros(28*oversample+5+5)
for i in range(28*oversample):
    convoluted_spectrum[i:i+11] += texture.flip(0)[i]*psf_line[i, 5, :]   # Flip because rendered image has to be flipped
    #convoluted_spectrum[i:i+11] += torch.sum(texture[i]*psf_line[i, :, :], dim=0)
convoluted_spectrum = convoluted_spectrum.flip(0)
plt.figure()

x_axis = 1/lens_group.pos_dispersed[:,0].diff()
convoluted_with_dispersion = x_axis*convoluted_spectrum[6:-5]

plt.plot(lens_group.pos_dispersed[1:,0], convoluted_with_dispersion)
plt.xlabel("Position relative to center [mm]")
plt.title("Convoluted spectrum")
plt.figure()
plt.plot(torch.sum(image_20_4, dim=-1)[128, 81:172])
plt.xlabel("Pixel")
plt.title("20 rays, oversampling 4")
plt.figure()
plt.plot(torch.sum(image_20_5, dim=-1)[128, 81:172])
plt.xlabel("Pixel")
plt.title("20 rays, oversampling 5")
plt.figure()
plt.plot(torch.mean(torch.sum(image_20_10, dim=-1)[128-5:128+5, 81:172], dim=0))
plt.xlabel("Pixel")
plt.title("20 rays, oversampling 10")
plt.figure()
plt.plot(torch.sum(image_180, dim=-1)[128, 81:172])
plt.xlabel("Pixel")
plt.title("180 rays, oversampling 4")
plt.show()





""" texture = torch.empty(256, 256, 3)
res = extract_acq_from_cube(image, lens_group.pixel_dispersed, (torch.tensor(list_film_size[-1])//2).int().flip(0), (torch.tensor(texture.shape[:2])).int().flip(0))
plt.imshow(res)
plt.show()
res = res.unsqueeze(0)
back_res = shift_back(res, lens_group.pixel_dispersed, n_lambda = 28)
print(back_res.shape) """
