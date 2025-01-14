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

    usecase = 'fzf'
    
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

    lens_group = HSSystem(list_systems_surfaces=[perfect_lens_setup, prism_setup, doublet_setup, parameters_break], list_systems_materials=[perfect_lens_materials, prism_materials, doublet_materials, parameters_break_materials],
                        list_d_sensor=list_d_sensor, list_r_last=list_r_last, list_film_size=list_film_size, list_pixel_size=list_pixel_size,
                        list_theta_y=list_theta_y, list_theta_x=list_theta_x, list_theta_z=list_theta_z,
                        list_origin=list_origin, list_shift=list_shift,
                        wavelengths = system_wavelengths,
                        device=device, save_dir="../system_comparison_with_zemax/single_prism/images/")

    lens_group.export_system("system_single.yml")
    #lens_group.d_subsystems = torch.tensor([0, 0, 0])
    #lens_group.update_system()
    #lens_group.system = [lens_group.system[0]]
    #lens_group.size_system = 1
    
    #lens_group.import_system("system_single.yml")
    #lens_group.wavelengths = system_wavelengths
    #lens_group.save_dir = "../system_comparison_with_zemax/single_prism/images/"
    
    lens_group.combined_plot_setup()

    if usecase in ['psf', 'spot']:   
        lens_group.system[0].d_sensor +=  0.5  # For plotting purposes
        lens_group.system[1].d_sensor +=  0.5  # For plotting purposes

    
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
        wavelengths = torch.tensor([450., 520.0, 650.]).float().to(device)
        nb_rays = 1

        texture = np.ones((512, 512, 3))

        #texture = scipy.io.loadmat("/home/lpaillet/Documents/Codes/simca/datasets_reconstruction/mst_datasets/cave_1024_28_train/scene109.mat")['img_expand'][:512,:512].astype('float32')

        texture = torch.from_numpy(texture).float().to(device)

        #mapping_cube = lens_group.get_mapping_scene_detector(wavelengths, shape_scene = [512, 512])
        mapping_cube = torch.load('mapping_cube_simple_single.pt', map_location='cpu')
        print(lens_group.central_positions_wavelengths(torch.linspace(450, 650, 28))[1])

        z0 = torch.tensor([lens_group.system[-1].d_sensor*torch.cos(lens_group.system[-1].theta_y*np.pi/180) + lens_group.system[-1].origin[-1] + lens_group.system[-1].shift[-1]]).item()
        image = lens_group.render(wavelengths=wavelengths, nb_rays=nb_rays, z0=z0,
                        texture=texture, numerical_aperture=0.05, plot=False)
        
        image = image.unsqueeze(0)
        image = image.permute(0, 3, 1, 2)

        acq = image.sum(1).flip(1)

        
        mapping_cube = mapping_cube.permute(2,0,1,3)
        mapping_cube[:, :, :, 0] = image.shape[2] -1  - mapping_cube[:, :, :, 0]

        # mapping_cube[1, :, :, :] = mapping_cube[14, :, :, :]
        # mapping_cube[2, :, :, :] = mapping_cube[-1, :, :, :]
        # mapping_cube = mapping_cube[:3, :, :, :]

        bs, nC, h, w = image.shape

        fake_output = torch.zeros(mapping_cube.shape[:-1]).unsqueeze(0).repeat(bs, 1, 1, 1).float()

        row_indices = mapping_cube[..., 0]
        col_indices = mapping_cube[..., 1]

        print(mapping_cube.shape)
        print(acq.shape)

        for b in range(bs):
            for i in range(nC):
                fake_output[b, i] = acq[b, row_indices[i], col_indices[i]].rot90(2, (0, 1)).flip(0)
            
        fig, axes = plt.subplots(1, nC, figsize=(15, 5))

        # Plot each slice of new_cube in a separate subplot
        for i in range(nC):
            axes[i].imshow(fake_output[0, i,:,:])
            axes[i].set_title(f'Slice {i}')

        plt.figure()
        plt.imshow(acq[0])
        plt.show()
    
    elif usecase == 'spot_compare_zemax':
        """
        Compare the spot diagram with Zemax.
        """
        lens_group.compare_spot_zemax(path_compare='/home/lpaillet/Documents/Codes/article-distorsions-dont-matter-data/data_zemax/single_prism_aligned/')
   

    elif usecase == 'mapping':
        """
        Create the mapping cube for the system.
        """
        wavelengths = torch.tensor(system_wavelengths).float().to(device)
        wavelengths = torch.tensor([450., 520.0, 650.]).float().to(device)

        #for i in range(lens_group.size_system):
        #    lens_group.system[i].film_size = [800, 600]  

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
        colors = ['b', 'lime', 'r']
        N = 2
        nb_ray = 1 + 3*N*(N-1) # Hexapolar number of rays based on N
        print(f"Nb rays: {nb_ray}")
        max_angle = 0.05*180/np.pi

        line_pos, col_pos = 0., 0.
        #line_pos2, col_pos2 = 0., 2.5
        #list_source_pos = [torch.tensor([col_pos, line_pos]), torch.tensor([col_pos2, line_pos2])]
        list_source_pos = [torch.tensor([col_pos, line_pos])]

        lens_group.compare_wavelength_trace(N, list_source_pos, max_angle, wavelengths, colors=colors, linewidth=1.2)

    elif usecase == 'render':
        """
        Render the acquisition for a given scene.
        """
        wavelengths = torch.tensor(system_wavelengths).float().to(device)
        wavelengths = torch.tensor([450., 520.0, 650.]).float().to(device)
        nb_rays = 1

        texture = np.kron([[1, 0] * 10, [0, 1] * 10] * 10, np.ones((25, 25)))[:,:,np.newaxis].repeat(len(wavelengths), axis=-1).astype(np.float32) # Checkboard with each square being 25x25 on a 500x500 size
        texture = np.ones((256, 256, len(wavelengths)), dtype=np.float32)

        #texture = scipy.io.loadmat("/home/lpaillet/Documents/Codes/simca/datasets_reconstruction/mst_datasets/cave_1024_28_train/scene1.mat")['img_expand'][100:356,300:556].astype('float32')
        
        texture = [1 for i in range(4)] + [5] + [4] + [3 for i in range(7)] + [1 for i in range(5)] + [4 for i in range(10)]
        texture = 1000*torch.tensor(texture).repeat_interleave(oversample).unsqueeze(0).unsqueeze(0).repeat(256, 256, 1).float().numpy()
        mask = np.zeros((256, 256), dtype=np.float32)
        mask[:, 128] = 1
        #mask[10, :] = 1
        texture = np.multiply(texture, mask[:,:,np.newaxis])   
        texture = texture[:, :, :]

        texture = np.ones((512, 512, 3))

        #for i in range(lens_group.size_system):
        #    lens_group.system[i].film_size = [800, 600]     
        
        texture = torch.from_numpy(texture).float().to(device)

        z0 = list_d_sensor[-1]
        z0 = torch.tensor([lens_group.system[-1].d_sensor*torch.cos(lens_group.system[-1].theta_y*np.pi/180) + lens_group.system[-1].origin[-1] + lens_group.system[-1].shift[-1]]).item()
        print("z0: ", z0)
        #offsets = -d_R + d_x1
        image = lens_group.render(wavelengths=wavelengths, nb_rays=nb_rays, z0=z0,
                        texture=texture, numerical_aperture=0.05, plot=True)
        
        torch.save(image, f'test_single_n{nb_rays}_ov{int(len(wavelengths)//28)}.pt')

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
                            texture=texture, numerical_aperture=0.05, plot=False)
            
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
        
        big_uv = torch.load("rays_single.pt", map_location='cpu')
        big_mask = torch.load("rays_valid_single.pt", map_location='cpu')

        texture = np.kron([[1, 0] * nb_squares, [0, 1] * nb_squares] * nb_squares, np.ones((16, 16)))[:,:,np.newaxis].repeat(len(wavelengths), axis=-1).astype(np.float32) # Checkboard with each square being 16x16 on a 256x256 size
        texture = texture[:256,:256,:]

        texture = np.ones(512,512)

        #texture = scipy.io.loadmat("/home/lpaillet/Documents/Codes/simca/datasets_reconstruction/mst_datasets/cave_1024_28_train/scene109.mat")['img_expand'][:512,:512, :].astype('float32')
        texture = torch.from_numpy(texture).float().unsqueeze(0)

        print(texture.shape)
        texture = torch.nn.functional.interpolate(texture, scale_factor=(1, oversample), mode='bilinear', align_corners=True)
        # texture = [1 for i in range(4)] + [5] + [4] + [3 for i in range(7)] + [1 for i in range(5)] + [4 for i in range(10)]
        # texture = 1000*torch.tensor(texture).repeat_interleave(4).unsqueeze(0).unsqueeze(0).repeat(256, 256, 1).float().numpy()
        texture = texture.squeeze(0)
        mask = np.zeros((512, 512), dtype=np.float32)
        
        #mask[:, 256] = 1
        #texture = np.multiply(texture, mask[:,:,np.newaxis]) 
        
        # texture = texture[:256,:256,:3]     
        
        texture = torch.from_numpy(texture).float().to(device) if isinstance(texture, np.ndarray) else texture.float().to(device)
        #texture = np.zeros((int(list_film_size[0][0]), int(list_film_size[0][1]), len(wavelengths)), dtype=np.float32)
        #texture[:, texture.shape[0]//2, :] = 1

        airy = torch.load("airy_single.pt", map_location='cpu')
        texture = torch.nn.functional.conv2d(texture.unsqueeze(0).permute(0, 3, 1, 2), airy.float(), padding = airy.shape[-1]//2, groups=wavelengths.shape[0]).squeeze()
        texture = texture.permute(1, 2, 0)

        image = lens_group.render_based_on_saved_pos(big_uv = big_uv, big_mask = big_mask, texture = texture, nb_rays=nb_rays, wavelengths = wavelengths,
                                    z0=z0, plot = True)
        torch.save(image, 'render_saved_pos_single.pt')

    elif usecase == 'save_pos_render_all':
        """
        Save the positions on the detector of traced rays from the scene, with another method.
        """
        wavelengths = torch.tensor([450., 520., 650.])

        nb_rays = 20


        #z0 = list_d_sensor[-1]
        z0 = torch.tensor([lens_group.system[-1].d_sensor*torch.cos(lens_group.system[-1].theta_y*np.pi/180).item() + lens_group.system[-1].origin[-1]  + lens_group.system[-1].shift[-1]]).cpu().detach().item()

        texture = np.ones((256, 256, len(wavelengths)), dtype=np.float32)
        
        big_uv, big_mask = lens_group.save_pos_render(wavelengths=wavelengths, nb_rays=nb_rays, z0=z0,
                          texture=texture, offsets=[0, 0, 0], numerical_aperture=0.05)
        
        texture = [1 for i in range(4)] + [5] + [4] + [3 for i in range(7)] + [1 for i in range(5)] + [4 for i in range(10)]
        texture = 1000*torch.tensor(texture).repeat_interleave(4).unsqueeze(0).unsqueeze(0).repeat(256, 256, 1).float().numpy()
        mask = np.zeros((256, 256), dtype=np.float32)
        mask[:, 128] = 1
        texture = np.multiply(texture, mask[:,:,np.newaxis]) 
        texture = texture[:256,:256,:3]     
        
        texture = torch.from_numpy(texture).float().to(device)
        texture = texture.unsqueeze(0)

        image = lens_group.render_all_based_on_saved_pos(big_uv = big_uv, big_mask = big_mask, texture = texture, nb_rays=nb_rays, wavelengths = wavelengths,
                                    z0=z0)
        plt.imshow(image[0].cpu().detach().numpy())
        plt.show()
        torch.save(image, 'test.pt')


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
        #max_angle = np.arctan2(12.7, 100)*180/np.pi
        kernel_size = list_film_size[0][0]
        kernel_size = tuple(list_film_size[0])
        wavelengths = system_wavelengths

        line_pos1, col_pos1 = 5.12/4, 0.
        source_pos1 = torch.tensor([col_pos1, line_pos1])

        line_pos2, col_pos2 = 0., 0.
        source_pos2 = torch.tensor([col_pos2, line_pos2])

        line_pos3, col_pos3 = -5.12/4, 0.
        source_pos3 = torch.tensor([col_pos3, line_pos3])

        source_pos_list = [source_pos1, source_pos2, source_pos3]
        #psf_line = torch.empty((len(source_pos_list), len(wavelengths), kernel_size[0], kernel_size[1]))
        psf_line = torch.empty((len(source_pos_list), len(wavelengths), kernel_size[1], kernel_size[0]))

        for s_id, source_pos in enumerate(source_pos_list):
            d = lens_group.extract_hexapolar_dir(N, source_pos, max_angle)
            
            for w_id, wavelength in enumerate(tqdm(wavelengths)):
                ps = lens_group.trace_psf_from_point_source(angles = None, x_pos = source_pos[0], y_pos = source_pos[1], z_pos = 0., wavelength = wavelength,
                            show_rays = False, d = d, ignore_invalid = False, show_res = False)
                ps[:, 0] = ps[:, 0] - list_pixel_size[0]/2
                bins_i, bins_j, centroid = find_bins(ps, list_pixel_size[0], kernel_size, same_grid = True, absolute_grid=True)
                """ hist_ps = torch.histogramdd(ps, bins=(bins_i, bins_j), density=False).hist
                hist_ps /= hist_ps.sum()
                psf_line[s_id, w_id, :, :] = hist_ps.reshape(kernel_size[0], kernel_size[1]) """

                hist_ps = torch.histogramdd(ps.flip(1), bins=(bins_j, bins_i), density=False).hist
                hist_ps /= hist_ps.sum()
                psf_line[s_id, w_id, :, :] = hist_ps.reshape(kernel_size[1], kernel_size[0])
                #plt.imshow(hist_ps, origin='lower', interpolation='nearest')
                #plt.show()
        torch.save(psf_line, 'psf_line_single.pt')        
        
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

        file_name = "/home/lpaillet/Documents/Codes/article-distorsions-dont-matter-data/data_zemax/single_prism_aligned/ray_positions_wavelength_W1_field_F1.txt"

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

        file_name = "/home/lpaillet/Documents/Codes/article-distorsions-dont-matter-data/data_zemax/single_prism_aligned/ray_positions_wavelength_W1_field_F1.txt"

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

""" oversample = 10


texture = [1 for i in range(4)] + [5] + [4] + [3 for i in range(7)] + [1 for i in range(5)] + [4 for i in range(10)]
texture = 1000*torch.tensor(texture).repeat_interleave(oversample).float()

texture = scipy.io.loadmat("../processing_reconstruction/datasets_reconstruction/mst_datasets/cave_1024_28_train/scene109.mat")['img_expand'][:512,:512].astype('float32')
#texture = scipy.io.loadmat("/home/lpaillet/Documents/Codes/simca/datasets_reconstruction/mst_datasets/cave_1024_28_train/scene105.mat")['img_expand'][:512,:512].astype('float32')
texture_acq = torch.nn.functional.interpolate(torch.from_numpy(texture).unsqueeze(0), scale_factor=(1, 4), mode='bilinear', align_corners=True).squeeze()
texture = torch.nn.functional.interpolate(torch.from_numpy(texture).unsqueeze(0), scale_factor=(1, oversample), mode='bilinear', align_corners=True).squeeze()
#texture = torch.nn.functional.conv2d(texture.unsqueeze(0).permute(0, 3, 1, 2), airy.float(), padding = airy.shape[-1]//2, groups=system_wavelengths.shape[0]).squeeze()
#texture = texture.permute(1, 2, 0)

# plt.figure()
# plt.imshow(np.sum(texture, axis=-1))


#image = torch.load("test_single_n100_ov10.pt").flip(0,1)
#image = torch.load("/home/lpaillet/Documents/Codes/DiffOptics/examples/render_saved_pos_single.pt").flip(0)


#image = torch.load("image_rendered.pt").flip(0)

mask = np.zeros((512, 512), dtype=np.float32)
mask[:, 256] = 1

texture_acq = np.multiply(texture_acq, mask[:,:,np.newaxis]).float().to(device)

z0 = torch.tensor([lens_group.system[-1].d_sensor*torch.cos(lens_group.system[-1].theta_y*np.pi/180) + lens_group.system[-1].origin[-1] + lens_group.system[-1].shift[-1]]).item()
image = lens_group.render(wavelengths=torch.linspace(450, 650, 28*4), nb_rays=20, z0=z0,
                    texture=texture_acq, numerical_aperture=0.05, plot=False).flip(0)

plt.figure()
plt.title("Panchro texture")
plt.imshow(torch.sum(texture, dim=-1))

colors = [(0, 0, 0), (0.45, 0.45, 0.45), (0.75, 0.75, 0.75), (0.9, 0.9, 0.9), (1, 1, 1)]
custom_gray_cmap = LinearSegmentedColormap.from_list("Custom", colors, N=2000)
plt.figure(figsize=(32, 18), dpi=60)
plt.axis('off')
plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
    hspace = 0, wspace = 0)
plt.margins(0,0)
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
#plt.imshow(image.sum(-1), cmap='gray')
plt.imshow(image.sum(-1), cmap=custom_gray_cmap)

#plt.savefig(lens_group.save_dir + "acquisition_rendering.svg", format='svg', bbox_inches = 'tight', pad_inches = 0)

# plt.figure()
# plt.plot(lens_group.wavelengths, texture)
# plt.plot(lens_group.wavelengths, texture[256, 256, :])
# plt.xlabel("Wavelength")
# plt.title("True spectrum")

psf_line = torch.load('/home/lpaillet/Documents/Codes/DiffOptics/examples/psf_line_single.pt')

print("Pixels: ", lens_group.pixel_dispersed)
#dispersed_texture = texture[1:]*1/lens_group.pos_dispersed[:,0].diff()
dispersed_texture = texture[256, 256, 1:]*1/lens_group.pos_dispersed[:,0].diff()

plt.figure()
plt.plot(lens_group.pos_dispersed[1:,0], dispersed_texture)
plt.xlabel("Position relative to center [mm]")
plt.title("Dispersion applied to spectrum")

#psf_line = psf_line.rot90(1, [2,3]).flip(2)
#psf_line = psf_line.flip(3)

def compute_airy_disk(wavelengths, pixel_size, na=0.05, grid_size = 7, magnification = 1):
    airy_disk_kernel = torch.zeros(wavelengths.shape[0], 1, grid_size, grid_size, device=wavelengths.device)
    for i in range(wavelengths.shape[0]):
        airy_disk_kernel[i, 0, :, :] = airy_disk(wavelengths[i]*1e-6, na, pixel_size, grid_size, magnification = 2)
    return airy_disk_kernel

def airy_disk(wavelength, na, pixel_size, grid_size, magnification = 1):
    # Create a grid of coordinates
    x = torch.linspace(-grid_size // 2 +1, grid_size// 2 , grid_size) * pixel_size
    y = torch.linspace(-grid_size // 2 +1, grid_size// 2 , grid_size) * pixel_size
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    # Calculate the radial distance from the center
    R = torch.sqrt(X**2 + Y**2)

    # Compute the Airy disk pattern
    k = 1/magnification * torch.pi* 2*R*torch.tan(torch.as_tensor(na))/ wavelength
    airy_pattern = (2*torch.special.bessel_j1(k) / k).pow(2)
    airy_pattern[R == 0] = 1  # Handle the singularity at the center

    # Normalize the pattern
    airy_pattern /= airy_pattern.sum()

    return airy_pattern

airy = compute_airy_disk(torch.linspace(450, 650, 280), lens_group.system[-1].pixel_size, na=0.05, grid_size = 7, magnification = 2)
#airy = airy[56, 0, :, :].unsqueeze(0).unsqueeze(0).repeat(280, 1, 1, 1)
psf_line = torch.nn.functional.conv2d(psf_line, airy.float(), padding = airy.shape[-1]//2, groups=280).squeeze()

dispersed_texture_pixel = torch.zeros(psf_line.shape[0], list_film_size[0][0])
#texture = texture[256, 256, :]
# for i, elem in enumerate(texture):
#     accu = elem*torch.sum(psf_line[i, :, :], axis=0)
#     dispersed_texture_pixel += accu/oversample

#texture = texture[256-10:256+10, 256, :]

plt.figure()
plt.imshow(psf_line[0, -1, :, :])

texture = texture[:, 256, :]
for pos in range(psf_line.shape[0]):
    sub_texture = texture[128*(pos+1)-10:128*(pos+1)+10, :]
    for i in range(sub_texture.shape[0]):
        for j in range(sub_texture.shape[1]):
            accu = sub_texture[i, j]*torch.sum(psf_line[pos, j, :, :], axis=0)
            dispersed_texture_pixel[pos, :] += accu/oversample
    dispersed_texture_pixel[pos, :] /= sub_texture.shape[0]


dispersed_texture_pixel = torch.zeros(psf_line.shape[0], list_film_size[0][0])
temp = torch.zeros(psf_line.shape[2], psf_line.shape[3])
fake_texture = torch.zeros(psf_line.shape[2], psf_line.shape[3])
for pos in range(psf_line.shape[0]):
    sub_texture = texture[128*(pos+1)+1-10:128*(pos+1)+1+10, :]
    for i in range(sub_texture.shape[0]):
        for j in range(sub_texture.shape[1]):
            accu = sub_texture[i, j]*psf_line[pos, j, :, :]
            temp += torch.roll(accu, i - sub_texture.shape[0]//2, 0)/oversample

# dispersed_texture_pixel[0, :] = torch.roll(temp[128-10:128+10, :].sum(0),-1)
# dispersed_texture_pixel[1, :] = torch.roll(temp[256-10:256+10, :].sum(0),-1)
# dispersed_texture_pixel[2, :] = torch.roll(temp[384-10:384+10, :].sum(0),-1)

dispersed_texture_pixel[0, :] = temp[128-10:128+10, :].sum(0)
dispersed_texture_pixel[1, :] = temp[256-10:256+10, :].sum(0)
dispersed_texture_pixel[2, :] = temp[384-10:384+10, :].sum(0)

dispersed_texture_pixel = dispersed_texture_pixel#.flip(1)

plt.figure()
plt.imshow(temp)

plt.figure(figsize=(32, 18), dpi=60)
plt.axis('off')
plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
    hspace = 0, wspace = 0)
plt.margins(0,0)
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
plt.plot(texture[128, :], linewidth=10, color="red")
plt.plot(texture[256, :], linewidth=10, color="green")
plt.plot(texture[384, :], linewidth=10, color="blue")

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
fig = plt.figure(figsize=(32, int(18*179.668/100.682)), dpi=60)
ax = fig.add_subplot(111)
#plt.rcParams.update({'font.size': 90,
#                     'text.usetex':True,
#                     "font.family": "Computer Modern Roman",})
plt.rcParams.update({'font.size': 90})
#plt.plot(dispersed_texture_pixel[81:172]/dispersed_texture_pixel[81:172].max())
plt.plot(list(range(270,380)), dispersed_texture_pixel[0, 270:380]/dispersed_texture_pixel[0, 270:380].max(), linewidth=10, color="#005673", linestyle='--')
plt.plot(list(range(270,380)), dispersed_texture_pixel[1, 270:380]/dispersed_texture_pixel[1, 270:380].max(), linewidth=10, color="#ab7b4e", linestyle='--', label='_nolegend_')
plt.plot(list(range(270,380)), dispersed_texture_pixel[2, 270:380]/dispersed_texture_pixel[2, 270:380].max(), linewidth=10, color="#9d9e9f", linestyle='--',label='_nolegend_')
#plt.plot(torch.mean(torch.sum(image, dim=-1)[128-5:128+5, 81:172], dim=0)/torch.mean(torch.sum(image, dim=-1)[128-5:128+5, 81:172], dim=0).max())
#plt.plot(list(range(250,360)), torch.sum(image[256,250:360], dim=-1)/torch.sum(image[256,250:360], dim=-1).max())
plt.plot(list(range(270,380)), torch.sum(torch.mean(image[128-20:128+20,270:380], dim=0), dim=-1)/torch.sum(torch.mean(image[128-20:128+20,270:380], dim=0), dim=-1).max(), linewidth=10, color="#008673")
plt.plot(list(range(270,380)), torch.sum(torch.mean(image[256-20:256+20,270:380], dim=0), dim=-1)/torch.sum(torch.mean(image[256-20:256+20,270:380], dim=0), dim=-1).max(), linewidth=10, color="#eb7b4e")
plt.plot(list(range(270,380)), torch.sum(torch.mean(image[384-20:384+20,270:380], dim=0), dim=-1)/torch.sum(torch.mean(image[384-20:384+20,270:380], dim=0), dim=-1).max(), linewidth=10, color="#bdbebf")
ax.set_xlabel("Pixels [px]", fontsize=90)
#ax.set_ylabel("Normalized intensity", rotation = 270, fontsize=90, labelpad=100)
#plt.title("(Si) system")
plt.legend(["Ground truth spectrum", "Acquisition"])
ax.tick_params(axis='both', which='major', labelsize=90, width=5, length=20)
ax.yaxis.set_label_position("right")
ax.yaxis.tick_right()

#plt.savefig(lens_group.save_dir + "rendering_comparison.svg", format='svg', bbox_inches = 'tight', pad_inches = 0)
plt.show()
 """
