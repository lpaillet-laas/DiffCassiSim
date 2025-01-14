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

device = 'cpu'  # Change device as wished

figure_number = 14 # Change figure to generate



def airy_disk(wavelength, na, pixel_size, grid_size, magnification = 1):
    """
    Compute the Airy disk pattern.

    Parameters:
        wavelength (float): Wavelength of the light.
        na (float): Angle of the numerical aperture (in radians).
        pixel_size (float): Size of the pixel.
        grid_size (int): Size of the grid for the computation.
        magnification (float): Magnification factor of the Airy disk.
    Returns:
        torch.Tensor: 2D tensor representing the Airy disk pattern.
    """
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

def compute_airy_disk(wavelengths, pixel_size, na=0.05, grid_size = 7, magnification = 2):
    airy_disk_kernel = torch.zeros(wavelengths.shape[0], 1, grid_size, grid_size, device=wavelengths.device)
    for i in range(wavelengths.shape[0]):
        airy_disk_kernel[i, 0, :, :] = airy_disk(wavelengths[i]*1e-6, na, pixel_size, grid_size, magnification = magnification)
    return airy_disk_kernel

if figure_number == 1:
    sp_system = HSSystem(config_file_path="../system_specs/system_single.yml", device = device)
    ap_system = HSSystem(config_file_path="../system_specs/system_amici.yml", device = device)

    wavelengths = [450., 520., 650.]
    colors = ['b', 'lime', 'r']
    N = 2
    nb_ray = 1 + 3*N*(N-1) # Hexapolar number of rays based on N
    print(f"Nb rays: {nb_ray}")
    max_angle = 0.05*180/np.pi

    line_pos, col_pos = 0., 0.
    list_source_pos = [torch.tensor([col_pos, line_pos])]

    sp_system.compare_wavelength_trace(N, list_source_pos, max_angle, wavelengths, colors=colors, linewidth=1.2)

    ap_system.compare_wavelength_trace(N, list_source_pos, max_angle, wavelengths, colors=colors, linewidth=1.2)

elif (figure_number == 3) or (figure_number == 5):
    sp_system = HSSystem(config_file_path="../system_specs/system_single.yml", device = device)
    msp_system = HSSystem(config_file_path="../system_specs/system_singlemis.yml", device = device)
    ap_system = HSSystem(config_file_path="../system_specs/system_amici.yml", device = device)
    map_system = HSSystem(config_file_path="../system_specs/system_amicimis.yml", device = device)

    sp_system.compare_spot_zemax(path_compare='./data_zemax/single_prism_aligned/')

    msp_system.compare_spot_zemax(path_compare='./data_zemax/single_prism_misaligned/')

    ap_system.compare_spot_zemax(path_compare='./data_zemax/amici_prism_aligned/')

    map_system.compare_spot_zemax(path_compare='./data_zemax/amici_prism_misaligned/')

elif (figure_number == 4) or (figure_number == 9):
    msp_system = HSSystem(config_file_path="../system_specs/system_singlemis.yml", device = device)

    N = 40
    nb_ray = 1 + 3*N*(N-1) # Hexapolar number of rays based on N
    print(f"Nb rays: {nb_ray}")
    max_angle = 0.05*180/np.pi

    pixel_size = 10e-3

    source_pos1 = torch.tensor([0., 0.])
    source_pos2 = torch.tensor([2.5, 0.])
    source_pos3 = torch.tensor([0., 2.5])
    source_pos4 = torch.tensor([2.5, 2.5])
    source_pos_list = [source_pos1, source_pos2, source_pos3, source_pos4]
    w_list = [450.0, 520., 650.]

    file_name = "./data_zemax/single_prism_misaligned_5deg_delta_beta_c/ray_positions_wavelength_W1_field_F1.txt"

    params = [[source_pos_list[i], w_list[j], extract_positions(file_name.replace('W1', f'W{j+1}').replace('F1', f'F{i+1}'))]
                for i in range(len(source_pos_list)) for j in range(len(w_list))]


    msp_system.compare_psf(N, params, max_angle, pixel_size, kernel_size = 11, show_rays = False, show_res = False)

elif (figure_number == 8):
    sp_system = HSSystem(config_file_path="../system_specs/system_single.yml", device = device)

    N = 40
    nb_ray = 1 + 3*N*(N-1) # Hexapolar number of rays based on N
    print(f"Nb rays: {nb_ray}")
    max_angle = 0.05*180/np.pi

    pixel_size = 10e-3

    source_pos1 = torch.tensor([0., 0.])
    source_pos2 = torch.tensor([2.5, 0.])
    source_pos3 = torch.tensor([0., 2.5])
    source_pos4 = torch.tensor([2.5, 2.5])
    source_pos_list = [source_pos1, source_pos2, source_pos3, source_pos4]
    w_list = [450.0, 520., 650.]

    file_name = "./data_zemax/single_prism_misaligned_5deg_delta_beta_c/ray_positions_wavelength_W1_field_F1.txt"

    params = [[source_pos_list[i], w_list[j], extract_positions(file_name.replace('W1', f'W{j+1}').replace('F1', f'F{i+1}'))]
                for i in range(len(source_pos_list)) for j in range(len(w_list))]


    sp_system.compare_psf(N, params, max_angle, pixel_size, kernel_size = 11, show_rays = False, show_res = False)

elif (figure_number == 10):
    ap_system = HSSystem(config_file_path="../system_specs/system_amici.yml", device = device)

    N = 40
    nb_ray = 1 + 3*N*(N-1) # Hexapolar number of rays based on N
    print(f"Nb rays: {nb_ray}")
    max_angle = 0.05*180/np.pi

    pixel_size = 10e-3

    source_pos1 = torch.tensor([0., 0.])
    source_pos2 = torch.tensor([2.5, 0.])
    source_pos3 = torch.tensor([0., 2.5])
    source_pos4 = torch.tensor([2.5, 2.5])
    source_pos_list = [source_pos1, source_pos2, source_pos3, source_pos4]
    w_list = [450.0, 520., 650.]

    file_name = "./data_zemax/single_prism_misaligned_5deg_delta_beta_c/ray_positions_wavelength_W1_field_F1.txt"

    params = [[source_pos_list[i], w_list[j], extract_positions(file_name.replace('W1', f'W{j+1}').replace('F1', f'F{i+1}'))]
                for i in range(len(source_pos_list)) for j in range(len(w_list))]


    ap_system.compare_psf(N, params, max_angle, pixel_size, kernel_size = 11, show_rays = False, show_res = False)

elif (figure_number == 11):
    map_system = HSSystem(config_file_path="../system_specs/system_amicimis.yml", device = device)

    N = 40
    nb_ray = 1 + 3*N*(N-1) # Hexapolar number of rays based on N
    print(f"Nb rays: {nb_ray}")
    max_angle = 0.05*180/np.pi

    pixel_size = 10e-3

    source_pos1 = torch.tensor([0., 0.])
    source_pos2 = torch.tensor([2.5, 0.])
    source_pos3 = torch.tensor([0., 2.5])
    source_pos4 = torch.tensor([2.5, 2.5])
    source_pos_list = [source_pos1, source_pos2, source_pos3, source_pos4]
    w_list = [450.0, 520., 650.]

    file_name = "./data_zemax/single_prism_misaligned_5deg_delta_beta_c/ray_positions_wavelength_W1_field_F1.txt"

    params = [[source_pos_list[i], w_list[j], extract_positions(file_name.replace('W1', f'W{j+1}').replace('F1', f'F{i+1}'))]
                for i in range(len(source_pos_list)) for j in range(len(w_list))]


    map_system.compare_psf(N, params, max_angle, pixel_size, kernel_size = 11, show_rays = False, show_res = False)

elif figure_number == 12:
    sp_system = HSSystem(config_file_path="../system_specs/system_single.yml", device = device)
    ap_system = HSSystem(config_file_path="../system_specs/system_amici.yml", device = device)

    disp_amici = ap_system.central_positions_wavelengths(torch.linspace(450, 650, 28))[0][:,0]

    disp_single = sp_system.central_positions_wavelengths(torch.linspace(450, 650, 28))[0][:,0]


    # Compare the two systems
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'
    params = {'axes.labelsize': 90/2.5,'axes.titlesize':90/2.5, 'legend.fontsize': 90/2.5, 'xtick.labelsize': 70/2.5, 'ytick.labelsize': 70/2.5}
    matplotlib.rcParams.update(params)
    plt.rcParams.update(params)
    fig = plt.figure(figsize=(32/2.5, 18/2.5), dpi=60*2.5)
    ax = fig.add_subplot(111)
    plt.rcParams.update({'font.size': 90/2.5})
    plt.plot(torch.linspace(450, 650, 28), 1000*disp_amici, label='Amici (AP)')
    plt.plot(torch.linspace(450, 650, 28), 1000*disp_single, label='Single (SP)')
    plt.legend()
    ax.set_xlabel("Wavelength [nm]", fontsize=90/2.5)
    ax.set_ylabel('Spreading [µm]', fontsize=90/2.5)
    ax.tick_params(axis='both', which='major', labelsize=90/2.5, width=5/2.5, length=20/2.5)
    plt.grid("on")
    plt.show()

elif figure_number == 13:
    sp_system = HSSystem(config_file_path="../system_specs/system_single.yml", device = device)
    msp_system = HSSystem(config_file_path="../system_specs/system_singlemis.yml", device = device)
    ap_system = HSSystem(config_file_path="../system_specs/system_amici.yml", device = device)
    map_system = HSSystem(config_file_path="../system_specs/system_amicimis.yml", device = device)

    wavelengths = torch.tensor([450, 520, 650])

    sp_system.plot_spot_less_points(9, 10*1e-3, wavelengths = wavelengths)
    msp_system.plot_spot_less_points(9, 10*1e-3, wavelengths = wavelengths)
    ap_system.plot_spot_less_points(9, 10*1e-3, wavelengths = wavelengths)
    map_system.plot_spot_less_points(9, 10*1e-3, wavelengths = wavelengths)

elif figure_number == 14:
    sp_system = HSSystem(config_file_path="../system_specs/system_single.yml", device = device)
    msp_system = HSSystem(config_file_path="../system_specs/system_singlemis.yml", device = device)
    ap_system = HSSystem(config_file_path="../system_specs/system_amici.yml", device = device)
    map_system = HSSystem(config_file_path="../system_specs/system_amicimis.yml", device = device)

    system = sp_system

    oversample = 4
    nb_rays = 1  # Adjust the number of rays as you wish

    wavelengths = torch.linspace(450, 650, 28*oversample)

    texture = scipy.io.loadmat("../processing_reconstruction/datasets_reconstruction/mst_datasets/cave_1024_28_train/scene109.mat")['img_expand'][:512,:512].astype('float32')
    texture_acq = torch.nn.functional.interpolate(torch.from_numpy(texture).unsqueeze(0), scale_factor=(1, oversample), mode='bilinear', align_corners=True).squeeze()

    mask = torch.load("../processing_reconstruction/mask.pt")

    texture_acq = np.multiply(texture_acq, mask[:,:,np.newaxis]).float().to(device)    

    z0 = torch.tensor([system.system[-1].d_sensor*torch.cos(system.system[-1].theta_y*np.pi/180) + system.system[-1].origin[-1] + system.system[-1].shift[-1]]).item()
    airy_acq = compute_airy_disk(wavelengths, sp_system.system[-1].pixel_size, na=0.05, grid_size = 7, magnification = 2)
    texture_acq = torch.nn.functional.conv2d(texture_acq.unsqueeze(0).permute(0, 3, 1, 2), airy_acq.float(), padding = airy_acq.shape[-1]//2, groups=wavelengths.shape[0]).squeeze()
    texture_acq = texture_acq.permute(1, 2, 0)

    image = system.render(wavelengths=wavelengths, nb_rays=nb_rays, z0=z0,
                    texture=texture, numerical_aperture=0.05, plot=False).flip(0)

    colors = [(0, 0, 0), (0.45, 0.45, 0.45), (0.75, 0.75, 0.75), (0.9, 0.9, 0.9), (1, 1, 1)]
    custom_gray_cmap = LinearSegmentedColormap.from_list("Custom", colors, N=2000)
    plt.figure(figsize=(32, 18), dpi=60)
    plt.axis('off')
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
        hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.imshow(image.sum(-1), cmap=custom_gray_cmap)
    if system.save_dir is not None:
        plt.savefig(system.save_dir + "acquisition_rendering.svg", format='svg', bbox_inches = 'tight', pad_inches = 0)

elif figure_number == 6:
    sp_system = HSSystem(config_file_path="../system_specs/system_single.yml", device = device)

    oversample_truth = 10
    oversample_acq = 4
    nb_rays = 20  # Adjust the number of rays as you wish

    wavelengths = torch.linspace(450, 650, 28*oversample_truth)
    wavelengths_acq = torch.linspace(450, 650, 28*oversample_acq)

    ### Create line of PSFs
    N = 58
    max_angle = 0.05*180/np.pi
    kernel_size = tuple(sp_system.system[-1].film_size)

    line_pos1, col_pos1 = 5.12/4, 0.
    source_pos1 = torch.tensor([col_pos1, line_pos1])

    line_pos2, col_pos2 = 0., 0.
    source_pos2 = torch.tensor([col_pos2, line_pos2])

    line_pos3, col_pos3 = -5.12/4, 0.
    source_pos3 = torch.tensor([col_pos3, line_pos3])

    source_pos_list = [source_pos1, source_pos2, source_pos3]
    psf_line = torch.empty((len(source_pos_list), len(wavelengths), kernel_size[1], kernel_size[0]))

    for s_id, source_pos in enumerate(source_pos_list):
        d = sp_system.extract_hexapolar_dir(N, source_pos, max_angle)
        
        for w_id, wavelength in enumerate(tqdm(wavelengths)):
            ps = sp_system.trace_psf_from_point_source(angles = None, x_pos = source_pos[0], y_pos = source_pos[1], z_pos = 0., wavelength = wavelength,
                        show_rays = False, d = d, ignore_invalid = False, show_res = False)
            ps[:, 0] = ps[:, 0] - sp_system.system[0].pixel_size/2
            bins_i, bins_j, centroid = find_bins(ps, sp_system.system[0].pixel_size, kernel_size, same_grid = True, absolute_grid=True)

            hist_ps = torch.histogramdd(ps.flip(1), bins=(bins_j, bins_i), density=False).hist
            hist_ps /= hist_ps.sum()
            psf_line[s_id, w_id, :, :] = hist_ps.reshape(kernel_size[1], kernel_size[0])



    texture = scipy.io.loadmat("../processing_reconstruction/datasets_reconstruction/mst_datasets/cave_1024_28_train/scene109.mat")['img_expand'][:512,:512].astype('float32')
    texture_acq = torch.nn.functional.interpolate(torch.from_numpy(texture).unsqueeze(0), scale_factor=(1, oversample_acq), mode='bilinear', align_corners=True).squeeze()
    texture = torch.nn.functional.interpolate(torch.from_numpy(texture).unsqueeze(0), scale_factor=(1, oversample_truth), mode='bilinear', align_corners=True).squeeze()
    airy = compute_airy_disk(wavelengths, sp_system.system[-1].pixel_size, na=0.05, grid_size = 7, magnification = 2)
    
    ### Render acquisition

    mask = np.zeros((512, 512), dtype=np.float32)
    mask[:, 255] = 1

    texture_acq = np.multiply(texture_acq, mask[:,:,np.newaxis]).float().to(device)    
    #texture_acq = torch.from_numpy(texture_acq).float().to(device)

    z0 = torch.tensor([sp_system.system[-1].d_sensor*torch.cos(sp_system.system[-1].theta_y*np.pi/180) + sp_system.system[-1].origin[-1] + sp_system.system[-1].shift[-1]]).item()

    airy_acq = compute_airy_disk(wavelengths_acq, sp_system.system[-1].pixel_size, na=0.05, grid_size = 7, magnification = 2)
    texture_acq = torch.nn.functional.conv2d(texture_acq.unsqueeze(0).permute(0, 3, 1, 2), airy_acq.float(), padding = airy_acq.shape[-1]//2, groups=wavelengths_acq.shape[0]).squeeze()
    texture_acq = texture_acq.permute(1, 2, 0)


    image = sp_system.render(wavelengths=wavelengths_acq, nb_rays=nb_rays, z0=z0,
                    texture=texture_acq, numerical_aperture=0.05, plot=False).flip(0)


    colors = [(0, 0, 0), (0.45, 0.45, 0.45), (0.75, 0.75, 0.75), (0.9, 0.9, 0.9), (1, 1, 1)]
    custom_gray_cmap = LinearSegmentedColormap.from_list("Custom", colors, N=2000)
    plt.figure(figsize=(32, 18), dpi=60)
    plt.axis('off')
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
        hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.imshow(image.sum(-1), cmap=custom_gray_cmap)
    if sp_system.save_dir is not None:
        plt.savefig(sp_system.save_dir + "acquisition_rendering.svg", format='svg', bbox_inches = 'tight', pad_inches = 0)


    psf_line = torch.nn.functional.conv2d(psf_line, airy.float(), padding = airy.shape[-1]//2, groups=wavelengths.shape[0]).squeeze()

    dispersed_texture_pixel = torch.zeros(psf_line.shape[0], sp_system.system[-1].film_size[0])

    texture = texture[:, 255, :]
    dispersed_texture_pixel = torch.zeros(psf_line.shape[0], sp_system.system[-1].film_size[0])
    temp = torch.zeros(psf_line.shape[2], psf_line.shape[3])
    fake_texture = torch.zeros(psf_line.shape[2], psf_line.shape[3])
    for pos in range(psf_line.shape[0]):
        sub_texture = texture[128*(pos+1)+1-10:128*(pos+1)+1+10, :]
        for i in range(sub_texture.shape[0]):
            for j in range(sub_texture.shape[1]):
                accu = sub_texture[i, j]*psf_line[pos, j, :, :]
                temp += torch.roll(accu, i - sub_texture.shape[0]//2, 0)/oversample_truth

    dispersed_texture_pixel[0, :] = temp[128-10:128+10, :].sum(0)
    dispersed_texture_pixel[1, :] = temp[256-10:256+10, :].sum(0)
    dispersed_texture_pixel[2, :] = temp[384-10:384+10, :].sum(0)

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
    plt.plot(list(range(270,380)), dispersed_texture_pixel[0, 270:380]/dispersed_texture_pixel[0, 270:380].max(), linewidth=10, color="#005673", linestyle='--')
    plt.plot(list(range(270,380)), dispersed_texture_pixel[1, 270:380]/dispersed_texture_pixel[1, 270:380].max(), linewidth=10, color="#ab7b4e", linestyle='--', label='_nolegend_')
    plt.plot(list(range(270,380)), dispersed_texture_pixel[2, 270:380]/dispersed_texture_pixel[2, 270:380].max(), linewidth=10, color="#9d9e9f", linestyle='--',label='_nolegend_')
    plt.plot(list(range(270,380)), torch.sum(torch.mean(image[128-20:128+20,270:380], dim=0), dim=-1)/torch.sum(torch.mean(image[128-20:128+20,270:380], dim=0), dim=-1).max(), linewidth=10, color="#008673")
    plt.plot(list(range(270,380)), torch.sum(torch.mean(image[256-20:256+20,270:380], dim=0), dim=-1)/torch.sum(torch.mean(image[256-20:256+20,270:380], dim=0), dim=-1).max(), linewidth=10, color="#eb7b4e")
    plt.plot(list(range(270,380)), torch.sum(torch.mean(image[384-20:384+20,270:380], dim=0), dim=-1)/torch.sum(torch.mean(image[384-20:384+20,270:380], dim=0), dim=-1).max(), linewidth=10, color="#bdbebf")
    ax.set_xlabel("Pixels [px]", fontsize=90)
    plt.legend(["Ground truth spectrum", "Acquisition"])
    ax.tick_params(axis='both', which='major', labelsize=90, width=5, length=20)
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    if sp_system.save_dir is not None:
        plt.savefig(sp_system.save_dir + "spectrum_comparison.svg", format='svg', bbox_inches = 'tight', pad_inches = 0)
    plt.show()