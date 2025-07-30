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

import argparse

from matplotlib.colors import ListedColormap, LinearSegmentedColormap

device = 'cpu'  # Change device as wished

figure_number = 6 # Change figure to generate

parser = argparse.ArgumentParser(description='Run a specific use case for the CASSI system.')
parser.add_argument('-f', '--figure', type=int, default=1, help='Figure to generate: can be 1, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16, 17 or 18')
args = parser.parse_args()

list_valid = [1, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18]

figure_number = args.figure

if figure_number not in list_valid:
    raise ValueError(f"Invalid figure number: {figure_number}. Valid options are: {list_valid}")


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

elif (figure_number == 4) or (figure_number == 6):
    sp_system = HSSystem(config_file_path="../system_specs/system_single.yml", device = device)
    msp_system = HSSystem(config_file_path="../system_specs/system_singlemis.yml", device = device)
    ap_system = HSSystem(config_file_path="../system_specs/system_amici.yml", device = device)
    map_system = HSSystem(config_file_path="../system_specs/system_amicimis.yml", device = device)

    sp_system.compare_spot_zemax(path_compare='./data_zemax/single_prism_aligned/')

    msp_system.compare_spot_zemax(path_compare='./data_zemax/single_prism_misaligned/')

    ap_system.compare_spot_zemax(path_compare='./data_zemax/amici_prism_aligned/')

    map_system.compare_spot_zemax(path_compare='./data_zemax/amici_prism_misaligned/')

elif (figure_number == 5) or (figure_number == 12):
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

    file_name = "./data_zemax/single_prism_misaligned/ray_positions_wavelength_W1_field_F1.txt"

    params = [[source_pos_list[i], w_list[j], extract_positions(file_name.replace('W1', f'W{j+1}').replace('F1', f'F{i+1}'))]
                for i in range(len(source_pos_list)) for j in range(len(w_list))]

    print(len(params))
    msp_system.compare_psf(N, params, max_angle, pixel_size, kernel_size = 11, show_rays = False, show_res = False)

elif (figure_number == 11):
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

    file_name = "./data_zemax/single_prism_aligned/ray_positions_wavelength_W1_field_F1.txt"

    params = [[source_pos_list[i], w_list[j], extract_positions(file_name.replace('W1', f'W{j+1}').replace('F1', f'F{i+1}'))]
                for i in range(len(source_pos_list)) for j in range(len(w_list))]


    sp_system.compare_psf(N, params, max_angle, pixel_size, kernel_size = 11, show_rays = False, show_res = False)

elif (figure_number == 13):
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

    file_name = "./data_zemax/amici_prism_aligned/ray_positions_wavelength_W1_field_F1.txt"

    params = [[source_pos_list[i], w_list[j], extract_positions(file_name.replace('W1', f'W{j+1}').replace('F1', f'F{i+1}'))]
                for i in range(len(source_pos_list)) for j in range(len(w_list))]


    ap_system.compare_psf(N, params, max_angle, pixel_size, kernel_size = 11, show_rays = False, show_res = False)

elif (figure_number == 14):
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

    file_name = "./data_zemax/amici_prism_misaligned/ray_positions_wavelength_W1_field_F1.txt"

    params = [[source_pos_list[i], w_list[j], extract_positions(file_name.replace('W1', f'W{j+1}').replace('F1', f'F{i+1}'))]
                for i in range(len(source_pos_list)) for j in range(len(w_list))]


    map_system.compare_psf(N, params, max_angle, pixel_size, kernel_size = 11, show_rays = False, show_res = False)

elif (figure_number == 3) or (figure_number == 16):
    sp_system = HSSystem(config_file_path="../system_specs/system_single.yml", device = device)
    ap_system = HSSystem(config_file_path="../system_specs/system_amici.yml", device = device)

    disp_amici = ap_system.central_positions_wavelengths(torch.linspace(450, 650, 28))[0][:,0]

    disp_single = sp_system.central_positions_wavelengths(torch.linspace(450, 650, 28))[0][:,0]

    print("Amici disp:", disp_amici.max() - disp_amici.min())
    print("Single disp:", disp_single.max() - disp_single.min())

    linear_spread = torch.linspace(-400, 430, 28)

    err_min = 1000
    pos_min = -1

    for pos in range(0, 800, 1):
        #temp_spread = torch.linspace(disp_amici.min()*1000 + pos, disp_amici.max()*1000 + pos, 28)
        temp_spread = torch.linspace(-400 + pos, 430 + pos, 28)
        err = torch.abs(temp_spread - 1000*disp_amici)
        errmin = err.mean()
        if errmin < err_min:
            err_min = errmin
            pos_min = pos
    
    #linear_spread_amici = torch.linspace(disp_amici.min()*1000 + pos_min, disp_amici.max()*1000 + pos_min, 28)
    linear_spread_amici = torch.linspace(-400 + pos_min, 430 + pos_min, 28)
    print("Shift amici:", pos_min)
    print("Err mean amici:", err_min)
    print("Err max amici: ", torch.abs(linear_spread_amici - 1000*disp_amici).max())

    err_min = 1000
    pos_min = -1

    for pos in range(0, 800, 1):
        temp_spread = torch.linspace(disp_single.min()*1000 + pos, disp_single.max()*1000 + pos, 28)
        #temp_spread = torch.linspace(-400 + pos, 430 + pos, 28)
        err = torch.abs(temp_spread - 1000*disp_single)
        errmin = err.mean()
        if errmin < err_min:
            err_min = errmin
            pos_min = pos
    
    linear_spread_single = torch.linspace(disp_single.min()*1000 + pos_min, disp_single.max()*1000 + pos_min, 28)
    #linear_spread_single = torch.linspace(-400 + pos_min, 430 + pos_min, 28)
    print("Shift single: ", pos_min)
    print("Err mean single: ", err_min)
    print("Err max single: ", torch.abs(linear_spread_single - 1000*disp_single).max())

    colors = plt.cm.tab10.colors

    # Compare the two systems
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'
    params = {'axes.labelsize': 90/2.5,'axes.titlesize':90/2.5, 'legend.fontsize': 90/2.5, 'xtick.labelsize': 70/2.5, 'ytick.labelsize': 70/2.5}
    matplotlib.rcParams.update(params)
    plt.rcParams.update(params)
    fig = plt.figure(figsize=(32/2.5, 18/2.5), dpi=60*2.5)
    ax = fig.add_subplot(111)
    plt.rcParams.update({'font.size': 90/2.5})
    plt.plot(torch.linspace(450, 650, 28), 1000*disp_amici, label='Amici (AP)', linewidth = 5, marker='o', markersize=10, c = colors[0])
    plt.plot(torch.linspace(450, 650, 28), 1000*disp_single, label='Single (SP)', linewidth = 5, marker='o', markersize=10, c = colors[1])
    #plt.plot(torch.linspace(450, 650, 28), [-400 + 20*i for i in range(28)], label='Usual linear spread')
    plt.plot(torch.linspace(450, 650, 28), linear_spread_amici, label='Corrected linear spread amici', linewidth = 5, linestyle='--', c =colors[0])
    plt.plot(torch.linspace(450, 650, 28), linear_spread_single, label='Corrected linear spread single', linewidth = 5, linestyle='--', c = colors[1])
    plt.legend()
    ax.set_xlabel("Wavelength [nm]", fontsize=90/2.5)
    ax.set_ylabel('Spreading [µm]', fontsize=90/2.5)
    ax.set_xticks(torch.arange(450, 650, 35).tolist() + [650])
    ax.tick_params(axis='both', which='major', labelsize=90/2.5, width=5/2.5, length=20/2.5)
    plt.grid("on")
    plt.savefig("/home/lpaillet/Documents/Codes/article-distorsions-dont-matter/round1/disp_w_lin.svg", format='svg', bbox_inches = 'tight', pad_inches = 0)

    

    fig = plt.figure(figsize=(32/2.5, 18/2.5), dpi=60*2.5)
    ax = fig.add_subplot(111)
    plt.rcParams.update({'font.size': 90/2.5})
    plt.plot(torch.linspace(450, 650, 28), 1000*disp_amici - linear_spread_amici, label='Amici (AP)', c= colors[0], linewidth = 5, marker='o', markersize=10)
    plt.legend()
    ax.set_xlabel("Wavelength [nm]", fontsize=90/2.5)
    ax.set_ylabel('Non-linearity [µm]', fontsize=90/2.5)
    ax.set_xticks(torch.arange(450, 650, 35).tolist() + [650])
    ax.set_ylim(-120, 50)
    ax.tick_params(axis='both', which='major', labelsize=90/2.5, width=5/2.5, length=20/2.5)
    plt.grid("on")
    plt.savefig("/home/lpaillet/Documents/Codes/article-distorsions-dont-matter/round1/ap_nl.svg", format='svg', bbox_inches = 'tight', pad_inches = 0)
    

    fig = plt.figure(figsize=(32/2.5, 18/2.5), dpi=60*2.5)
    ax = fig.add_subplot(111)
    plt.rcParams.update({'font.size': 90/2.5})
    plt.plot(torch.linspace(450, 650, 28), 1000*disp_single- linear_spread_single, label='Single (SP)', c = colors[1], linewidth = 5, marker='o', markersize=10)
    plt.legend()
    ax.set_xlabel("Wavelength [nm]", fontsize=90/2.5)
    ax.set_ylabel('Non-linearity [µm]', fontsize=90/2.5)
    ax.set_xticks(torch.arange(450, 650, 35).tolist() + [650])
    ax.set_ylim(-120, 50)
    ax.tick_params(axis='both', which='major', labelsize=90/2.5, width=5/2.5, length=20/2.5)
    plt.grid("on")
    plt.savefig("/home/lpaillet/Documents/Codes/article-distorsions-dont-matter/round1/sp_nl.svg", format='svg', bbox_inches = 'tight', pad_inches = 0)

    fig = plt.figure(figsize=(32/2.5, 18/2.5), dpi=60*2.5)
    ax = fig.add_subplot(111)
    plt.rcParams.update({'font.size': 90/2.5})
    plt.plot(torch.linspace(450, 650, 28), 1000*disp_amici - linear_spread_amici, label='Amici (AP)', c= colors[0], linewidth = 5, marker='o', markersize=10)
    plt.plot(torch.linspace(450, 650, 28), 1000*disp_single- linear_spread_single, label='Single (SP)', c = colors[1], linewidth = 5, marker='o', markersize=10)
    plt.legend()
    ax.set_xlabel("Wavelength [nm]", fontsize=90/2.5)
    ax.set_ylabel('Non-linearity [µm]', fontsize=90/2.5)
    ax.set_xticks(torch.arange(450, 650, 35).tolist() + [650])
    ax.set_ylim(-120, 50)
    ax.tick_params(axis='both', which='major', labelsize=90/2.5, width=5/2.5, length=20/2.5)
    plt.grid("on")
    plt.savefig("/home/lpaillet/Documents/Codes/article-distorsions-dont-matter/round1/both.svg", format='svg', bbox_inches = 'tight', pad_inches = 0)


    plt.show()

elif figure_number == 17:
    sp_system = HSSystem(config_file_path="../system_specs/system_single.yml", device = device)
    msp_system = HSSystem(config_file_path="../system_specs/system_singlemis.yml", device = device)
    ap_system = HSSystem(config_file_path="../system_specs/system_amici.yml", device = device)
    map_system = HSSystem(config_file_path="../system_specs/system_amicimis.yml", device = device)

    def create_keystone(ps):
        # keystone = {'450nm': np.zeros(9),
        #             '520nm': np.zeros(9),
        #             '650nm': np.zeros(9)}
        keystone = {'450nm': np.zeros(3),
                    '520nm': np.zeros(3),
                    '650nm': np.zeros(3)}
        #keystone['450nm'][:3] = ps[0][:3,1] - 2.5
        keystone['450nm'][0] = np.max(ps[0][:3,1]) - np.min(ps[0][:3,1])
        #keystone['450nm'][3:6] = ps[0][3:6,1] - 0.0
        keystone['450nm'][1] = np.max(ps[0][3:6,1]) - np.min(ps[0][3:6,1])
        #keystone['450nm'][6:] = ps[0][6:,1] - -2.5
        keystone['450nm'][2] = np.max(ps[0][6:,1]) - np.min(ps[0][6:,1])
        #keystone['520nm'][:3] = ps[1][:3,1] - 2.5
        keystone['520nm'][0] = np.max(ps[1][:3,1]) - np.min(ps[1][:3,1])
        #keystone['520nm'][3:6] = ps[1][3:6,1] - 0.0
        keystone['520nm'][1] = np.max(ps[1][3:6,1]) - np.min(ps[1][3:6,1])
        #keystone['520nm'][6:] = ps[1][6:,1] - -2.5
        keystone['520nm'][2] = np.max(ps[1][6:,1]) - np.min(ps[1][6:,1])
        #keystone['650nm'][:3] = ps[2][:3,1] - 2.5
        keystone['650nm'][0] = np.max(ps[2][:3,1]) - np.min(ps[2][:3,1])
        #keystone['650nm'][3:6] = ps[2][3:6,1] - 0.0
        keystone['650nm'][1] = np.max(ps[2][3:6,1]) - np.min(ps[2][3:6,1])
        #keystone['650nm'][6:] = ps[2][6:,1] - -2.5
        keystone['650nm'][2] = np.max(ps[2][6:,1]) - np.min(ps[2][6:,1])
        return keystone
    
    def create_smile(ps):
        smile = {'450nm': np.zeros(3),
                '520nm': np.zeros(3),
                '650nm': np.zeros(3)}
        smile['450nm'][0] = np.max(ps[0][::3,0]) - np.min(ps[0][::3,0])
        smile['450nm'][1] = np.max(ps[0][1::3,0]) - np.min(ps[0][1::3,0])
        smile['450nm'][2] = np.max(ps[0][2::3,0]) - np.min(ps[0][2::3,0])
        smile['520nm'][0] = np.max(ps[1][::3,0]) - np.min(ps[1][::3,0])
        smile['520nm'][1] = np.max(ps[1][1::3,0]) - np.min(ps[1][1::3,0])
        smile['520nm'][2] = np.max(ps[1][2::3,0]) - np.min(ps[1][2::3,0])
        smile['650nm'][0] = np.max(ps[2][::3,0]) - np.min(ps[2][::3,0])
        smile['650nm'][1] = np.max(ps[2][1::3,0]) - np.min(ps[2][1::3,0])
        smile['650nm'][2] = np.max(ps[2][2::3,0]) - np.min(ps[2][2::3,0])
        return smile
    
    def create_spectral_keystone(ps):
        keystone = {'left': 0.,
                    'center': 0.,
                    'right': 0.}
        all_heights_left = np.zeros(3)
        for i in range(3):
            all_heights_left[i] = np.max(ps[i][2::3,1]) - np.min(ps[i][2::3,1])
        all_heights_left = np.array(all_heights_left)
        keystone['left'] = np.max(all_heights_left) - np.min(all_heights_left)
        all_heights_center = np.zeros(3)
        for i in range(3):
            all_heights_center[i] = np.max(ps[i][1::3,1]) - np.min(ps[i][1::3,1])
        all_heights_center = np.array(all_heights_center)
        keystone['center'] = np.max(all_heights_center) - np.min(all_heights_center)
        all_heights_right = np.zeros(3)
        for i in range(3):
            all_heights_right[i] = np.max(ps[i][::3,1]) - np.min(ps[i][::3,1])
        all_heights_right = np.array(all_heights_right)
        keystone['right'] = np.max(all_heights_right) - np.min(all_heights_right)
        return keystone
    
    def create_max_keystone(ps):
        keystone = {'top': 0.,
                    'center': 0.,
                    'bottom': 0.}
        all_heights_top = np.zeros(9)
        for i in range(3):
            for j in range(3):
                all_heights_top[i*3+j] = ps[i][j,1]
        
        keystone['top'] = round((np.max(all_heights_top) - np.min(all_heights_top))*1000,1)
        all_heights_center = np.zeros(9)
        for i in range(3):
            for j in range(3):
                all_heights_center[i*3+j] = ps[i][j+3,1]
        keystone['center'] = round((np.max(all_heights_center) - np.min(all_heights_center))*1000,1)
        all_heights_bottom = np.zeros(9)
        for i in range(3):
            for j in range(3):
                all_heights_bottom[i*3+j] = ps[i][j+6,1]
        keystone['bottom'] = round((np.max(all_heights_bottom) - np.min(all_heights_bottom))*1000,1)
        return keystone
    
    def create_max_smile(smiles):
        smile = {'left': 0.,
                'center': 0.,
                'right': 0.}
        smile['left'] = round(max(smiles['450nm'][0], smiles['520nm'][0], smiles['650nm'][0])*1000,1)
        smile['center'] = round(max(smiles['450nm'][1], smiles['520nm'][1], smiles['650nm'][1])*1000,1)
        smile['right'] = round(max(smiles['450nm'][2], smiles['520nm'][2], smiles['650nm'][2])*1000,1)
        return smile


    wavelengths = torch.tensor([450, 520, 650])

    ps_sp = sp_system.plot_spot_less_points(9, 10*1e-3, wavelengths = wavelengths)
    #print("SP: ", ps_sp)
    keystones_sp = create_keystone(ps_sp)
    smiles_sp = create_smile(ps_sp)
    ps_msp = msp_system.plot_spot_less_points(9, 10*1e-3, wavelengths = wavelengths)
    #print("mSP: ", ps_msp)
    keystones_msp = create_keystone(ps_msp)
    smiles_msp = create_smile(ps_msp)
    ps_ap = ap_system.plot_spot_less_points(9, 10*1e-3, wavelengths = wavelengths)
    #print("AP: ", ps_ap)
    keystones_ap = create_keystone(ps_ap)
    smiles_ap = create_smile(ps_ap)
    ps_map = map_system.plot_spot_less_points(9, 10*1e-3, wavelengths = wavelengths)
    #print("mAP: ", ps_map) # right -> left, top -> bottom
    keystones_map = create_keystone(ps_map)
    smiles_map = create_smile(ps_map)

    spectral_ks_sp = create_spectral_keystone(ps_sp)
    spectral_ks_ap = create_spectral_keystone(ps_ap)

    print("======= Keystones =======")
    #print("SP: ", keystones_sp)
    print("mSP: ", keystones_msp)
    #print("AP: ", keystones_ap)
    print("mAP: ", keystones_map)

    print("======= Smiles =======")
    #print("SP: ", smiles_sp)
    print("mSP: ", smiles_msp)
    #print("AP: ", smiles_ap)
    print("mAP: ", smiles_map)

    print("======= Spectral keystones =======")
    print("SP: ", spectral_ks_sp)
    print("AP: ", spectral_ks_ap)

    print("======= Max keystones =======")
    max_ks_sp = create_max_keystone(ps_sp)
    max_ks_msp = create_max_keystone(ps_msp)
    max_ks_ap = create_max_keystone(ps_ap)
    max_ks_map = create_max_keystone(ps_map)
    print("SP: ", max_ks_sp)
    print("mSP: ", max_ks_msp)
    print("AP: ", max_ks_ap)
    print("mAP: ", max_ks_map)

    print("======= Max smiles =======")
    max_sm_sp = create_max_smile(smiles_sp)
    max_sm_msp = create_max_smile(smiles_msp)
    max_sm_ap = create_max_smile(smiles_ap)
    max_sm_map = create_max_smile(smiles_map)
    print("SP: ", max_sm_sp)    
    print("mSP: ", max_sm_msp)
    print("AP: ", max_sm_ap)
    print("mAP: ", max_sm_map)

    

elif figure_number == 18:
    sp_system = HSSystem(config_file_path="../system_specs/system_single.yml", device = device)
    msp_system = HSSystem(config_file_path="../system_specs/system_singlemis.yml", device = device)
    ap_system = HSSystem(config_file_path="../system_specs/system_amici.yml", device = device)
    map_system = HSSystem(config_file_path="../system_specs/system_amicimis.yml", device = device)

    system_name = 'sp'

    if system_name == 'sp':
        system = sp_system
    elif system_name == 'msp':
        system = msp_system
    elif system_name == 'ap':
        system = ap_system
    elif system_name == 'map':
        system = map_system

    #system = sp_system
    

    system.save_dir = "../images/"

    oversample = 4
    nb_rays = 20  # Adjust the number of rays as you wish

    id_scene = 109

    wavelengths = torch.linspace(450, 650, 28*oversample)

    #texture = scipy.io.loadmat(f"../processing_reconstruction/datasets_reconstruction/mst_datasets/cave_1024_28_train/scene109.mat")['img_expand'][:512,:512].astype('float32')
    texture = scipy.io.loadmat(f"../processing_reconstruction/datasets_reconstruction/mst_datasets/cave_1024_28_train/scene{id_scene}.mat")['img_expand'][:512,:512].astype('float32')
    texture_acq = torch.nn.functional.interpolate(torch.from_numpy(texture).unsqueeze(0), scale_factor=(1, oversample), mode='bilinear', align_corners=True).squeeze()

    mask = torch.load("../processing_reconstruction/mask.pt")

    texture_acq = np.multiply(texture_acq, mask[:,:,np.newaxis]).float().to(device)    

    z0 = torch.tensor([system.system[-1].d_sensor*torch.cos(system.system[-1].theta_y*np.pi/180) + system.system[-1].origin[-1] + system.system[-1].shift[-1]]).item()
    airy_acq = compute_airy_disk(wavelengths, sp_system.system[-1].pixel_size, na=0.05, grid_size = 7, magnification = 2)
    texture_acq = torch.nn.functional.conv2d(texture_acq.unsqueeze(0).permute(0, 3, 1, 2), airy_acq.float(), padding = airy_acq.shape[-1]//2, groups=wavelengths.shape[0]).squeeze()
    texture_acq = texture_acq.permute(1, 2, 0)

    image = system.render(wavelengths=wavelengths, nb_rays=nb_rays, z0=z0,
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
    if system.save_dir is not None:
        plt.savefig(system.save_dir + f"acquisition_rendering{id_scene}_{system_name}.svg", format='svg', bbox_inches = 'tight', pad_inches = 0)

elif (figure_number == 8) or (figure_number == 15):
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

    # torch.save(image, f"/home/lpaillet/Documents/Codes/article-distorsions-dont-matter/round1/fig14_ov{oversample_acq}_supp.pt")

    # image = torch.load(f"/home/lpaillet/Documents/Codes/article-distorsions-dont-matter/round1/fig14_ov{oversample_acq}_supp.pt")

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
    plt.savefig(f"/home/lpaillet/Téléchargements/figure6_acq_{int(28*oversample_acq)}.svg", format='svg', bbox_inches = 'tight', pad_inches = 0)
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
    # plt.plot(list(range(270,380)), torch.sum(torch.mean(image[128-20:128+20,270:380], dim=0), dim=-1)/torch.sum(torch.mean(image[128-20:128+20,270:380], dim=0), dim=-1).max(), linewidth=10, color="#008673")
    # plt.plot(list(range(270,380)), torch.sum(torch.mean(image[256-20:256+20,270:380], dim=0), dim=-1)/torch.sum(torch.mean(image[256-20:256+20,270:380], dim=0), dim=-1).max(), linewidth=10, color="#eb7b4e")
    # plt.plot(list(range(270,380)), torch.sum(torch.mean(image[384-20:384+20,270:380], dim=0), dim=-1)/torch.sum(torch.mean(image[384-20:384+20,270:380], dim=0), dim=-1).max(), linewidth=10, color="#bdbebf")
    factor0 = torch.mean(dispersed_texture_pixel[0, 270:380])/torch.mean(torch.sum(torch.mean(image[128-20:128+20,270:380], dim=0), dim=-1))
    factor1 = torch.mean(dispersed_texture_pixel[1, 270:380])/torch.mean(torch.sum(torch.mean(image[256-20:256+20,270:380], dim=0), dim=-1))
    factor2 = torch.mean(dispersed_texture_pixel[2, 270:380])/torch.mean(torch.sum(torch.mean(image[384-20:384+20,270:380], dim=0), dim=-1))
    plt.plot(list(range(270,380)), torch.sum(torch.mean(image[128-20:128+20,270:380], dim=0), dim=-1)/(dispersed_texture_pixel[0, 270:380].max()/factor0), linewidth=10, color="#008673")
    plt.plot(list(range(270,380)), torch.sum(torch.mean(image[256-20:256+20,270:380], dim=0), dim=-1)/(dispersed_texture_pixel[1, 270:380].max()/factor1), linewidth=10, color="#eb7b4e")
    plt.plot(list(range(270,380)), torch.sum(torch.mean(image[384-20:384+20,270:380], dim=0), dim=-1)/(dispersed_texture_pixel[2, 270:380].max()/factor2), linewidth=10, color="#bdbebf")
    ax.set_xlabel("Pixels [px]", fontsize=90)
    ax.set_ylim(-0.05, 1.05)
    plt.legend(["Ground truth spectrum", "Acquisition"])
    ax.tick_params(axis='both', which='major', labelsize=90, width=5, length=20)
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    sp_system.save_dir = "/home/lpaillet/Téléchargements/"
    if sp_system.save_dir is not None:
        plt.savefig(sp_system.save_dir + f"figure6_spectrum_{int(28*oversample_acq)}.svg", format='svg', bbox_inches = 'tight', pad_inches = 0)
        #plt.savefig(sp_system.save_dir + "spectrum_comparison.svg", format='svg', bbox_inches = 'tight', pad_inches = 0)
    
    # fig = plt.figure(figsize=(32, int(18*179.668*(3/10)/100.682)), dpi=60)
    # ax = fig.add_subplot(111)
    # plt.rcParams.update({'font.size': 90})
    # plt.plot(list(range(270,380)), dispersed_texture_pixel[0, 270:380]/dispersed_texture_pixel[0, 270:380].max(), linewidth=10, color="#005673", linestyle='--')
    # plt.plot(list(range(270,380)), torch.sum(torch.mean(image[128-20:128+20,270:380], dim=0), dim=-1)/torch.sum(torch.mean(image[128-20:128+20,270:380], dim=0), dim=-1).max(), linewidth=10, color="#008673")
    # ax.set_xlabel("Pixels [px]", fontsize=90)
    # plt.legend(["Ground truth", "Acquisition"])
    # ax.tick_params(axis='both', which='major', labelsize=90, width=5, length=20)
    # ax.yaxis.set_label_position("right")
    # ax.yaxis.tick_right()
    # sp_system.save_dir = "/home/lpaillet/Documents/Codes/article-distorsions-dont-matter/images_round1/"
    # if sp_system.save_dir is not None:
    #     plt.savefig(sp_system.save_dir + f"figure6_spectrum_{int(28*oversample_acq)}_1.svg", format='svg', bbox_inches = 'tight', pad_inches = 0)
    #     #plt.savefig(sp_system.save_dir + "spectrum_comparison.svg", format='svg', bbox_inches = 'tight', pad_inches = 0)
    
    # fig = plt.figure(figsize=(32, int(18*179.668*(3/10)/100.682)), dpi=60)
    # ax = fig.add_subplot(111)
    # plt.rcParams.update({'font.size': 90})
    # plt.plot(list(range(270,380)), dispersed_texture_pixel[1, 270:380]/dispersed_texture_pixel[1, 270:380].max(), linewidth=10, color="#ab7b4e", linestyle='--')
    # plt.plot(list(range(270,380)), torch.sum(torch.mean(image[256-20:256+20,270:380], dim=0), dim=-1)/torch.sum(torch.mean(image[256-20:256+20,270:380], dim=0), dim=-1).max(), linewidth=10, color="#eb7b4e")
    # ax.set_xlabel("Pixels [px]", fontsize=90)
    # plt.legend(["Ground truth", "Acquisition"])
    # ax.tick_params(axis='both', which='major', labelsize=90, width=5, length=20)
    # ax.yaxis.set_label_position("right")
    # ax.yaxis.tick_right()
    # sp_system.save_dir = "/home/lpaillet/Documents/Codes/article-distorsions-dont-matter/images_round1/"
    # if sp_system.save_dir is not None:
    #     plt.savefig(sp_system.save_dir + f"figure6_spectrum_{int(28*oversample_acq)}_2.svg", format='svg', bbox_inches = 'tight', pad_inches = 0)
        
    # fig = plt.figure(figsize=(32, int(18*179.668*(3/10)/100.682)), dpi=60)
    # ax = fig.add_subplot(111)
    # plt.rcParams.update({'font.size': 90})
    # plt.plot(list(range(270,380)), dispersed_texture_pixel[2, 270:380]/dispersed_texture_pixel[2, 270:380].max(), linewidth=10, color="#9d9e9f", linestyle='--')
    # plt.plot(list(range(270,380)), torch.sum(torch.mean(image[384-20:384+20,270:380], dim=0), dim=-1)/torch.sum(torch.mean(image[384-20:384+20,270:380], dim=0), dim=-1).max(), linewidth=10, color="#bdbebf")
    # ax.set_xlabel("Pixels [px]", fontsize=90)
    # plt.legend(["Ground truth", "Acquisition"])
    # ax.tick_params(axis='both', which='major', labelsize=90, width=5, length=20)
    # ax.yaxis.set_label_position("right")
    # ax.yaxis.tick_right()
    # sp_system.save_dir = "/home/lpaillet/Documents/Codes/article-distorsions-dont-matter/images_round1/"
    # if sp_system.save_dir is not None:
    #     plt.savefig(sp_system.save_dir + f"figure6_spectrum_{int(28*oversample_acq)}_3.svg", format='svg', bbox_inches = 'tight', pad_inches = 0)
    
    plt.show()

elif figure_number == 20:

    def calculate_fwhm(x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        
        peak_idx = np.argmax(y)
        y_max = y[peak_idx]
        half_max = y_max / 2.0
        
        if half_max <= 0:
            return 0.0
        
        # Left crossing
        left_y = y[:peak_idx]
        left_below = np.where(left_y < half_max)[0]
        if len(left_below) == 0:
            x_left = x[0]
        else:
            left_idx = left_below[-1]
            x0, x1 = x[left_idx], x[left_idx + 1]
            y0, y1 = y[left_idx], y[left_idx + 1]
            x_left = x0 + (half_max - y0) * (x1 - x0) / (y1 - y0)
        
        # Right crossing
        right_y = y[peak_idx + 1:]
        right_below = np.where(right_y < half_max)[0]
        if len(right_below) == 0:
            x_right = x[-1]
        else:
            right_idx_in_right = right_below[0]
            right_idx = peak_idx + 1 + right_idx_in_right
            x0, x1 = x[right_idx - 1], x[right_idx]
            y0, y1 = y[right_idx - 1], y[right_idx]
            x_right = x0 + (half_max - y0) * (x1 - x0) / (y1 - y0)
        
        return x_right - x_left

    pos = 2 # 0: center, 1: top, 2: right

    sp_system = HSSystem(config_file_path="../system_specs/system_single.yml", device = device)
    #sp_system.system[-1].pixel_size = 1e-3
    #sp_system.system[0].pixel_size = 100e-3
    sp_system.system[-1].pixel_size = 1e-3
    sp_system.system[0].pixel_size = 10e-3
    if pos == 2:
        o = torch.zeros((1,1,3))
        o[0,0,0] = 245*10e-3
        sp_displacement = sp_system.central_positions_wavelengths(torch.linspace(450, 650, 28), o=o)[0][:,0]
        #sp_displacement = sp_system.central_positions_wavelengths(torch.linspace(450, 650, 28))[0][:,0]
        #sp_displacement += -245*10e-3
    elif pos == 1:
        o = torch.zeros((1,1,3))
        o[0,0,1] = -245*10e-3
        sp_displacement = sp_system.central_positions_wavelengths(torch.linspace(450, 650, 28), o = o)[0][:,0]
    else:
        sp_displacement = sp_system.central_positions_wavelengths(torch.linspace(450, 650, 28))[0][:,0]
    print(sp_displacement)
    basic_shift = sp_system.system[-1].shift[0].clone()
    
    sp_system.system[-1].shift[0] = basic_shift + sp_displacement[0]
    if pos == 1:
        sp_system.system[-1].shift[1] += 230*10e-3
    #sp_system.system[-1].shift[0] += 0.42
    sp_system.system[-1].film_size = (80, 64)
    sp_system.system[-1].update()
    wavelengths = torch.linspace(450, 650, 28)
    #wavelengths = [torch.tensor([650.])]
    #sp_system.system[-1].shift[0] += 0.82
    #wavelengths = [torch.tensor([531.48])]
    nb_rays = 1000
    texture = torch.ones((512, 512, 1), dtype=torch.float32)
    mask = torch.zeros((512, 512), dtype=torch.float32)
    if pos ==2:
        mask[:, 10] = 1
    else:
        mask[:, 255] = 1
    texture = texture*mask[:,:,None]

    z0 = torch.tensor([sp_system.system[-1].d_sensor*torch.cos(sp_system.system[-1].theta_y*np.pi/180) + sp_system.system[-1].origin[-1] + sp_system.system[-1].shift[-1]]).item()

    FWHMs = torch.zeros(len(wavelengths)) - 1

    for w_id, wavelength in enumerate(wavelengths):
        image = sp_system.render(wavelengths=torch.tensor([wavelength]), nb_rays=nb_rays, z0=z0, texture=texture, numerical_aperture=0.05, plot=False).flip(0)
        print(f"Processing wavelength {wavelength.item()} nm")
        plt.close('all')
        print("Shift: ", sp_system.system[0].shift[0])
        if w_id < len(wavelengths) -1:
            sp_system.system[0].shift[0] = basic_shift + sp_displacement[w_id+1]
            sp_system.system[0].update()

        from scipy import optimize
        def gaussian(x, amplitude, mean, stddev):
            return amplitude * np.exp(-((x - mean) / 4 / stddev)**2)
        
        #amplitude = max(image[sp_system.system[0].film_size[1]//2, :, 0].cpu().numpy())
        amplitude = max(image[10, :, 0].cpu().numpy())
        amplitude = 1
        #mean = image[128, :, 0].cpu().numpy().argmax()

        mean = np.mean(np.where(image[sp_system.system[0].film_size[1]//2, :, 0].cpu().numpy() > 1e-5)[0])
        #mean = np.mean(np.where(image[10, :, 0].cpu().numpy() > 1e-5)[0])

        def gaussian(x, stddev):
            
            return amplitude * np.exp(-1/2 * ((x - mean) / (stddev))**2)
        
        #popt, _ = optimize.curve_fit(gaussian, [i for i in range(sp_system.system[0].film_size[0])], image[sp_system.system[0].film_size[1]//2, :, 0].cpu().numpy()/max(image[sp_system.system[0].film_size[1]//2, :, 0].cpu().numpy()))
        #popt, _ = optimize.curve_fit(gaussian, [i for i in range(sp_system.system[0].film_size[0])], image[10, :, 0].cpu().numpy()/max(image[10, :, 0].cpu().numpy()))
        
        #FWHMs[w_id] = 2*np.sqrt(2*np.log(2))*popt[0] * sp_system.system[0].pixel_size * 1e3
        FWHMs[w_id] = calculate_fwhm([i for i in range(sp_system.system[0].film_size[0])], image[sp_system.system[0].film_size[1]//2, :, 0].cpu().numpy()/max(image[sp_system.system[0].film_size[1]//2, :, 0].cpu().numpy()))
        print("FWHM: ", FWHMs[w_id])

        # plt.figure()
        # plt.imshow(image.sum(-1))
        # plt.figure()
        # plt.plot(image[sp_system.system[0].film_size[1]//2, :, :]/image[sp_system.system[0].film_size[1]//2, :, :].max(), linewidth=2, color="red")
        # #plt.plot(image[10, :, :]/image[10, :, :].max(), linewidth=2, color="red")
        # #plt.plot(gaussian(np.array([i for i in range(sp_system.system[0].film_size[0])]), *popt), linewidth=2, color="blue")
        # plt.show()

    # plt.figure()
    # plt.imshow(image.sum(-1))
    # plt.figure()
    # plt.plot(image[128, :, :]/image[128, :, :].max(), linewidth=2, color="red")
    # plt.plot(gaussian(np.array([i for i in range(650)]), *popt), linewidth=2, color="blue")
    # plt.show()

    if pos == 0:
        torch.save(FWHMs, "/home/lpaillet/Documents/Codes/DiffCassiSim/simulator_examples/FWHM_single.pt")
    elif pos == 1:
        torch.save(FWHMs, "/home/lpaillet/Documents/Codes/DiffCassiSim/simulator_examples/FWHM_single_top.pt")
    elif pos == 2:
        torch.save(FWHMs, "/home/lpaillet/Documents/Codes/DiffCassiSim/simulator_examples/FWHM_single_right.pt")
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'
    params = {'axes.labelsize': 90/2.5,'axes.titlesize':90/2.5, 'legend.fontsize': 90/2.5, 'xtick.labelsize': 70/2.5, 'ytick.labelsize': 70/2.5}
    matplotlib.rcParams.update(params)
    plt.rcParams.update(params)
    fig = plt.figure(figsize=(32/2.5, 18/2.5), dpi=60*2.5)
    ax = fig.add_subplot(111)
    plt.rcParams.update({'font.size': 90/2.5})
    plt.plot(wavelengths, FWHMs, linewidth=10, color=plt.cm.tab10.colors[1], label='FWHM')
    plt.legend()
    ax.set_xlabel("Wavelength [nm]", fontsize=90/2.5)
    ax.set_ylabel('FWHM [µm]', fontsize=90/2.5)
    ax.tick_params(axis='both', which='major', labelsize=90/2.5, width=5/2.5, length=20/2.5)
    plt.grid("on")
    if pos == 0:
        plt.savefig("/home/lpaillet/Documents/Codes/article-distorsions-dont-matter/round1/fwhm_sp.svg", format='svg', bbox_inches = 'tight', pad_inches = 0)
    elif pos == 1:
        plt.savefig("/home/lpaillet/Documents/Codes/article-distorsions-dont-matter/round1/fwhm_sp_top.svg", format='svg', bbox_inches = 'tight', pad_inches = 0)
    elif pos == 2:
        plt.savefig("/home/lpaillet/Documents/Codes/article-distorsions-dont-matter/round1/fwhm_sp_right.svg", format='svg', bbox_inches = 'tight', pad_inches = 0)
    plt.show()

    ap_system = HSSystem(config_file_path="../system_specs/system_amici.yml", device = device)
    ap_system.system[-1].pixel_size = 1e-3
    ap_system.system[0].pixel_size = 10e-3

    if pos == 2:
        o = torch.zeros((1,1,3))
        o[0,0,0] = 245*10e-3
        ap_displacement = ap_system.central_positions_wavelengths(torch.linspace(450, 650, 28), o=o)[0][:,0]
        #ap_displacement = ap_system.central_positions_wavelengths(torch.linspace(450, 650, 28))[0][:,0]
        #ap_displacement += -245*10e-3
    elif pos == 1:
        o = torch.zeros((1,1,3))
        o[0,0,1] = -245*10e-3
        ap_displacement = ap_system.central_positions_wavelengths(torch.linspace(450, 650, 28), o = o)[0][:,0]

    else:
        ap_displacement = ap_system.central_positions_wavelengths(torch.linspace(450, 650, 28))[0][:,0]
    print(ap_displacement)
    #ap_displacement = [ap_displacement[-2]]
    basic_shift = ap_system.system[-1].shift[0].clone()
    print("Prev shift: ", ap_system.system[-1].shift[0])
    ap_system.system[-1].shift[0] = basic_shift + ap_displacement[0]

    if pos == 1:
        ap_system.system[-1].shift[1] += 230*10e-3

    ap_system.system[-1].film_size = (80, 64)
    print("New shift: ", ap_system.system[-1].shift[0])
    ap_system.update_system()
    wavelengths = torch.linspace(450, 650, 28)
    #wavelengths = [wavelengths[-2]]
    nb_rays = 1000
    texture = torch.ones((512, 512, 1), dtype=torch.float32)

    mask = torch.zeros((512, 512), dtype=torch.float32)
    if pos == 2:
        mask[:, 10] = 1
    else:
        mask[:, 255] = 1
    texture = texture*mask[:,:,None]

    z0 = torch.tensor([ap_system.system[-1].d_sensor*torch.cos(ap_system.system[-1].theta_y*np.pi/180) + ap_system.system[-1].origin[-1] + ap_system.system[-1].shift[-1]]).item()

    FWHMs = torch.zeros(len(wavelengths)) - 1

    for w_id, wavelength in enumerate(wavelengths):
        print(f"Processing wavelength {wavelength.item()} nm")
        image = ap_system.render(wavelengths=torch.tensor([wavelength]), nb_rays=nb_rays, z0=z0, texture=texture, numerical_aperture=0.05, plot=False).flip(0)
        plt.close('all')
        print("Shift: ", ap_system.system[0].shift[0])
        if w_id < len(wavelengths) - 1:
            ap_system.system[0].shift[0] = basic_shift + ap_displacement[w_id+1]
            ap_system.system[0].update()

        from scipy import optimize
        def gaussian(x, amplitude, mean, stddev):
            return amplitude * np.exp(-((x - mean) / 4 / stddev)**2)
        
        #amplitude = max(image[ap_system.system[0].film_size[1]//2, :, 0].cpu().numpy())
        amplitude = max(image[10, :, 0].cpu().numpy())
        amplitude = 1
        #mean = image[128, :, 0].cpu().numpy().argmax()

        mean = np.mean(np.where(image[ap_system.system[0].film_size[1]//2, :, 0].cpu().numpy() > 1e-5)[0])
        mean = np.argmax(image[ap_system.system[0].film_size[1]//2, :, 0].cpu().numpy())
        print("Mean: ", mean)
        #mean = np.mean(np.where(image[10, :, 0].cpu().numpy() > 1e-5)[0])

        def gaussian(x, stddev):
            
            return amplitude * np.exp(-1/2 * ((x - mean) / (stddev))**2)
        
        #popt, _ = optimize.curve_fit(gaussian, [i for i in range(ap_system.system[0].film_size[0])], image[ap_system.system[0].film_size[1]//2, :, 0].cpu().numpy()/max(image[ap_system.system[0].film_size[1]//2, :, 0].cpu().numpy()))
        #popt, _ = optimize.curve_fit(gaussian, [i for i in range(ap_system.system[0].film_size[0])], image[10, :, 0].cpu().numpy()/max(image[10, :, 0].cpu().numpy()))
        
        #FWHMs[w_id] = 2*np.sqrt(2*np.log(2))*popt[0] * ap_system.system[0].pixel_size * 1e3
        FWHMs[w_id] = calculate_fwhm([float(i) for i in range(ap_system.system[0].film_size[0])], image[ap_system.system[0].film_size[1]//2, :, 0].cpu().numpy()/max(image[ap_system.system[0].film_size[1]//2, :, 0].cpu().numpy())) * ap_system.system[0].pixel_size * 1e3
        print("FWHM: ", FWHMs[w_id])

        # plt.figure()
        # plt.imshow(image.sum(-1))
        # plt.figure()
        # plt.plot(image[ap_system.system[0].film_size[1]//2, :, :]/image[ap_system.system[0].film_size[1]//2, :, :].max(), linewidth=2, color="red")
        # #plt.plot(image[10, :, :]/image[10, :, :].max(), linewidth=2, color="red")
        # # plt.plot(gaussian(np.array([i for i in range(ap_system.system[0].film_size[0])]), *popt), linewidth=2, color="blue")
        # plt.show()

    # plt.figure()
    # plt.imshow(image.sum(-1))
    # plt.figure()
    # plt.plot(image[ap_system.system[0].film_size[1]//2, :, :]/image[ap_system.system[0].film_size[1]//2, :, :].max(), linewidth=2, color="red")
    # plt.plot(gaussian(np.array([i for i in range(ap_system.system[0].film_size[0])]), *popt), linewidth=2, color="blue")
    # plt.show()

    if pos == 0:
        torch.save(FWHMs, "/home/lpaillet/Documents/Codes/DiffCassiSim/simulator_examples/FWHM_amici.pt")
    elif pos == 1:
        torch.save(FWHMs, "/home/lpaillet/Documents/Codes/DiffCassiSim/simulator_examples/FWHM_amici_top.pt")
    elif pos == 2:
        torch.save(FWHMs, "/home/lpaillet/Documents/Codes/DiffCassiSim/simulator_examples/FWHM_amici_right.pt")
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'
    params = {'axes.labelsize': 90/2.5,'axes.titlesize':90/2.5, 'legend.fontsize': 90/2.5, 'xtick.labelsize': 70/2.5, 'ytick.labelsize': 70/2.5}
    matplotlib.rcParams.update(params)
    plt.rcParams.update(params)
    fig = plt.figure(figsize=(32/2.5, 18/2.5), dpi=60*2.5)
    ax = fig.add_subplot(111)
    plt.rcParams.update({'font.size': 90/2.5})
    plt.plot(wavelengths, FWHMs, linewidth=10, color=plt.cm.tab10.colors[0], label='FWHM')
    plt.legend()
    ax.set_xlabel("Wavelength [nm]", fontsize=90/2.5)
    ax.set_ylabel('FWHM [µm]', fontsize=90/2.5)
    ax.tick_params(axis='both', which='major', labelsize=90/2.5, width=5/2.5, length=20/2.5)
    plt.grid("on")
    if pos == 0:
        plt.savefig("/home/lpaillet/Documents/Codes/article-distorsions-dont-matter/round1/fwhm_ap.svg", format='svg', bbox_inches = 'tight', pad_inches = 0)
    elif pos == 1:
        plt.savefig("/home/lpaillet/Documents/Codes/article-distorsions-dont-matter/round1/fwhm_ap_top.svg", format='svg', bbox_inches = 'tight', pad_inches = 0)
    elif pos == 2:
        plt.savefig("/home/lpaillet/Documents/Codes/article-distorsions-dont-matter/round1/fwhm_ap_right.svg", format='svg', bbox_inches = 'tight', pad_inches = 0)
    plt.show()

    wavelengths = torch.linspace(450, 650, 28)
    if pos == 0:
        FWHMs_single = torch.load("/home/lpaillet/Documents/Codes/DiffCassiSim/simulator_examples/FWHM_single.pt")
        FWHMs_amici = torch.load("/home/lpaillet/Documents/Codes/DiffCassiSim/simulator_examples/FWHM_amici.pt")
    elif pos == 1:
        FWHMs_single = torch.load("/home/lpaillet/Documents/Codes/DiffCassiSim/simulator_examples/FWHM_single_top.pt")
        FWHMs_amici = torch.load("/home/lpaillet/Documents/Codes/DiffCassiSim/simulator_examples/FWHM_amici_top.pt")
    elif pos == 2:
        FWHMs_single = torch.load("/home/lpaillet/Documents/Codes/DiffCassiSim/simulator_examples/FWHM_single_right.pt")
        FWHMs_amici = torch.load("/home/lpaillet/Documents/Codes/DiffCassiSim/simulator_examples/FWHM_amici_right.pt")

    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'
    params = {'axes.labelsize': 90/2.5,'axes.titlesize':90/2.5, 'legend.fontsize': 90/2.5, 'xtick.labelsize': 70/2.5, 'ytick.labelsize': 70/2.5}
    matplotlib.rcParams.update(params)
    plt.rcParams.update(params)
    fig = plt.figure(figsize=(32/2.5, 18/2.5), dpi=60*2.5)
    ax = fig.add_subplot(111)
    plt.rcParams.update({'font.size': 90/2.5})
    plt.plot(wavelengths, FWHMs_amici, linewidth=5, color=plt.cm.tab10.colors[0], label='FWHM (AP)', marker='o', markersize=10)
    plt.plot(wavelengths, FWHMs_single, linewidth=5, color=plt.cm.tab10.colors[1], label='FWHM (SP)', marker='o', markersize=10)
    plt.legend()
    ax.set_xlabel("Wavelength [nm]", fontsize=90/2.5)
    ax.set_ylabel('FWHM [µm]', fontsize=90/2.5)
    ax.tick_params(axis='both', which='major', labelsize=90/2.5, width=5/2.5, length=20/2.5)
    plt.grid("on")
    if pos == 0:
        plt.savefig("/home/lpaillet/Documents/Codes/article-distorsions-dont-matter/round1/fwhm_full.svg", format='svg', bbox_inches = 'tight', pad_inches = 0)
    elif pos == 1:
        plt.savefig("/home/lpaillet/Documents/Codes/article-distorsions-dont-matter/round1/fwhm_full_top.svg", format='svg', bbox_inches = 'tight', pad_inches = 0)
    elif pos == 2:
        plt.savefig("/home/lpaillet/Documents/Codes/article-distorsions-dont-matter/round1/fwhm_full_right.svg", format='svg', bbox_inches = 'tight', pad_inches = 0)
    plt.show()