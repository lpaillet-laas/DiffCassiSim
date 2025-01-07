import torch
import numpy as np

import argparse
import yaml

from datetime import datetime
from models.DGSMP.Simulation.Model import HSI_CS
from models.PADUT.simulation.train_code.architecture.padut import PADUT
from models.RDLUF_MixS2.simulation.train_code.architecture.duf_mixs2 import DUF_MixS2
from models.RDLUF_MixS2.simulation.train_code.options import merge_duf_mixs2_opt
from models.Cai_models.MST import MST
from models.Cai_models.DAUHST import DAUHST


def load_yaml_config(file_path):
    """
    Load a YAML configuration file as a dictionary

    Args:
        file_path (str): path to the YAML configuration file

    Returns:
        dict: configuration dictionary
    """
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def save_config_file(config_file_name,config_file,result_directory):
    """
    Save a configuration file in a YAML file

    Args:
        config_file_name (str): name of the file
        config_file (dict): configuration file to save
        result_directory (str): path to the directory where the results will be stored

    """
    with open(result_directory + f"/{config_file_name}.yml", 'w') as file:
        yaml.safe_dump(config_file, file)

def model_generator(method, dispersion_pixels, mapping_cube=None, pretrained_model_path=None):
    if method == 'dgsmp':
        model = HSI_CS(Ch=28, stages=4, dispersion_pixels=dispersion_pixels, mapping_cube=mapping_cube)
    elif method == 'mst':
        model = MST(dim=28, stage=2, num_blocks=[4, 7, 5], dispersion_pixels=dispersion_pixels)
    elif method == 'padut':
        model = PADUT(in_c=28, n_feat=28,nums_stages=5, dispersion_pixels=dispersion_pixels, mapping_cube=mapping_cube)
    elif method == 'duf':
        parser = argparse.ArgumentParser(description="HyperSpectral Image Reconstruction Toolbox")
        parser = merge_duf_mixs2_opt(parser)
        opt = parser.parse_known_args()[0]

        model = DUF_MixS2(opt, dispersion_pixels, mapping_cube=mapping_cube)
    elif 'dauhst' in method:
        #num_iterations = int(method.split('_')[1][0])
        model = DAUHST(num_iterations=9, start_size=[512, 512], dispersion_pixels=dispersion_pixels, mapping_cube=mapping_cube)
    else:
        raise NotImplementedError(f'Method {method} is not defined')
    
    if pretrained_model_path is not None:
        # print(f'load model from {pretrained_model_path}')
        checkpoint = torch.load(pretrained_model_path, map_location=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        if ".pth" in pretrained_model_path:
            model.load_state_dict({k.replace('reconstruction_model.', ''): v for k, v in checkpoint.items()},
                                strict=False)
        else:            
            adjusted_state_dict = {key.replace('reconstruction_module.reconstruction_model.', '').replace('reconstruction_model.', ''): value
                                    for key, value in checkpoint['state_dict'].items()}
            # Filter out unexpected keys
            model_keys = set(model.state_dict().keys())
            filtered_state_dict = {k: v for k, v in adjusted_state_dict.items() if k in model_keys}
            model.load_state_dict(filtered_state_dict)
    return model


def extract_acq_from_cube(cube_3d, dispersion_pixels, middle_pos, texture_size):
        acq_2d = cube_3d.sum(-1)
        rounded_disp = dispersion_pixels.round().int() # Pixels to indices

        return acq_2d[middle_pos[0] - texture_size[0]//2 + rounded_disp[0,0]: middle_pos[0] + texture_size[0]//2 + rounded_disp[-1,0],
                        middle_pos[1] - texture_size[1]//2 + rounded_disp[0,1]: middle_pos[1] + texture_size[1]//2 + rounded_disp[-1,1]]

# def shift_back(inputs, dispersion_pixels, n_lambda = 28): 
#     """
#         Input [bs, H + disp[0], W + disp[1]]
#         Output [bs, n_wav, H, W]
#     """
#     bs, H, W = inputs.shape
#     rounded_disp = dispersion_pixels.round().int() # Pixels to indices
#     max_min = rounded_disp.max(dim=0).values - rounded_disp.min(dim=0).values
#     output = torch.zeros(bs, n_lambda, H - max_min[0], W - max_min[1])
#     abs_shift = rounded_disp - rounded_disp.min(dim=0).values
#     for i in range(n_lambda):
#         output[:, i, :, :] = inputs[:, abs_shift[i, 0]: H - max_min[0] + abs_shift[i, 0], abs_shift[i, 1]: W - max_min[1] + abs_shift[i, 1]]
#     return output

# def shift(inputs, dispersion_pixels, n_lambda = 28):
#     """
#         Input [bs, n_wav, H, W]
#         Output [bs, n_wav, H + disp[0], W + disp[1]]
#     """
#     bs, n_lambda, H, W = inputs.shape
#     rounded_disp = dispersion_pixels.round().int() # Pixels to indices
#     max_min = rounded_disp.max(dim=0).values - rounded_disp.min(dim=0).values
#     output = torch.zeros(bs, n_lambda, H + max_min[0], W + max_min[1])
#     abs_shift = rounded_disp - rounded_disp.min(dim=0).values
#     for i in range(n_lambda):
#         output[:, i, abs_shift[i, 0]: H + abs_shift[i, 0], abs_shift[i, 1]: W + abs_shift[i, 1]] = inputs[:, i, :, :]
#     return output
def shift_back(inputs, dispersion_pixels, mapping_cube, n_lambda = 28):
    """
        Input [bs, H + disp[0], W + disp[1]]
            Mapping cube [n_wav, H, W, 2]
        Output [bs, n_wav, H, W]
    """
    bs,row,col = inputs.shape    
    output = torch.zeros(mapping_cube.shape[:-1]).unsqueeze(0).repeat(bs, 1, 1, 1).float() # [bs, n_lambda, H, W]

    row_indices = mapping_cube[..., 0]
    col_indices = mapping_cube[..., 1]

    for b in range(bs):
        for i in range(n_lambda):
            output[b, i] = inputs[b, row_indices[i], col_indices[i]].rot90(2, (0, 1)).flip(0)
    
    return output

def shift(inputs, dispersion_pixels, n_lambda = 28):
    """
        Input [bs, n_wav, H, W]
        Output [bs, n_wav, H, W + +-||disp||]
    """
    bs, n_lambda, H, W = inputs.shape
    normed_disp = (torch.norm(dispersion_pixels, dim=1)*torch.sign(dispersion_pixels[:, 1])).round().int() # Pixels to indices
    max_min = (normed_disp.max() - normed_disp.min()).round().int().item()
    abs_shift = normed_disp - normed_disp.min()
    output = torch.zeros(bs, n_lambda, H, W + max_min)
    for i in range(n_lambda):
        output[:, i, :, abs_shift[i]: W + abs_shift[i]] = inputs[:, i, :, :]
    return output

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
