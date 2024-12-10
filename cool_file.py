import argparse
import torch

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
    output = torch.zeros(bs, n_lambda, H - max_min[0], W - max_min[1], device=inputs.device)
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
    output = torch.zeros(bs, n_lambda, H + max_min[0], W + max_min[1], device=inputs.device)
    abs_shift = rounded_disp - rounded_disp.min(dim=0).values
    for i in range(n_lambda):
        output[:, i, abs_shift[i, 0]: H + abs_shift[i, 0], abs_shift[i, 1]: W + abs_shift[i, 1]] = inputs[:, i, :, :]
    return output

def expand_wav(inputs, dispersion_pixels, n_lambda = 28):
    """
        Input [bs, H + disp[0], W + disp[1]]
        Output [bs, n_wav, H + disp[0], W + disp[1]]
    """
    bs,row,col = inputs.shape
    rounded_disp = dispersion_pixels.round().int() # Pixels to indices
    max_min = rounded_disp.max(dim=0).values - rounded_disp.min(dim=0).values
    
    abs_shift = rounded_disp - rounded_disp.min(dim=0).values
    
    output = torch.zeros(bs, n_lambda, row, col, device=inputs.device).float()#.cuda()
    for i in range(n_lambda):
        output[:, i, abs_shift[i, 0]: row - max_min[0] + abs_shift[i, 0], abs_shift[i, 1]: col - max_min[1] + abs_shift[i, 1]] = inputs[:, abs_shift[i, 0]: row - max_min[0] + abs_shift[i, 0], abs_shift[i, 1]: col - max_min[1] + abs_shift[i, 1]]
    return output

def shift_3d(inputs, dispersion_pixels):
    """
        Input [bs, n_wav, H + disp[0], W + disp[1]]
        Output [bs, n_wav, H + disp[0], W + disp[1]]
        Rolls the input along the row axis
    """
    [bs, nC, row, col] = inputs.shape
    rounded_disp = dispersion_pixels.round().int() # Pixels to indices    
    abs_shift = rounded_disp - rounded_disp.min(dim=0).values
    for i in range(nC):
        inputs[:,i,:,:] = torch.roll(inputs[:,i,:,:], shifts=int(abs_shift[i, 0]), dims=1) # Roll along row axis too
        inputs[:,i,:,:] = torch.roll(inputs[:,i,:,:], shifts=int(abs_shift[i, 1]), dims=2)
    return inputs


def shift_back_3d(inputs, dispersion_pixels):
    """
        Input [bs, n_wav, H + disp[0], W + disp[1]]
        Output [bs, n_wav, H + disp[0], W + disp[1]]
        Rolls input in the opposite direction
    """
    [bs, nC, row, col] = inputs.shape
    rounded_disp = dispersion_pixels.round().int() # Pixels to indices    
    abs_shift = rounded_disp - rounded_disp.min(dim=0).values
    for i in range(nC):
        inputs[:,i,:,:] = torch.roll(inputs[:,i,:,:], shifts=(-1)*int(abs_shift[i, 0]), dims=1) # Roll along row axis too
        inputs[:,i,:,:] = torch.roll(inputs[:,i,:,:], shifts=(-1)*int(abs_shift[i, 1]), dims=2)
    return inputs

def shift_back_mst(inputs, dispersion_pixels): 
    """
        Input [bs, n_wav, H + disp[0], W + disp[1]]
        Output [bs, n_wav, H, W]
    """
    bs, nC, H, W = inputs.shape
    down_sample = nC//28
    rounded_disp = dispersion_pixels.round().int()//down_sample # Pixels to indices
    rounded_disp = torch.nn.functional.interpolate(rounded_disp.unsqueeze(0).unsqueeze(0).float(),
                                                   size=(rounded_disp.shape[0]*down_sample, 2), mode='bilinear').squeeze().round().int()
    max_min = rounded_disp.max(dim=0).values - rounded_disp.min(dim=0).values
    output = torch.zeros(bs, nC, H - max_min[0], W - max_min[1], device=inputs.device)
    abs_shift = rounded_disp - rounded_disp.min(dim=0).values
    for i in range(nC):
        output[:, i, :, :] = inputs[:, i, abs_shift[i, 0]: H - max_min[0] + abs_shift[i, 0], abs_shift[i, 1]: W - max_min[1] + abs_shift[i, 1]]
    return output


cube_3d = torch.load("/home/lpaillet/Documents/Codes/DiffOptics/examples/test_28.pt")
print("cube:" ,cube_3d.shape)
cube_3d_bis = torch.zeros((256, 512, 28))
cube_3d_bis[:, :256, :] = cube_3d
cube_3d = cube_3d_bis
middle_pos = torch.tensor([128, 256])
texture_size = torch.tensor([256, 256])
dispersion_pixels = torch.tensor([[  0.0000, -16.5782], [  0.0000, -14.2779],[  0.0000, -12.1396],[  0.0000, -10.1470],[  0.0000,  -8.2867],[  0.0000,  -6.5461],[  0.0000,  -4.9144],[  0.0000,  -3.3823],[  0.0000,  -1.9408],[  0.0000,  -0.5835],[  0.0000,   0.6974],[  0.0000,   1.9074],[  0.0000,   3.0524],[  0.0000,   4.1366],[  0.0000,   5.1650],[  0.0000,   6.1412],[  0.0000,   7.0698],[  0.0000,   7.9524],[  0.0000,   8.7937],[  0.0000,   9.5950],[  0.0000,  10.3599],[  0.0000,  11.0909],[  0.0000,  11.7885],[  0.0000,  12.4566],[  0.0000,  13.0962],[  0.0000,  13.7087],[  0.0000,  14.2956],[  0.0000,  14.8592]])

input = extract_acq_from_cube(cube_3d, dispersion_pixels, middle_pos, texture_size)
label = torch.ones(28, 256, 256)
mask = torch.ones(256, 256)

from DGSMP.Simulation.Model import HSI_CS

""" model = HSI_CS(28, 4, dispersion_pixels)
input_here = input.unsqueeze(0)
out = model(input_here)
print("Input shape: ", input_here.shape)
print("Output shape: ", out.shape) """

from DWMT_main.simulation.train.model.DWMT import DWMT

""" model = DWMT(num_blocks=[2, 4, 6])
input_here = shift_back(input.unsqueeze(0), dispersion_pixels)
mask_here = mask.unsqueeze(0).repeat(28,1,1).unsqueeze(0)
out = model(input_here, mask_here)
print("Input shape: ", input_here.shape)
print("Mask shape: ", mask_here.shape)
print("Output shape: ", out.shape) """

from PADUT.simulation.train_code.architecture.padut import PADUT

""" model = PADUT(in_c=28, n_feat=28,nums_stages=5, dispersion_pixels=dispersion_pixels)
input_here = input.unsqueeze(0)
mask_here = shift(mask.unsqueeze(0).repeat(28,1,1).unsqueeze(0), dispersion_pixels)
mask_s = torch.sum(mask_here**2,1)
mask_s[mask_s==0] = 1
mask_here = (mask_here, mask_s)
out = model(input_here, mask_here)
print("Input shape: ", input_here.shape)
print("Mask shape: ", [m.shape for m in mask_here])
print("Output shape: ", out.shape) """

from RDLUF_MixS2.simulation.train_code.architecture.duf_mixs2 import DUF_MixS2
from RDLUF_MixS2.simulation.train_code.options import merge_duf_mixs2_opt

""" parser = argparse.ArgumentParser(description="HyperSpectral Image Reconstruction Toolbox")
parser = merge_duf_mixs2_opt(parser)
opt = parser.parse_known_args()[0]

model = DUF_MixS2(opt, dispersion_pixels)
input_here = input.unsqueeze(0)
mask_here = shift(mask.unsqueeze(0).repeat(28,1,1).unsqueeze(0), dispersion_pixels)
out, log_dict = model(input_here, mask_here)
print("Input shape: ", input_here.shape)
print("Mask shape: ", mask_here.shape)
print("Output shape: ", out.shape) """


from Cai_models.MST import MST

""" model = MST(dim=28, stage=2, num_blocks=[4, 7, 5], dispersion_pixels=dispersion_pixels)
input_here = shift_back(input.unsqueeze(0), dispersion_pixels)
mask_here = shift(mask.unsqueeze(0).repeat(28,1,1).unsqueeze(0), dispersion_pixels)
out = model(input_here, mask_here)
print("Input shape: ", input_here.shape)
print("Mask shape: ", mask_here.shape)
print("Output shape: ", out.shape) """

from Cai_models.DAUHST import DAUHST

model = DAUHST(num_iterations=9, start_size=[256, 256], dispersion_pixels=dispersion_pixels)
input_here = input.unsqueeze(0)
mask_here = shift(mask.unsqueeze(0).repeat(28,1,1).unsqueeze(0), dispersion_pixels)
mask_s = torch.sum(mask_here**2,1)
mask_s[mask_s==0] = 1
mask_here = (mask_here, mask_s)
print(input_here.shape)
out = model(input_here, mask_here)
print("Input shape: ", input_here.shape)
print("Mask shape: ", [m.shape for m in mask_here])
print("Output shape: ", out.shape)



