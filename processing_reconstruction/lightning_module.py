import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision
import numpy as np
from piqa import SSIM
from torch.utils.tensorboard import SummaryWriter
import io
import torchvision.transforms as transforms
from torchmetrics.image import PeakSignalNoiseRatio, SpectralAngleMapper
import os
from PIL import Image

import time

from utils import *
import sys
sys.path.append("../")
from main_class import HSSystem

import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap

class ReconstructionCASSI(L.LightningModule):

    def __init__(self, net_model_name, optics_model_file_path, rays_file = None, valid_file = None, oversample=1, nb_rays=15, mask_path = None, log_dir="tb_logs", reconstruction_checkpoint=None, train=True):
        super().__init__()

        self.net_model_name = net_model_name.lower()

        self.optics_model_file_path = optics_model_file_path

        self.optics = HSSystem(config_file_path = optics_model_file_path)

        self.oversample = oversample
        self.optics.wavelengths = torch.linspace(450, 650, 28*oversample)

        self.optics.middle_pos = (np.flip(np.array(self.optics.system[-1].film_size)) // 2).tolist()


        self.pixel_dispersed_28 = self.optics.central_positions_wavelengths(torch.linspace(450, 650, 28))[1]
        
        if "straight_simple" in log_dir:
            try:
                self.mapping_cube = torch.load(f"./mapping_cubes/mapping_cube_straight_simple_{optics_model_file_path.split('_')[-1][:-4]}.pt", map_location=self.device)
            except:
                self.mapping_cube = self.optics.create_simple_mapping(torch.linspace(450, 650, 28), shape_scene = [512, 512], remove_y = True)
                torch.save(self.mapping_cube, f"./mapping_cubes/mapping_cube_straight_simple_{optics_model_file_path.split('_')[-1][:-4]}.pt")
        elif "simple" in log_dir:
            try:
                self.mapping_cube = torch.load(f"./mapping_cubes/mapping_cube_simple_{optics_model_file_path.split('_')[-1][:-4]}.pt", map_location=self.device)
            except:
                self.mapping_cube = self.optics.create_simple_mapping(torch.linspace(450, 650, 28), shape_scene = [512, 512])
                torch.save(self.mapping_cube, f"./mapping_cubes/mapping_cube_simple_{optics_model_file_path.split('_')[-1][:-4]}.pt")
        elif "wrong" in log_dir:
            if "mis" in log_dir:
                try:
                    self.mapping_cube = torch.load(f"./mapping_cubes/mapping_cube_amiciwrongmis.pt", map_location=self.device)
                except:
                    wrongoptics = HSSystem(config_file_path='../system_specs/system_amiciwrongmis.yml')
                    self.mapping_cube = wrongoptics.get_mapping_scene_detector(torch.linspace(450, 650, 28), shape_scene = [512, 512])
                    del wrongoptics
                    torch.save(self.mapping_cube, "./mapping_cubes/mapping_cube_amiciwrongmis.pt")
                
                singleoptics = HSSystem(config_file_path='../system_specs/system_singlemis.yml')
            else:
                try:
                    self.mapping_cube = torch.load(f"./mapping_cubes/mapping_cube_amiciwrong.pt", map_location=self.device)
                except:
                    wrongoptics = HSSystem(config_file_path='../system_specs/system_amiciwrong.yml')
                    self.mapping_cube = wrongoptics.get_mapping_scene_detector(torch.linspace(450, 650, 28), shape_scene = [512, 512])
                    del wrongoptics
                    torch.save(self.mapping_cube, "./mapping_cubes/mapping_cube_amiciwrong.pt")
                singleoptics = HSSystem(config_file_path='../system_specs/system_single.yml')
                
            self.pixel_dispersed_28 = singleoptics.central_positions_wavelengths(torch.linspace(450, 650, 28))[1] # New duspersion according to the real mesurement
            self.pixel_dispersed_28[:, 0] = 0 # Fix y spread to 0
            wrongoptics = HSSystem(config_file_path='../system_specs/system_amiciwrongmis.yml')
            self.amici_pixel_dispersed_28 = wrongoptics.central_positions_wavelengths(torch.linspace(450, 650, 28))[1]
            del wrongoptics
            del singleoptics
            
            roll_correction = self.pixel_dispersed_28 - self.amici_pixel_dispersed_28 # Correct to adjust to single prism's dispersion
            for l in range(self.mapping_cube.shape[2]):
                self.mapping_cube[:,:,l,1] += roll_correction[l][1].int().item()

        else:
            try:
                self.mapping_cube = torch.load(f"./mapping_cubes/mapping_cube_{optics_model_file_path.split('_')[-1][:-4]}.pt", map_location=self.device)
            except:
                self.mapping_cube = self.optics.get_mapping_scene_detector(torch.linspace(450, 650, 28), shape_scene = [512, 512])
                torch.save(self.mapping_cube, f"./mapping_cubes/mapping_cube_{optics_model_file_path.split('_')[-1][:-4]}.pt")

        if "amici" or "single" in self.optics_model_file_path.lower():
            self.mapping_cube[:, :, :, 0] = self.optics.system[0].film_size[1]-1 - self.mapping_cube[:, :, :, 0] # Since acquisition will be flipped, we have to flip along the y axis
        
        self.mapping_cube = self.mapping_cube.permute(2,0,1,3) # [nC, H, W, 2]

        self.reconstruction_model = model_generator(self.net_model_name, self.pixel_dispersed_28, self.mapping_cube, reconstruction_checkpoint)

        if not train:
            for param in self.reconstruction_model.parameters():
                param.requires_grad = False

        if mask_path is None:
            raise ValueError("Mask path is required")
        else:
            self.mask = torch.load(mask_path, map_location = self.device) # 512 x 512

        if self.net_model_name in ['dwmt']: #['duf', 'mst', 'dwmt']:
            self.mask[256:, :] = 0
            self.mask[:256, :256] = 0

        self.loss_fn = nn.MSELoss()
        self.ssim_loss = SSIM(window_size=11, n_channels=28)
        self.psnr = PeakSignalNoiseRatio()
        self.sam = SpectralAngleMapper()

        self.writer = SummaryWriter(log_dir)
        
        self.log_dir = log_dir

        if self.net_model_name == 'dgsmp':
            self.mask_model = self.mask.unsqueeze(0)
        elif self.net_model_name == "dwmt":
            self.mask_model = self.mask[:256, 256:].unsqueeze(0).repeat(28,1,1).unsqueeze(0)
        elif self.net_model_name == 'padut':
            mask_here = shift(self.mask.unsqueeze(0).repeat(28,1,1).unsqueeze(0), self.pixel_dispersed_28)
            mask_s = torch.sum(mask_here**2,1)
            mask_s[mask_s==0] = 1
            self.mask_model = (mask_here, mask_s)
        elif self.net_model_name == 'duf':
            self.mask_model = shift(self.mask.unsqueeze(0).repeat(28,1,1).unsqueeze(0), self.pixel_dispersed_28)
        elif self.net_model_name == 'mst':
            self.mask_model = shift(self.mask.unsqueeze(0).repeat(28,1,1).unsqueeze(0), self.pixel_dispersed_28)
        elif 'dauhst' in self.net_model_name:
            mask_here = shift(self.mask.unsqueeze(0).repeat(28,1,1).unsqueeze(0), self.pixel_dispersed_28)
            mask_s = torch.sum(mask_here**2,1)
            mask_s[mask_s==0] = 1
            self.mask_model = (mask_here, mask_s)
        else:
            raise ValueError(f'Method {self.net_model_name} is not defined')
        
        self.nb_rays = nb_rays

        texture = torch.ones((self.mask.shape[0], self.mask.shape[1], 1), dtype=torch.float)
        z0 = torch.tensor([self.optics.system[-1].d_sensor*torch.cos(self.optics.system[-1].theta_y*np.pi/180).item() + self.optics.system[-1].origin[-1] + self.optics.system[-1].shift[-1]]).item()
        self.z0 = z0

        try:
            self.ray_pos = torch.load(rays_file, map_location=self.device)
            self.ray_valid = torch.load(valid_file, map_location=self.device)
        except:
            self.ray_pos, self.ray_valid = self.optics.save_pos_render(wavelengths=self.optics.wavelengths, nb_rays=self.nb_rays, z0=z0,
                        texture=texture, offsets=None, numerical_aperture=0.05)
            torch.save(self.ray_pos, f"./rays/rays_{optics_model_file_path.split('_')[-1][:-4]}.pt")
            torch.save(self.ray_valid, f"./rays/rays_valid_{optics_model_file_path.split('_')[-1][:-4]}.pt")

        try:
            self.airy = torch.load(f"./airy.pt", map_location=self.device).half()
            if self.airy.shape[0] != self.optics.wavelengths.shape[0]:
                raise ValueError("Airy disk not the same size as the wavelengths")
        except:
            self.airy = compute_airy_disk(self.optics.wavelengths, self.optics.system[-1].pixel_size, na=0.05, grid_size = 7, magnification = 2).half()
            torch.save(self.airy, f"./airy.pt")

    def on_fit_start(self):
        self.mask = self.mask.to(self.device)
        self.ray_pos, self.ray_valid = self.ray_pos.to(self.device), self.ray_valid.to(self.device)
        self.mask_model = self.mask_model.to(self.device) if not isinstance(self.mask_model, tuple) else (self.mask_model[0].to(self.device), self.mask_model[1].to(self.device))
        self.optics.wavelengths = self.optics.wavelengths.to(self.device)
        self.optics.device = self.device
        self.airy = self.airy.to(self.device)
        self.pixel_dispersed_28 = self.pixel_dispersed_28.to(self.device)
        self.mapping_cube = self.mapping_cube.to(self.device)
        self.reconstruction_model = self.reconstruction_model.to(self.device)
        

    def on_validation_start(self,stage=None):
        print("---VALIDATION START---")
        

    def on_predict_start(self,stage=None):
        print("---PREDICT START---")
        self.mask = self.mask.to(self.device)
        self.ray_pos, self.ray_valid = self.ray_pos.to(self.device), self.ray_valid.to(self.device)
        self.mask_model = self.mask_model.to(self.device) if not isinstance(self.mask_model, tuple) else (self.mask_model[0].to(self.device), self.mask_model[1].to(self.device))
        self.optics.wavelengths = self.optics.wavelengths.to(self.device)
        self.optics.device = self.device
        self.airy = self.airy.to(self.device)
        self.pixel_dispersed_28 = self.pixel_dispersed_28.to(self.device)
        self.mapping_cube = self.mapping_cube.to(self.device)
        self.reconstruction_model = self.reconstruction_model.to(self.device)

        if not os.path.exists(f'./results/{self.net_model_name}'):
            os.makedirs(f'./results/{self.net_model_name}')

        self.predict_results = {}

    def on_predict_end(self):
        list_rmse = []
        list_ssim = []
        list_psnr = []
        list_sam = []
        for key in self.predict_results.keys():
            val = self.predict_results[key]
            if 'SSIM' in key:
                list_ssim.append(val[0])
            elif 'RMSE' in key:
                list_rmse.append(val[0])
            elif 'PSNR' in key:
                list_psnr.append(val[0])
            elif 'SAM' in key:
                list_sam.append(val[0])
        
        array_list_ssim = np.array(list_ssim)
        array_list_rmse = np.array(list_rmse)
        array_list_psnr = np.array(list_psnr)
        array_list_sam = np.array(list_sam)

        array_list_ssim = array_list_ssim[np.nonzero(array_list_ssim)]
        array_list_rmse = array_list_rmse[np.nonzero(array_list_rmse)]
        array_list_psnr = array_list_psnr[np.nonzero(array_list_psnr)]
        array_list_sam = array_list_sam[np.nonzero(array_list_sam)]

        self.predict_results["Overall_SSIM"] = float(np.mean(array_list_ssim))
        self.predict_results["Overall_RMSE"] = float(np.mean(array_list_rmse))
        self.predict_results["Overall_PSNR"] = float(np.mean(array_list_psnr))
        self.predict_results["Overall_SAM"] = float(np.mean(array_list_sam))

        if "straight_simple" in self.log_dir:
            file_name = f"predict_results_straight_simple_{self.optics_model_file_path.split('_')[-1][:-4]}"
        elif "simple" in self.log_dir:
            file_name = f"predict_results_simple_{self.optics_model_file_path.split('_')[-1][:-4]}"
        elif "wrong" in self.log_dir:
            file_name = f"predict_results_wrong_{self.optics_model_file_path.split('_')[-1][:-4]}"
        else:
            file_name = f"predict_results_{self.optics_model_file_path.split('_')[-1][:-4]}"
        save_config_file(f'./results/{self.net_model_name}/' + file_name, self.predict_results,".")

    def _normalize_data_by_itself(self, data):
        # Calculate the mean and std for each batch individually
        # Keep dimensions for broadcasting
        mean = torch.mean(data, dim=[1, 2], keepdim=True)
        std = torch.std(data, dim=[1, 2], keepdim=True)

        # Normalize each batch by its mean and std
        normalized_data = (data - mean) / std
        return normalized_data


    def forward(self, x):
        print("---FORWARD---")
        # x: acquisition
        batch_size = x.shape[0]
        #print(f"Memory used pre network: {torch.cuda.memory_allocated() / 1024 / 1024:.2f}Mb")
        if self.net_model_name == 'dgsmp':
            reconstructed_cube = self.reconstruction_model(x)
        else:
            mask = self.mask_model.repeat(batch_size, 1, 1, 1) if not isinstance(self.mask_model, tuple) else (self.mask_model[0].repeat(batch_size, 1, 1, 1), self.mask_model[1].repeat(batch_size, 1, 1))
            reconstructed_cube = self.reconstruction_model(x, mask)
        
        if self.net_model_name == 'duf':
            reconstructed_cube = reconstructed_cube[0]
        #print(f"Memory used post network: {torch.cuda.memory_allocated() / 1024 / 1024:.2f}Mb")
	
        return reconstructed_cube


    def training_step(self, batch, batch_idx):
        print("Training step")

        loss, ssim_loss, psnr_loss, reconstructed_cube, ref_cube = self._common_step(batch, batch_idx)


        output_images = self._convert_output_to_images(self._normalize_image_tensor(self.acq))
        patterns = self._convert_output_to_images(self._normalize_image_tensor(self.mask.unsqueeze(0).int()))
        input_images = self._convert_output_to_images(self._normalize_image_tensor(ref_cube[:,0,:,:]))
        reconstructed_image = self._convert_output_to_images(self._normalize_image_tensor(reconstructed_cube[:,0,:,:]))

        if self.global_step % 30 == 0:
            self._log_images('train/acquisition', output_images, self.global_step)
            self._log_images('train/ground_truth', input_images, self.global_step)
            self._log_images('train/reconstructed', reconstructed_image, self.global_step)
            self._log_images('train/patterns', patterns, self.global_step)

            spectral_filter_plot = self.plot_spectral_filter(ref_cube,reconstructed_cube)

            self.writer.add_image('Spectral Filter', spectral_filter_plot, self.global_step)

        self.log_dict(
            { "train_loss": loss,
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        self.log_dict(
            { "train_ssim_loss": ssim_loss,
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        self.log_dict(
            { "train_psnr_loss": psnr_loss,
            },
            on_step=True,
            on_epoch=True,
            prog_bar=False,
        )

        return {"loss": loss}

    def _normalize_image_tensor(self, tensor):
        # Normalize the tensor to the range [0, 1]
        min_val = tensor.min()
        max_val = tensor.max()
        normalized_tensor = (tensor - min_val) / (max_val - min_val)
        return normalized_tensor

    def validation_step(self, batch, batch_idx):

        print("Validation step")
        loss, ssim_loss, psnr_loss, reconstructed_cube, ref_cube = self._common_step(batch, batch_idx)

        self.log_dict(
            { "val_loss": loss,
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        self.log_dict(
            { "val_ssim_loss": ssim_loss,
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        self.log_dict(
            { "val_psnr_loss": psnr_loss,
            },
            on_step=True,
            on_epoch=True,
            prog_bar=False,
        )

        return {"loss": loss}

    def test_step(self, batch, batch_idx):
        print("Test step")
        loss, ssim_loss, psnr_loss, reconstructed_cube, ref_cube = self._common_step(batch, batch_idx)
        self.log_dict(
            { "test_loss": loss,
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        return {"loss": loss}

    def predict_step(self, batch, batch_idx):
        print("Predict step")
        batch = torch.clamp(torch.nn.functional.interpolate(batch, scale_factor=(2, 2), mode='bilinear', align_corners=True), 0, 1)
        loss, ssim_loss, psnr_loss, reconstructed_cube, ref_cube = self._common_step(batch, batch_idx)

        sam_loss = self.sam(reconstructed_cube, ref_cube)

        if self.net_model_name=="padut" and batch_idx == 1:
            if "straight_simple" in self.log_dir:
                torch.save(self.acq, f"./results/{self.net_model_name}/straight_simple_{self.optics_model_file_path.split('_')[-1][:-4]}_acq.pt")
                torch.save(ref_cube[0,:,:,:], f"./results/{self.net_model_name}/ref.pt")
                torch.save(reconstructed_cube[0,:,:,:], f"./results/{self.net_model_name}/straight_simple_{self.optics_model_file_path.split('_')[-1][:-4]}_recons.pt")
            elif "simple" in self.log_dir:
                torch.save(self.acq, f"./results/{self.net_model_name}/simple_{self.optics_model_file_path.split('_')[-1][:-4]}_acq.pt")
                torch.save(ref_cube[0,:,:,:], f"./results/{self.net_model_name}/ref.pt")
                torch.save(reconstructed_cube[0,:,:,:], f"./results/{self.net_model_name}/simple_{self.optics_model_file_path.split('_')[-1][:-4]}_recons.pt")
            else:
                torch.save(self.acq, f"./results/{self.net_model_name}/{self.optics_model_file_path.split('_')[-1][:-4]}_acq.pt")
                torch.save(ref_cube[0,:,:,:], f"./results/{self.net_model_name}/ref.pt")
                torch.save(reconstructed_cube[0,:,:,:], f"./results/{self.net_model_name}/{self.optics_model_file_path.split('_')[-1][:-4]}_recons.pt")

        output_images = self._convert_output_to_images(self._normalize_image_tensor(self.acq))
        input_images = self._convert_output_to_images(self._normalize_image_tensor(ref_cube[:,0,:,:]))
        reconstructed_image = self._convert_output_to_images(self._normalize_image_tensor(reconstructed_cube[:,0,:,:]))

        self._log_images('train/acquisition', output_images.float(), batch_idx)
        self._log_images('train/ground_truth', input_images.float(), batch_idx)
        self._log_images('train/reconstructed', reconstructed_image.float(), batch_idx)

        spectral_filter_plot = self.plot_spectral_filter(ref_cube.float(),reconstructed_cube.float())

        self.writer.add_image('Spectral Filter', spectral_filter_plot, batch_idx)

        print("Predict PSNR loss: ", psnr_loss.item())
        print("Predict RMSE loss: ", loss.item())
        print("Predict SSIM loss: ", ssim_loss.item())
        print("Predict SAM loss: ", sam_loss.item())

        try:
            self.predict_results[f"RMSE_scene{batch_idx+1}"].append(loss.item())
            self.predict_results[f"SSIM_scene{batch_idx+1}"].append(ssim_loss.item())
            self.predict_results[f"PSNR_scene{batch_idx+1}"].append(psnr_loss.item())
            self.predict_results[f"SAM_scene{batch_idx+1}"].append(sam_loss.item())
        except:
            self.predict_results[f"RMSE_scene{batch_idx+1}"] = [loss.item()]
            self.predict_results[f"SSIM_scene{batch_idx+1}"] = [ssim_loss.item()]
            self.predict_results[f"PSNR_scene{batch_idx+1}"] = [psnr_loss.item()]
            self.predict_results[f"SAM_scene{batch_idx+1}"] = [sam_loss.item()]

        #if batch_idx == 19-1:
        #    torch.save(reconstructed_cube, f'./results/recons_cube_random.pt')

        return loss

    def _common_step(self, batch, batch_idx):

        texture = batch.permute(0, 2, 3, 1) # batchsize x H x W x nC

        texture = torch.mul(texture, self.mask[None, :, :, None])

        texture = torch.nn.functional.interpolate(texture, scale_factor=(1, self.oversample), mode='bilinear', align_corners=True)

        time_start = time.time()
        
        if self.net_model_name in ['dauhst']:
            # To reduce memory usage with dauhst we render the batch in parts
            batch_acq = self.optics.render_batch_based_on_saved_pos(big_uv = self.ray_pos,
                                                                            big_mask = self.ray_valid,
                                                                            texture = texture, nb_rays=self.nb_rays,
                                                                            wavelengths = self.optics.wavelengths,
                                                                            z0=self.z0) # batchsize x H x W x nC
        else:
            batch_acq = self.optics.render_all_based_on_saved_pos(big_uv = self.ray_pos,
                                                                            big_mask = self.ray_valid,
                                                                            texture = texture, nb_rays=self.nb_rays,
                                                                            wavelengths = self.optics.wavelengths,
                                                                            z0=self.z0) # batchsize x H x W x nC
            
        print(f"Rendering time: {time.time()-time_start:.3f}s")
        batch_acq /= self.oversample
        
        # Convolve by the Airy disk
        batch_acq = F.conv2d(batch_acq.permute(0, 3, 1, 2), self.airy, padding = self.airy.shape[-1]//2, groups=self.optics.wavelengths.shape[0])
        batch_acq = batch_acq.permute(0, 2, 3, 1)

        if "amici" or "single" in self.optics_model_file_path.lower():
            batch_acq = batch_acq.flip(1)
                
        if "wrong" in self.log_dir and "amici" in self.optics_model_file_path.lower():
            roll_correction = self.pixel_dispersed_28 - self.amici_pixel_dispersed_28
            roll_correction = roll_correction.int()[:,1]
            for w in range(28):
                batch_acq[:,:,:,w] = torch.roll(batch_acq[:,:,:,w], shifts=roll_correction[w].item(), dims=2)

        batch_acq = batch_acq.sum(-1)
        self.acq = batch_acq
        if self.net_model_name == 'dgsmp':
            batch_acq = batch_acq
        elif self.net_model_name == "dwmt":
            batch_acq = batch_acq / batch_acq.shape[1] * 2
            batch_acq = shift_back(batch_acq, self.pixel_dispersed_28, self.mapping_cube)[:, :, :256, 256:]
        elif self.net_model_name == 'padut':
            batch_acq = batch_acq
        elif self.net_model_name == 'duf':
            batch_acq = batch_acq
        elif self.net_model_name == 'mst':
            batch_acq = batch_acq / batch_acq.shape[1] * 2
            batch_acq = shift_back(batch_acq, self.pixel_dispersed_28, self.mapping_cube)#[:, :, :256, 256:]
        elif 'dauhst' in self.net_model_name:
            batch_acq = batch_acq
        else:
            raise ValueError(f'Method {self.net_model_name} is not defined')
        
        if self.net_model_name in ['dwmt']:#['duf', 'mst', 'dwmt']:
            batch = batch[:, :, :256, 256:]
        reconstructed_cube = self.forward(batch_acq.to(self.device))
        loss = torch.sqrt(self.loss_fn(reconstructed_cube, batch))
        ssim_loss = self.ssim_loss(torch.clamp(reconstructed_cube, 0, 1), batch)
        psnr_loss = self.psnr(torch.clamp(reconstructed_cube, 0, 1), batch)

        return loss, ssim_loss, psnr_loss, reconstructed_cube, batch

    def configure_optimizers(self):
        if "dauhst" in self.net_model_name:
            optimizer = torch.optim.Adam(self.parameters(), lr=4e-4)
            return { "optimizer":optimizer,
                    "lr_scheduler":{
                    "scheduler":torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 500, eta_min=1e-6),
                    "interval": "epoch"
                    }
            }
        elif "dgsmp" in self.net_model_name:
            optimizer = torch.optim.Adam(self.parameters(), lr=0.8e-4)#lr=1e-4)
            return { "optimizer":optimizer,
            }
        elif "mst" in self.net_model_name:
            optimizer = torch.optim.Adam(self.parameters(), lr=4e-4)
            return { "optimizer":optimizer,
                    "lr_scheduler":{
                    "scheduler":torch.optim.lr_scheduler.StepLR(optimizer, 50, gamma=0.5),
                    "interval": "epoch"
                    }
            }
        elif "dwmt" in self.net_model_name:
            optimizer = torch.optim.Adam(self.parameters(), lr=4e-4)
            return { "optimizer":optimizer,
                    "lr_scheduler":{
                    "scheduler":torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 500, eta_min=1e-6),
                    "interval": "epoch"
                    }
            }
        elif "padut" in self.net_model_name:
            optimizer = torch.optim.Adam(self.parameters(), lr=4e-4)
            return { "optimizer":optimizer,
                    "lr_scheduler":{
                    "scheduler":torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 500, eta_min=1e-6),
                    "interval": "epoch"
                    }
            }
        elif "duf" in self.net_model_name:
            optimizer = torch.optim.Adam(self.parameters(), lr=2e-4)
            return { "optimizer":optimizer,
                    "lr_scheduler":{
                    "scheduler":torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 500, eta_min=1e-6),
                    "interval": "epoch"
                    }
            }
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
            return { "optimizer":optimizer,
            }

    def _log_images(self, tag, images, global_step):
        # Convert model output to image grid and log to TensorBoard
        img_grid = torchvision.utils.make_grid(images)
        self.writer.add_image(tag, img_grid, global_step)

    def _convert_output_to_images(self, acquired_images):

        acquired_images = acquired_images.unsqueeze(1)

        # Create a grid of images for visualization
        img_grid = torchvision.utils.make_grid(acquired_images)
        return img_grid

    def plot_spectral_filter(self,ref_hyperspectral_cube,recontructed_hyperspectral_cube):


        batch_size, lambda_, y,x,  = ref_hyperspectral_cube.shape

        # Create a figure with subplots arranged horizontally
        fig, axs = plt.subplots(1, batch_size, figsize=(batch_size * 5, 4))  # Adjust figure size as needed

        # Check if batch_size is 1, axs might not be iterable
        if batch_size == 1:
            axs = [axs]

        # Plot each spectral filter in its own subplot
        for i in range(batch_size):
            colors = ['b', 'g', 'r']
            for j in range(3):
                pix_j_row_value = np.random.randint(0,y)
                pix_j_col_value = np.random.randint(0,x)

                pix_j_ref = ref_hyperspectral_cube[i, :, pix_j_row_value,pix_j_col_value].cpu().detach().numpy()
                pix_j_reconstructed = recontructed_hyperspectral_cube[i, :, pix_j_row_value,pix_j_col_value].cpu().detach().numpy()
                axs[i].plot(pix_j_reconstructed, label="pix reconstructed" + str(j),c=colors[j])
                axs[i].plot(pix_j_ref, label="pix" + str(j), linestyle='--',c=colors[j])

            axs[i].set_title(f"Reconstruction quality")

            axs[i].set_xlabel("Wavelength index")
            axs[i].set_ylabel("pix values")
            axs[i].grid(True)

        plt.legend()
        # Adjust layout
        plt.tight_layout()

        # Create a buffer to save plot
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)

        # Convert PNG buffer to PIL Image
        image = Image.open(buf)

        # Convert PIL Image to Tensor
        image_tensor = transforms.ToTensor()(image)
        return image_tensor


def subsample(input, origin_sampling, target_sampling):
    # Subsample input from origin_sampling to target_sampling
    indices = torch.zeros(len(target_sampling), dtype=torch.int)
    for i in range(len(target_sampling)):
        sample = target_sampling[i]
        idx = torch.abs(origin_sampling-sample).argmin()
        indices[i] = idx
    return input[:,:,:,indices]

def extract_acq_from_cube_batch(cube_3d, dispersion_pixels, middle_pos, texture_size):
    """
        Parameters:
            cube_3d (torch.Tensor): 4D tensor of shape (batch_size, H, W, n_wav)
            dispersion_pixels (torch.Tensor): 2D tensor of shape (n_wav, 2)
            middle_pos (tuple): Middle position of the film
            texture_size (tuple): Size of the texture
    """
    acq_2d = cube_3d.sum(-1)
    rounded_disp = dispersion_pixels.round().int() # Pixels to indices
    return acq_2d[:,
                  middle_pos[0] - texture_size[0]//2 + rounded_disp[0,0]: middle_pos[0] + texture_size[0]//2 + rounded_disp[-1,0],
                  middle_pos[1] - texture_size[1]//2 + rounded_disp[0,1]: middle_pos[1] + texture_size[1]//2 + rounded_disp[-1,1]]

def compute_airy_disk(wavelengths, pixel_size, na=0.05, grid_size = 7, magnification = 1):
    airy_disk_kernel = torch.zeros(wavelengths.shape[0], 1, grid_size, grid_size, device=wavelengths.device)
    for i in range(wavelengths.shape[0]):
        airy_disk_kernel[i, 0, :, :] = airy_disk(wavelengths[i]*1e-6, na, pixel_size, grid_size, magnification = 2)
    return airy_disk_kernel
