from CASSI_class import HSSystem
import diffoptics as do
import matplotlib.pyplot as plt
import numpy as np
from diffoptics.basics import InterpolationMode
import yaml
import torch

import time
from tqdm import tqdm
import os


class DD_CASSI(HSSystem):
    def __init__(self, list_systems_surfaces=None, list_systems_materials=None,
                 list_d_sensor=None, list_r_last=None, list_film_size=None, list_pixel_size=None, list_theta_x=None, list_theta_y=None, list_theta_z=None, list_origin=None, list_shift=None, list_rotation_order=None,
                 mask = None, mask_pixelsize = None, mask_d = 0., mask_theta = None, mask_origin = None, mask_shift = None, mask_index = None,
                 wavelengths=None, device='cuda',
                 save_dir = None, 
                 config_file_path = None):
        """
        Main class for handling the CASSI optical system.
        Arguments are either provided directly or through a configuration file.
        
        Args:
            list_systems_surfaces (list[list[dict]], optional): List of lists of dictionaries representing the surfaces of each lens group. Defaults to None.
            list_systems_materials (list[list[str]], optional): List of lists of the names of the desired materials for each lens group. Defaults to None.
            list_d_sensor (list[float], optional): List of distances from the origin to the sensor for each lens group. Defaults to None.
            list_r_last (list[float], optional): List of radii of the last surface for each lens group. Defaults to None.
            list_film_size (list[tuple[int, int]], optional): List of number of pixels of the sensor in x, y coordinates for each lens group. Defaults to None.
            list_pixel_size (list[float], optional): List of pixel sizes of the sensor for each lens group. Defaults to None.
            list_theta_x (list[float], optional): List of rotation angles (in degrees) along the x axis for each lens group. Defaults to None.
            list_theta_y (list[float], optional): List of rotation angles (in degrees) along the y axis for each lens group. Defaults to None.
            list_theta_z (list[float], optional): List of rotation angles (in degrees) along the z axis for each lens group. Defaults to None.
            list_origin (list[tuple[float, float, float]], optional): List of origin positions in x, y, z coordinates for each lens group. Defaults to None.
            list_shift (list[tuple[float, float, float]], optional): List of shifts of the lens groups relative to the origin in x, y, z coordinates. Defaults to None.
            list_rotation_order (list[str], optional): List of operation orders for the computation of rotation matrices for each lens group. Defaults to None.
            mask (torch.Tensor, optional): Mask for the DD CASSI system. Defaults to None.
            mask_pixelsize (tuple[int, int], optional): Pixel sizes of the mask. Defaults to None.
            mask_d (float, optional): Distance from the origin to the mask. Defaults to 0.
            mask_theta (list[float], optional): Rotation angles (in degrees) for the mask, along the x, y, z axis. Defaults to None.
            mask_origin (tuple[float, float, float], optional): Origin positions in x, y, z coordinates for the mask. Defaults to None.
            mask_shift (tuple[float, float, float], optional): Shifts of the mask relative to the origin in x, y, z coordinates. Defaults to None.
            mask_index (int, optional): Index of the last surface before the mask. If None, will be initialized to len(self.system)//2 - 1. Defaults to None.
            wavelengths (torch.Tensor, optional): Considered wavelengths for the usage of the system. Defaults to None.
            device (str, optional): Device to use for the computations. Defaults to 'cuda'.
            save_dir (str, optional): Directory to save the results. Defaults to None.
            config_file_path (str, optional): Path to a configuration file. Defaults to None.
        """
        if config_file_path is not None:
            self.import_system(config_file_path)
            if device != 'cuda':
                self.device = device
            if save_dir is not None:
                self.save_dir = save_dir

            self.mask = mask
        else:
            self.systems_surfaces = list_systems_surfaces
            self.systems_materials = list_systems_materials
            self.wavelengths = wavelengths
            self.device = device
            self.save_dir = save_dir

            self.list_d_sensor = list_d_sensor
            self.list_r_last = list_r_last
            self.list_film_size = list_film_size
            self.list_pixel_size = list_pixel_size
            self.list_theta_x = list_theta_x if list_theta_x is not None else [0. for _ in range(len(list_systems_surfaces))]
            self.list_theta_y = list_theta_y if list_theta_y is not None else [0. for _ in range(len(list_systems_surfaces))]
            self.list_theta_z = list_theta_z if list_theta_z is not None else [0. for _ in range(len(list_systems_surfaces))]
            self.list_origin = list_origin if list_origin is not None else [[0., 0., 0.] for _ in range(len(list_systems_surfaces))]
            self.list_shift = list_shift if list_shift is not None else [[0., 0., 0.] for _ in range(len(list_systems_surfaces))]
            self.list_rotation_order = list_rotation_order

            self.mask = mask
            self.mask_pixelsize = mask_pixelsize if mask_pixelsize is not None else 0. # In x, y coordinates
            self.mask_d = mask_d

            self.mask_theta = mask_theta if mask_theta is not None else [0., 0., 0.]
            self.mask_origin = mask_origin if mask_origin is not None else [0., 0., 0.]
            self.mask_shift = mask_shift if mask_shift is not None else [0., 0., 0.]
            self.mask_index = mask_index
        
        self.entry_radius = self.systems_surfaces[0][0]['R']

        self.create_system(self.list_d_sensor, self.list_r_last, self.list_film_size, self.list_pixel_size, self.list_theta_x, self.list_theta_y, self.list_theta_z, self.list_origin, self.list_shift, self.list_rotation_order)
        self.pos_dispersed, self.pixel_dispersed = self.central_positions_wavelengths(self.wavelengths) # In x,y coordinates for pos, in lines, columns (y, x) for pixel
        
        if self.mask_index is None:
            self.mask_index = len(self.system)//2 - 1 # - 1 because the indexation starts at 0.
        
        self.compute_mask_transformation(self.mask_origin, self.mask_shift, self.mask_theta[0], self.mask_theta[1], self.mask_theta[2])

        self.mask_mts_prepared = False

    def export_system(self, filepath = "system.yml"):
        """
        Export the system to a YAML file.

        Args:
            filepath (str, optional): The path to the YAML file. Defaults to "system.yml".
        
        Returns:
            None
        """
        system_dict = {}
        system_dict['systems_surfaces'] = list(self.systems_surfaces)
        system_dict['systems_materials'] = list(self.systems_materials)
        system_dict['list_d_sensor'] = list(self.list_d_sensor)
        system_dict['list_r_last'] = list(self.list_r_last)
        system_dict['list_film_size'] = list(self.list_film_size)
        system_dict['list_pixel_size'] = list(self.list_pixel_size)
        system_dict['list_theta_x'] = list(self.list_theta_x)
        system_dict['list_theta_y'] = list(self.list_theta_y)
        system_dict['list_theta_z'] = list(self.list_theta_z)
        system_dict['list_origin'] = list(self.list_origin)
        system_dict['list_shift'] = list(self.list_shift)
        system_dict['list_rotation_order'] = list(self.list_rotation_order)

        system_dict['mask_pixelsize'] = self.mask_pixelsize
        system_dict['mask_d'] = self.mask_d
        system_dict['mask_theta'] = list(self.mask_theta)
        system_dict['mask_origin'] = list(self.mask_origin)
        system_dict['mask_shift'] = list(self.mask_shift)
        system_dict['mask_index'] = self.mask_index

        system_dict['wavelengths'] = self.wavelengths.tolist()
        system_dict['device'] = str(self.device)
        system_dict['save_dir'] = self.save_dir

        with open(filepath, 'w') as file:
            yaml.dump(system_dict, file)


    def import_system(self, filepath = "./system.yml"):
        """
        Import the system from a YAML file.

        Args:
            filepath (str, optional): The path to the YAML file. Defaults to "./system.yml".
        Returns:
            None
        """
        with open(filepath, 'r') as file:
            system_dict = yaml.safe_load(file)
        
        self.systems_surfaces = system_dict['systems_surfaces']
        self.systems_materials = system_dict['systems_materials']
        self.list_d_sensor = system_dict['list_d_sensor']
        self.list_r_last = system_dict['list_r_last']
        self.list_film_size = system_dict['list_film_size']
        self.list_pixel_size = system_dict['list_pixel_size']
        self.list_theta_x = system_dict['list_theta_x']
        self.list_theta_y = system_dict['list_theta_y']
        self.list_theta_z = system_dict['list_theta_z']
        self.list_origin = system_dict['list_origin']
        self.list_shift = system_dict['list_shift']
        self.list_rotation_order = system_dict['list_rotation_order']

        self.mask_d = system_dict['mask_d']
        self.mask_pixelsize = system_dict['mask_pixelsize']
        self.mask_theta = system_dict['mask_theta']
        self.mask_origin = system_dict['mask_origin']
        self.mask_shift = system_dict['mask_shift']
        self.mask_index = system_dict['mask_index']

        self.wavelengths = torch.tensor(system_dict['wavelengths']).float()
        self.device = system_dict['device']
        self.save_dir = system_dict['save_dir']
    
    def modify_mask(self, mask):
        self.mask = mask
    
    def modify_mask_pixelsize(self, mask_pixelsize):
        self.mask_pixelsize = mask_pixelsize
    
    def compute_mask_transformation(self, origin = [0., 0., 0.], shift = [0., 0., 0.], theta_x = 0., theta_y = 0., theta_z = 0., rotation_order = 'xyz'):
        """
        Compute the transformation matrix for the mask.

        Args:
            origin (list[float], optional): Origin of the mask. Defaults to [0., 0., 0.].
            shift (list[float], optional): Shift of the mask. Defaults to [0., 0., 0.].
            theta_x (float, optional): Rotation angle along the x axis. Defaults to 0..
            theta_y (float, optional): Rotation angle along the y axis. Defaults to 0..
            theta_z (float, optional): Rotation angle along the z axis. Defaults to 0..
            rotation_order (str, optional): Order of the rotations. Defaults to 'xyz'.

        Returns:
            tuple: A tuple containing the rotation matrix and the translation vector.
        """
        mask_lens_object = do.Lensgroup(origin, shift, theta_x, theta_y, theta_z, rotation_order, device = self.device)
        if self.mask is not None:
            surf = [do.Aspheric(self.mask_pixelsize * self.mask.shape[0], self.mask_d), do.Aspheric(self.mask_pixelsize * self.mask.shape[0], self.mask_d)]
        else:
            surf = [do.Aspheric(self.mask_pixelsize, self.mask_d), do.Aspheric(self.mask_pixelsize, self.mask_d)]
        materials = ['air', 'air', 'air']
        materials_processed = []
        for material in materials:
            materials_processed.append(do.Material(material))
        mask_lens_object.load(surf, materials_processed)

        self.mask_R, self.mask_t = mask_lens_object._compute_transformation().R.to(self.device), mask_lens_object._compute_transformation().t.to(self.device)

        if self.mask_d != 0:
            new_origin = mask_lens_object.to_world.transform_point(torch.tensor([0., 0., self.mask_d]).to(self.device)).cpu().detach()
            self.mask_d = 0
            self.mask_origin = new_origin.tolist()
            self.compute_mask_transformation(new_origin, shift, theta_x, theta_y, theta_z, rotation_order=rotation_order)
        

        #self.mask_t[2] += self.mask_d #TODO does that work in all cases ?

        self.mask_lens_object = mask_lens_object

        return self.mask_R, self.mask_t
    
    def prepare_mts_mask(self, start_distance = 0):
        """
        Prepares the lens for multi-surface tracing (MTS) by setting the necessary parameters. This function should be called before rendering the lens.
        It is based on the original prepare_mts function in the diffoptics library, with the option to specify the starting distance for the lens surfaces to allow for smoother rendering with several lens groups.

        Args:
            start_distance (float, optional): The starting distance for the mask. Defaults to 0.0.
        """
        self.mask_d = -self.mask_d

        self.mask_origin = torch.tensor([self.mask_origin[0], - self.mask_origin[1], start_distance - self.mask_origin[2]]).float().to(device=self.device)
        self.mask_shift = torch.tensor([self.mask_shift[0], - self.mask_shift[1], - self.mask_shift[2]]).float().to(device=self.device)
        self.mask_theta = [-self.mask_theta[0], -self.mask_theta[1], self.mask_theta[2]]

        self.mask_index = len(self.system) - (self.mask_index + 1) - 1
        
        self.compute_mask_transformation(self.mask_origin, self.mask_shift, self.mask_theta[0], self.mask_theta[1], self.mask_theta[2])

        self.mask_mts_prepared = True

        


    def apply_mask(self, z0, rays, valid, R_mask = None, t_mask = None, shading_type = "single", save_pos = False):
        """
        Applies the mask to the rays.

        Args:
            z0 (float): z Position of the mask.
            rays (do.Rays): Rays to apply the mask to.
            valid (torch.Tensor): Valid rays.
            R_mask (np.ndarray, optional): Rotation matrix of the mask. Defaults to None.
            t_mask (np.ndarray, optional): Translation vector of the mask. Defaults to None.
            shading_type (str, optional): Type of shading to apply. Defaults to "single". Available options are "all", "batch" and "single".
            save_pos (bool, optional): Whether to save the positions of the resulting rays (True), or directly apply the mask (False). Defaults to False.
        Returns:
            valid_rays (torch.Tensor): Valid rays after mask intersection
        """

        if self.mask is None:
            return valid

        texturesize = np.array(self.mask.shape)

        if R_mask is None:
            R_mask = np.eye(3)
        
        if t_mask is None:
            t_mask = np.array([0, 0, z0])

        texture = torch.from_numpy(self.mask) if not isinstance(self.mask, torch.Tensor) else self.mask
        #texture = texture.clone().rot90(1, [0, 1]).to(self.device)
        
        #t_mask = torch.Tensor([0., 0., z0]).to(self.device)
        screen = do.Screen(
            do.Transformation(R_mask, t_mask),
            texturesize * self.mask_pixelsize, texture, device=self.device
        )

        # print("R mask: ", R_mask)
        # print("t mask: ", t_mask)
        # print("d mask: ", self.mask_d)
        #local, uv, valid_screen = screen.intersect(rays)[:]
        # print("Local: ", local[...,2])
        # print("Rays : ", rays.o[...,2])
        uv, valid_screen = screen.intersect(rays)[1:] #uv is the intersection points, in [0,1]^2, of shape (N, 2) with N the number of rays, valid_screen is a boolean tensor indicating if the ray intersects the screen

        valid_last = valid & valid_screen

        if shading_type == "single":
            masked_image = screen.shading(uv, valid_last, lmode = InterpolationMode.nearest)
        elif shading_type == "batch":
            raise NotImplementedError
            #masked_image = screen.shading_batch(uv, valid_last, lmode = InterpolationMode.nearest)
        elif shading_type == "all":
            raise NotImplementedError
            #masked_image = screen.shading_all(uv, valid_last, lmode = InterpolationMode.nearest)



        valid_rays = valid_last & torch.tensor((masked_image > 0))

        if save_pos:
            return uv, valid_last
        else:
            return valid_rays
    
    def render_single_back(self, wavelength, screen, numerical_aperture = 0.05, z0_mask = None, save_pos = False, fixed_mask = False):
        """
        Renders back propagation of single ray through a series of lenses onto a screen.

        Args:
            wavelength (float): The wavelength of the light.
            screen (object): The screen object onto which the light is projected.
            numerical_aperture (float, optional): The numerical aperture of the system. Defaults to 0.05.
            z0_mask (float, optional): z Position of the mask. Defaults to 0..
            save_pos (bool, optional): Whether to save the positions of the resulting rays (True), or return a rendered image (False). Defaults to False.
            fixed_mask (bool, optional): Whether to use a fixed mask. If the mask is fixed, the rendering will be faster and less data will need to be processed. Defaults to False.
        Returns:
            tuple: A tuple containing the intensity values (I) and the mask indicating valid pixels on the screen.
        """
        # Sample rays from the sensor
        valid, ray_mid = self.system[0].sample_ray_sensor(wavelength.item(), numerical_aperture = numerical_aperture)

        #print("Outgoing rays: ", ray_mid)
        if z0_mask is None:
            #z0_mask = self.mask_origin[2] + self.mask_shift[2] + self.mask_d
            z0_mask = self.mask_lens_object.to_world.transform_point(torch.tensor([0., 0., self.mask_d]).to(self.device)).cpu().detach().numpy()[2]

        print("z0 mask: ", z0_mask)

        if self.mask_index == 0:
            if save_pos and not fixed_mask:
                uv_mask, valid_mask = self.apply_mask(z0_mask, ray_mid, valid, self.mask_R, self.mask_t, shading_type = "single", save_pos = True)
            else:
                valid = self.apply_mask(z0_mask, ray_mid, valid, self.mask_R, self.mask_t, shading_type = "single", save_pos = False)   

        #print("Ray 1: ", ray_mid)
        #print("Nb valid 1: ", torch.sum(valid))
        # Trace rays through each lens in the system
        if len(self.system) > 1:
            for ind, lens in enumerate(self.system[1:-1]):
                ray_mid = lens.to_object.transform_ray(ray_mid)
                valid_1, ray_mid = lens._trace(ray_mid)

                ray_mid = lens.to_world.transform_ray(ray_mid)

                if self.mask_index == ind + 1:
                    if save_pos and not fixed_mask:
                        uv_mask, valid_mask = self.apply_mask(z0_mask, ray_mid, valid_1, self.mask_R, self.mask_t, shading_type = "single", save_pos = True)
                    else:
                        valid_1 = self.apply_mask(z0_mask, ray_mid, valid_1, self.mask_R, self.mask_t, shading_type = "single", save_pos = False)

                valid = valid & valid_1

            # Trace rays to the first lens
            ray_mid = self.system[-1].to_object.transform_ray(ray_mid)

            valid_last, ray_last = self.system[-1]._trace(ray_mid)

            #print("Nb valid last: ", torch.sum(valid_last))
            valid_last = valid & valid_last
            #print("Ray last before transform: ", ray_last)

            ray_last = self.system[-1].to_world.transform_ray(ray_last)
            #print("Ray last: ", ray_last)

            if self.mask_index == len(self.system) - 1:
                if save_pos and not fixed_mask:
                    uv_mask, valid_mask = self.apply_mask(z0_mask, ray_last, valid_last, self.mask_R, self.mask_t, shading_type = "single", save_pos = True)
                else:
                    valid_last = self.apply_mask(z0_mask, ray_last, valid_last, self.mask_R, self.mask_t, shading_type = "single", save_pos = False)
        else:
            ray_last = ray_mid
            valid_last = valid
        

        # Intersect rays with the screen
        uv, valid_screen = screen.intersect(ray_last)[1:]
        # Apply mask to filter out invalid rays
        valid_render = valid_last & valid_screen
        
        print("Ratio valid rays: ", (torch.sum(valid_render)/(screen.texture.shape[0]*screen.texture.shape[1])).item())

        if save_pos and not fixed_mask:
            return uv, valid_render, uv_mask, valid_mask
        elif save_pos and fixed_mask:
            return uv, valid_render
        else:
            # Calculate intensity values on the screen
            I = screen.shading(uv, valid_render)
            
            return I, valid_render

    def render_single_back_save_pos(self, wavelength, screen, numerical_aperture = 0.05, fixed_mask = False):
        """
        Renders back propagation of single ray through a series of lenses onto a screen. Used to save the positions of the resulting rays.

        Args:
            wavelength (float): The wavelength of the light.
            screen (object): The screen object onto which the light is projected.
            numerical_aperture (float, optional): The numerical aperture of the system. Defaults to 0.05.
            fixed_mask (bool, optional): Whether to use a fixed mask. If the mask is fixed, the rendering will be faster and less data will need to be processed. Defaults to False.

        Returns:
            tuple: A tuple containing the rays positions (uv) and the mask indicating valid pixels on the screen.
        """
        if fixed_mask:
            uv, valid_render = self.render_single_back(wavelength, screen, numerical_aperture, save_pos = True, fixed_mask = fixed_mask)

            return uv, valid_render
        else:
            uv, valid_render, uv_mask, valid_mask = self.render_single_back(wavelength, screen, numerical_aperture, save_pos = True, fixed_mask = fixed_mask)

            return uv, valid_render, uv_mask, valid_mask
    
    def save_pos_render(self, texture = None, nb_rays=20, wavelengths = [656.2725, 587.5618, 486.1327], z0=0, offsets=None, numerical_aperture = 0.05, fixed_mask = False):
        """
        Renders a dummy image to save the positions of rays passing through the optical system.
        Args:
            texture (ndarray, optional): The texture image to be rendered. Defaults to None.
            nb_rays (int, optional): The number of rays to be rendered per pixel. Defaults to 20.
            wavelengths (list, optional): The wavelengths of the rays to be rendered. Defaults to [656.2725, 587.5618, 486.1327].
            z0 (int, optional): The z-coordinate of the screen. Defaults to 0.
            offsets (list, optional): The offsets for each lens in the system. Defaults to None.
            numerical_aperture (int, optional): The reduction factor for the aperture. Defaults to 0.05.
            fixed_mask (bool, optional): Whether to use a fixed mask. If the mask is fixed, the rendering will be faster and less data will need to be processed. Defaults to False.
        Returns:
            tuple: A tuple containing the positions of the rays (big_uv) and the corresponding mask (big_mask).
        """
        #start_distance = self.system[0].d_sensor*torch.cos(self.system[0].theta_y*np.pi/180) + self.system[0].origin[-1] + self.system[0].shift[-1]
        start_distance = z0
        self.prepare_mts_mask(start_distance=start_distance)
        
        if offsets is None:
            offsets = [0 for i in range(self.size_system)]

        # set a rendering image sensor, and call prepare_mts to prepare the lensgroup for rendering
        for i, lens in enumerate(self.system[::-1]):
            lens_mts_R, lens_mts_t = lens._compute_transformation().R, lens._compute_transformation().t
            if i > 0:
                #self.prepare_mts(-i-1, lens.pixel_size, lens.film_size, start_distance = self.system[::-1][i-1].surfaces[-1].d + offsets[-i-1], R=lens_mts_R, t=lens_mts_t)
                #self.prepare_mts(-i-1, lens.pixel_size, lens.film_size, start_distance = self.system[::-1][i-1].origin[-1] + offsets[-i-1], R=lens_mts_R, t=lens_mts_t)
                self.prepare_mts(-i-1, lens.pixel_size, lens.film_size, start_distance = max_z + offsets[-i-1], R=lens_mts_R, t=lens_mts_t)
            else:
                #self.prepare_mts(-1, lens.pixel_size, lens.film_size, start_distance = offsets[-1], R=lens_mts_R, t=lens_mts_t)  
                max_z = lens.d_sensor*torch.cos(lens.theta_y*np.pi/180) + lens.origin[-1] + lens.shift[-1] # Last z coordinate in absolute coordinates
                self.prepare_mts(-1, lens.pixel_size, lens.film_size, start_distance = max_z + offsets[-1], R=lens_mts_R, t=lens_mts_t)   

        self.system = self.system[::-1] # The system is reversed
        
        # create a dummy screen
        pixelsize = self.system[-1].pixel_size # [mm]

        nb_wavelengths = len(wavelengths)

        # default texture
        if texture is None:
            texture = np.ones(self.system[0].film_size + (nb_wavelengths,)).astype(np.float32)
        
        texture_torch = torch.Tensor(texture).float().to(device=self.device)

        texture_torch = texture_torch.rot90(1, dims=[0, 1])
        texturesize = np.array(texture_torch.shape[0:2])

        screen = do.Screen(
            do.Transformation(np.eye(3), np.array([0, 0, z0])),
            texturesize * pixelsize, texture_torch, device=self.device
        )
        
        # render
        ray_counts_per_pixel = nb_rays
        time_start = time.time()
        big_uv = torch.zeros((nb_wavelengths, ray_counts_per_pixel, self.system[0].film_size[0]*self.system[0].film_size[1], 2), dtype=torch.float,  device=self.device)
        big_mask = torch.zeros((nb_wavelengths, ray_counts_per_pixel, self.system[0].film_size[0]*self.system[0].film_size[1]), dtype=torch.bool, device=self.device)
        screen.update_texture(texture_torch[..., 0])

        if not fixed_mask:
            big_uv_mask = torch.zeros((nb_wavelengths, ray_counts_per_pixel, self.system[0].film_size[0]*self.system[0].film_size[1], 2), dtype=torch.float,  device=self.device)
            big_mask_mask = torch.zeros((nb_wavelengths, ray_counts_per_pixel, self.system[0].film_size[0]*self.system[0].film_size[1]), dtype=torch.bool, device=self.device)

        for wavelength_id, wavelength in enumerate(wavelengths):
            # multi-pass rendering by sampling the aperture
            for i in tqdm(range(ray_counts_per_pixel)):
                uv, valid_render, uv_mask, valid_mask = self.render_single_back_save_pos(wavelength, screen, numerical_aperture=numerical_aperture, fixed_mask = fixed_mask)
                big_uv[wavelength_id, i, :, :] = uv
                big_mask[wavelength_id, i, :] = valid_render

                if not fixed_mask:
                    big_uv_mask[wavelength_id, i, :, :] = uv_mask
                    big_mask_mask[wavelength_id, i, :] = valid_mask

            print(f"Elapsed rendering time: {time.time()-time_start:.3f}s for wavelength {wavelength_id+1}")
        
        if fixed_mask:
            return big_uv, big_mask
        else:
            return big_uv, big_mask, big_uv_mask, big_mask_mask
    
    def render_based_on_saved_pos(self, big_uv = None, big_mask = None, big_uv_mask = None, big_mask_mask = None, texture = None, mask_pattern = None, nb_rays=20, wavelengths = [656.2725, 587.5618, 486.1327],
                                  z0=0, fixed_mask = False, save=False, plot = False):
        """
        Renders an image based on the saved ray positions and validity.

        Args:
            big_uv (ndarray, optional): Array of UV coordinates. Defaults to None.
            big_mask (ndarray, optional): Array of masks. Defaults to None.
            big_uv_mask (ndarray, optional): Array of UV coordinates for the mask. Defaults to None.
            big_mask_mask (ndarray, optional): Array of masks for the mask. Defaults to None.
            texture (ndarray, optional): Texture array. Defaults to None.
            mask_pattern (ndarray, optional): Mask array. Defaults to None.
            nb_rays (int, optional): Number of rays. Defaults to 20.
            wavelengths (list, optional): List of wavelengths. Defaults to [656.2725, 587.5618, 486.1327].
            z0 (int, optional): Z-coordinate. Defaults to 0.
            fixed_mask (bool, optional): Whether to use a fixed mask. If the mask is fixed, the rendering will be faster and less data will need to be processed. Defaults to False.
            save (bool, optional): Whether to save the rendered image. Defaults to False.
            plot (bool, optional): Whether to plot the rendered image. Defaults to False.

        Returns:
            ndarray: Rendered image.
        """
        if fixed_mask:
            return super().render_based_on_saved_pos(big_uv, big_mask, texture, nb_rays, wavelengths, z0, save, plot)

        # create a dummy screen
        pixelsize = self.system[0].pixel_size # [mm]

        nb_wavelengths = len(wavelengths)

        # default texture
        if texture is None:
            raise ValueError("Texture must be provided.")

        if plot and (texture.shape[2] == 3):
            plt.figure()
            plt.plot()
            plt.imshow(texture)
            plt.show()

        texture_torch = torch.Tensor(texture).float().to(device=self.device)
        # texture_torch = torch.permute(texture_torch, (1,0,2)) # Permute
        # texture_torch = texture_torch.flip(dims=[0]) # Flip
        texture_torch = texture_torch.rot90(1, dims=[0, 1])
        texturesize = torch.tensor(texture_torch.shape[0:2], device=self.device)

        screen = do.Screen(
            do.Transformation(np.eye(3), np.array([0, 0, z0])),
            texturesize * pixelsize, texture_torch, device=self.device
        )

        if mask_pattern is None:
            mask_pattern = self.mask
        
        masktexturesize = np.array(mask_pattern.shape)

        masktexture = torch.from_numpy(mask_pattern) if not isinstance(mask_pattern, torch.Tensor) else mask_pattern
        #masktexture = masktexture.clone().rot90(1, [0, 1]).to(self.device)
        
        #t_mask = torch.Tensor([0., 0., z0]).to(self.device)
        mask_screen = do.Screen(
            do.Transformation(self.mask_R, self.mask_t),
            masktexturesize * self.mask_pixelsize, masktexture, device=self.device
        )

        # render
        ray_counts_per_pixel = nb_rays
        time_start = time.time()
        Is = []
        for wavelength_id, wavelength in enumerate(wavelengths):
            screen.update_texture(texture_torch[..., wavelength_id])

            # multi-pass rendering by sampling the aperture
            I = 0
            M = 0
            for i in tqdm(range(ray_counts_per_pixel)):
                uv = big_uv[wavelength_id, i, :, :]
                mask = big_mask[wavelength_id, i, :]

                uv_mask = big_uv_mask[wavelength_id, i, :, :]
                mask_mask = big_mask_mask[wavelength_id, i, :]

                valid_mask_screen = mask_screen.shading(uv_mask, mask_mask, lmode = InterpolationMode.nearest)
                valid_mask_screen = torch.tensor((valid_mask_screen > 0))

                mask = mask & valid_mask_screen

                I_current = screen.shading(uv, mask)
                I = I + I_current
                M = M + mask
            I = I / (M + 1e-10)
            # reshape data to a 2D image
            #I = I.reshape(*np.flip(np.asarray(lenses[0].film_size))).permute(1,0)
            print(f"Image {wavelength_id} nonzero count: {I.count_nonzero()}")
            #I = torch.flip(I.reshape(*np.flip(np.asarray(lenses[0].film_size))).permute(1,0), dims = [0])
            #I = torch.flip(I.reshape(*np.flip(np.asarray(lenses[0].film_size))), dims = [0])
            #I = torch.flip(I.reshape(*np.flip(np.asarray(lenses[0].film_size))), dims=[0, 1])
            I = I.reshape(*np.flip(np.asarray(self.system[0].film_size))) # Flip
            Is.append(I)
        # show image
        I_rendered = torch.stack(Is, axis=-1)#.astype(np.uint8)
        I_rendered_plot = I_rendered.clone().detach().cpu().numpy()
        print(f"Elapsed rendering time: {time.time()-time_start:.3f}s")
        if plot and (nb_wavelengths==3):
            plt.imshow(np.flip(I_rendered_plot/I_rendered_plot.max(axis=(0,1))[np.newaxis, np.newaxis, :], axis=2))
            plt.title("RGB rendered with dO")
            plt.show()
        ax, fig = plt.subplots((nb_wavelengths+2)// 3, 3, figsize=(15, 5))
        ax.suptitle("Rendered with dO")
        for i in range(nb_wavelengths):
            fig.ravel()[i].set_title("Wavelength: " + str(float(wavelengths[i])) + " nm")
            if nb_wavelengths > 3:
                fig[i//3, i % 3].imshow(I_rendered_plot[:,:,i])
            else:
                fig[i].imshow(I_rendered_plot[:,:,i])
            if save and self.save_dir is not None:
                plt.imsave(os.path.join(self.save_dir, f"rendered_{wavelengths[i]}.png"), I_rendered_plot[:,:,i])
        if plot:
            plt.show()
            plt.imshow(np.sum(I_rendered_plot, axis=2))
            plt.title("Sum of all wavelengths")
            plt.show()

        if save and nb_wavelengths==3 and self.save_dir is not None:
            plt.imsave(os.path.join(self.save_dir, "rendered_rgb.png"), I_rendered_plot/I_rendered_plot.max(axis=(0,1))[np.newaxis, np.newaxis, :])

        return I_rendered

    def render_batch_based_on_saved_pos(self, big_uv = None, big_mask = None, big_uv_mask = None, big_mask_mask = None, texture = None, mask_pattern = None, nb_rays=20, wavelengths = [656.2725, 587.5618, 486.1327],
                                  z0=0, fixed_mask = False):
        """
        Renders an image based on the saved ray positions and validity.

        Args:
            big_uv (ndarray, optional): Array of UV coordinates. Defaults to None.
            big_mask (ndarray, optional): Array of masks. Defaults to None.
            big_uv_mask (ndarray, optional): Array of UV coordinates for the mask. Defaults to None.
            big_mask_mask (ndarray, optional): Array of masks for the mask. Defaults to None.
            texture (ndarray, optional): Texture array. Defaults to None.
            mask_pattern (ndarray, optional): Mask array. Defaults to None.
            nb_rays (int, optional): Number of rays. Defaults to 20.
            wavelengths (list, optional): List of wavelengths. Defaults to [656.2725, 587.5618, 486.1327].
            z0 (int, optional): Z-coordinate. Defaults to 0.
            fixed_mask (bool, optional): Whether to use a fixed mask. If the mask is fixed, the rendering will be faster and less data will need to be processed. Defaults to False.
        Returns:
            ndarray: Rendered image.
        """
        if fixed_mask:
            return super().render_batch_based_on_saved_pos(big_uv, big_mask, texture, nb_rays, wavelengths, z0)

        # create a dummy screen
        pixelsize = self.system[0].pixel_size # [mm]

        nb_wavelengths = len(wavelengths)

        texture_torch = torch.Tensor(texture).float().to(device=self.device)
        # texture_torch = torch.permute(texture_torch, (0, 2, 1, 3)) # Permute
        # texture_torch = texture_torch.flip(dims=[1]) # Flip
        texture_torch = texture_torch.rot90(1, dims=[1, 2])
        texturesize = np.array(texture_torch.shape[1:3])

        screen = do.Screen(
            do.Transformation(np.eye(3), np.array([0, 0, z0])),
            texturesize * pixelsize, texture_torch, device=self.device
        )
        
        if mask_pattern is None:
            mask_pattern = self.mask
        
        masktexturesize = np.array(mask_pattern.shape)

        masktexture = torch.from_numpy(mask_pattern) if not isinstance(mask_pattern, torch.Tensor) else mask_pattern
        #masktexture = masktexture.clone().rot90(1, [0, 1]).to(self.device)
        
        #t_mask = torch.Tensor([0., 0., z0]).to(self.device)
        mask_screen = do.Screen(
            do.Transformation(self.mask_R, self.mask_t),
            masktexturesize * self.mask_pixelsize, masktexture, device=self.device
        )

        mask_screen.update_texture_batch(masktexture.unsqueeze(0).repeat(texture_torch.shape[0], 1, 1))

        # render
        ray_counts_per_pixel = nb_rays
        Is = []
        print("Simulating acquisition")
        for wavelength_id, wavelength in tqdm(enumerate(wavelengths)):
            screen.update_texture_batch(texture_torch[..., wavelength_id])

            # multi-pass rendering by sampling the aperture
            I = 0
            M = 0
            for i in range(ray_counts_per_pixel):
                uv = big_uv[wavelength_id, i, :, :]
                mask = big_mask[wavelength_id, i, :]

                
                uv_mask = big_uv_mask[wavelength_id, i, :, :]
                mask_mask = big_mask_mask[wavelength_id, i, :]

                valid_mask_screen = mask_screen.shading_batch(uv_mask, mask_mask, lmode = InterpolationMode.nearest)
                valid_mask_screen = torch.tensor((valid_mask_screen > 0))

                if len(valid_mask_screen.shape) > 1:
                    valid_mask_screen = valid_mask_screen[0, ...]

                mask = mask & valid_mask_screen

                I_current = screen.shading_batch(uv, mask)
                I = I + I_current
                M = M + mask
            I = I / (M + 1e-10)
            # reshape data to a 2D image
            #I = I.reshape(*np.flip(np.asarray(lenses[0].film_size))).permute(1,0)
            #####print(f"Image {wavelength_id} nonzero count: {I.count_nonzero()}")
            #I = torch.flip(I.reshape(*np.flip(np.asarray(lenses[0].film_size))).permute(1,0), dims = [0])
            #I = torch.flip(I.reshape(*np.flip(np.asarray(lenses[0].film_size))), dims = [0])
            #I = torch.flip(I.reshape(*np.flip(np.asarray(lenses[0].film_size))), dims=[0, 1])
            # I = I.reshape((-1, self.system[0].film_size[1], self.system[0].film_size[0])).flip(2) # Flip
            I = I.reshape((-1, self.system[0].film_size[1], self.system[0].film_size[0]))
            Is.append(I)
        # show image
        I_rendered = torch.stack(Is, axis=-1)#.astype(np.uint8)
        return I_rendered
    
    def render_all_based_on_saved_pos(self, big_uv = None, big_mask = None, big_uv_mask = None, big_mask_mask = None, texture = None, mask_pattern = None, nb_rays=20, wavelengths = [656.2725, 587.5618, 486.1327],
                                  z0=0, fixed_mask = False):
        """
        Renders an image based on the saved ray positions and validity.

        Args:
            big_uv (ndarray, optional): Array of UV coordinates. Defaults to None.
            big_mask (ndarray, optional): Array of masks. Defaults to None.
            big_uv_mask (ndarray, optional): Array of UV coordinates for the mask. Defaults to None.
            big_mask_mask (ndarray, optional): Array of masks for the mask. Defaults to None.
            texture (ndarray, optional): Texture array. Defaults to None.
            mask_pattern (ndarray, optional): Mask array. Defaults to None.
            nb_rays (int, optional): Number of rays. Defaults to 20.
            wavelengths (list, optional): List of wavelengths. Defaults to [656.2725, 587.5618, 486.1327].
            z0 (int, optional): Z-coordinate. Defaults to 0.
            fixed_mask (bool, optional): Whether to use a fixed mask. If the mask is fixed, the rendering will be faster and less data will need to be processed. Defaults to False.

        Returns:
            ndarray: Rendered image.
        """
        if fixed_mask:
            return super().render_all_based_on_saved_pos(big_uv, big_mask, texture, nb_rays, wavelengths, z0)

        # create a dummy screen
        pixelsize = self.system[0].pixel_size # [mm]

        nb_wavelengths = len(wavelengths)

        texture_torch = torch.Tensor(texture).float().to(device=self.device)
        # texture_torch = torch.permute(texture_torch, (0, 2, 1, 3)) # Permute
        # texture_torch = texture_torch.flip(dims=[1]) # Flip
        texture_torch = texture_torch.rot90(1, dims=[1, 2])
        texturesize = np.array(texture_torch.shape[1:3])

        screen = do.Screen(
            do.Transformation(np.eye(3), np.array([0, 0, z0])),
            texturesize * pixelsize, texture_torch, device=self.device
        )
        
        if mask_pattern is None:
            mask_pattern = self.mask
        
        masktexturesize = np.array(mask_pattern.shape)

        masktexture = torch.from_numpy(mask_pattern) if not isinstance(mask_pattern, torch.Tensor) else mask_pattern
        #masktexture = masktexture.clone().rot90(1, [0, 1]).to(self.device)
        
        #t_mask = torch.Tensor([0., 0., z0]).to(self.device)
        mask_screen = do.Screen(
            do.Transformation(self.mask_R, self.mask_t),
            masktexturesize * self.mask_pixelsize, masktexture, device=self.device
        )

        # render
        print("Simulating acquisition")
        screen.update_texture_all(texture_torch.permute(0, 3, 1, 2))

        mask_screen.update_texture_all(masktexture.unsqueeze(0).unsqueeze(0).repeat(texture_torch.shape[0], texture_torch.shape[1], 1, 1))

        # multi-pass rendering by sampling the aperture
        nb_cut = 4
        I = 0
        for cut in range(nb_cut):
            valid_mask_screen = mask_screen.shading_all(big_uv_mask[:, big_uv_mask.shape[1]//nb_cut*cut:big_uv_mask.shape[1]//nb_cut*(cut+1), :, :], big_mask_mask[:, big_uv_mask.shape[1]//nb_cut*cut:big_uv_mask.shape[1]//nb_cut*(cut+1), :], lmode = InterpolationMode.nearest)

            valid_mask_screen = valid_mask_screen[0, ...]

            valid_mask_screen = torch.tensor((valid_mask_screen > 0))

            Imid = screen.shading_all(big_uv[:, big_uv.shape[1]//nb_cut*cut:big_uv.shape[1]//nb_cut*(cut+1), :, :], big_mask[:, big_uv.shape[1]//nb_cut*cut:big_uv.shape[1]//nb_cut*(cut+1), :] & valid_mask_screen) # [batchsize, nC, nb_rays, N]
            I = I + Imid.sum(dim=2) # [batchsize, nC, N]
        M = big_mask.sum(dim=1) # [nC, N]
        I = I / (M.unsqueeze(0) + 1e-10)
        # reshape data to a 2D image
        #I = I.reshape(*np.flip(np.asarray(lenses[0].film_size))).permute(1,0)
        #####print(f"Image {wavelength_id} nonzero count: {I.count_nonzero()}")
        #I = torch.flip(I.reshape(*np.flip(np.asarray(lenses[0].film_size))).permute(1,0), dims = [0])
        #I = torch.flip(I.reshape(*np.flip(np.asarray(lenses[0].film_size))), dims = [0])
        #I = torch.flip(I.reshape(*np.flip(np.asarray(lenses[0].film_size))), dims=[0, 1])
        # I = I.reshape((-1, self.system[0].film_size[1], self.system[0].film_size[0])).flip(2) # Flip
        I = I.reshape((-1, I.shape[1], self.system[0].film_size[1], self.system[0].film_size[0]))
        # show image
        I_rendered = I.permute(0, 2, 3, 1)#.astype(np.uint8)
        return I_rendered
    
    def propagate(self, texture = None, nb_rays=20, wavelengths = [656.2725, 587.5618, 486.1327], z0=0, offsets=None,
                numerical_aperture = 0.05, save=False, plot = False):
        """
        Perform ray tracing simulation for propagating light through the lens system. Renders the texture on a screen

        Args:
            texture (ndarray, optional): Texture pattern used for rendering. Defaults to None.
            nb_rays (int, optional): Number of rays to be traced per pixel. Defaults to 20.
            wavelengths (list, optional): List of wavelengths to be simulated. Defaults to [656.2725, 587.5618, 486.1327] (RGB).
            z0 (float, optional): Initial z-coordinate of the screen. Defaults to 0.
            offsets (list, optional): List of offsets for each lens in the system. Defaults to None.
            numerical_aperture (float, optional): Numerical aperture of the system. Defaults to 0.05.
            save (bool, optional): Whether to save the rendered image. Defaults to False.
            plot (bool, optional): Whether to plot the rendered image. Defaults to False.

        Returns:
            I_rendered (ndarray): The rendered image.
        """
        #start_distance = self.system[0].d_sensor*torch.cos(self.system[0].theta_y*np.pi/180) + self.system[0].origin[-1] + self.system[0].shift[-1]
        start_distance = z0
        
        self.prepare_mts_mask(start_distance=start_distance)
        return super().propagate(texture, nb_rays, wavelengths, z0, offsets, numerical_aperture, save, plot)
    
    def combined_plot_setup(self, with_sensor=False):
        """
        Plot the setup in a combined figure.

        Args:
            with_sensor (bool, optional): Whether to include the sensor in the plot. Defaults to False.

        Returns:
            fig (matplotlib.figure.Figure): The generated figure object.
            ax (matplotlib.axes.Axes): The generated axes object.
        """
        # Create a figure and axes for the plot
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Plot the setup of each lens in 2D
        for i, lens in enumerate(self.system[:-1]):
            lens.plot_setup2D(ax=ax, fig=fig, with_sensor=with_sensor, show=False)
        
        # Plot the mask position
        if self.mask is not None:
            self.mask_lens_object.plot_setup2D(ax=ax, fig=fig, color='r', with_sensor=False, show=False)

        # Plot the setup of the last lens with the sensor
        self.system[-1].plot_setup2D(ax=ax, fig=fig, with_sensor=True, show=True)
        
        # Return the figure and axes objects
        return fig, ax
    
    def plot_setup_with_rays(self, oss, ax=None, fig=None, color='b-', linewidth=1.0, show=True):
        """
        Plots the setup with rays for a given list of lenses and optical systems.

        Args:
            oss (list): A list of optical system records.
            ax (matplotlib.axes.Axes, optional): The matplotlib axes object. Defaults to None.
            fig (matplotlib.figure.Figure, optional): The matplotlib figure object. Defaults to None.
            color (str, optional): The color of the rays. Defaults to 'b-'.
            linewidth (float, optional): The width of the rays. Defaults to 1.0.
            show (bool, optional): Whether to display the plot. Defaults to True.

        Returns:
            ax (matplotlib.axes.Axes): The matplotlib axes object.
            fig (matplotlib.figure.Figure): The matplotlib figure object.
        """
        
        # If there is only one lens, plot the raytraces with the sensor
        if self.size_system==1:
            ax, fig = self.system[0].plot_raytraces(oss[0], ax=ax, fig=fig, linewidth=linewidth, show=show, with_sensor=True, color=color)
            return ax, fig
        
        # Plot the raytraces for the first lens without the sensor
        ax, fig = self.system[0].plot_raytraces(oss[0], ax=ax, fig=fig, color=color, linewidth=linewidth, show=False, with_sensor=False)
        
        # Plot the raytraces for the intermediate lenses without the sensor
        for i, lens in enumerate(self.system[1:-1]):
            ax, fig = lens.plot_raytraces(oss[i+1], ax=ax, fig=fig, color=color, linewidth=linewidth, show=False, with_sensor=False)
        
        # Plot the mask position
        if self.mask is not None:
            self.mask_lens_object.plot_setup2D(ax=ax, fig=fig, color='r', with_sensor=False, show=False)

        # Plot the raytraces for the last lens with the sensor
        ax, fig = self.system[-1].plot_raytraces(oss[-1], ax=ax, fig=fig, color=color, linewidth=linewidth, show=show, with_sensor=True)
        
        return ax, fig
    
    def expand_system_symmetrically(self, symmetry_ax = "horizontal", ax_position = 0., ax_normal = np.zeros(3)):
        """
        Expands the system symmetrically along a given axis. The expansion will be set at the end of the current system.

        Args:
            symmetry_ax (str, optional): The axis of symmetry. Can be either horizontal, vertical or any. If any, ax_position needs to be a 3D vector. Defaults to "horizontal".
            ax_position (float, optional): The position of the axis of symmetry. It is a z position if the symmetry is "horizontal" and a
                                        x position if the symmetry is "vertical". Defaults to 0.
        """

        ax_normal = np.array(ax_normal) / np.linalg.norm(np.array(ax_normal) + 1e-10)

        new_system_surfaces = []
        new_system_materials = []
        new_system_d_sensor = []
        new_system_r_last = []
        new_system_film_size = []
        new_system_pixel_size = []
        new_system_theta_x = []
        new_system_theta_y = []
        new_system_theta_z = []
        new_system_origin = []
        new_system_shift = []
        new_system_rotation_order = []

        if symmetry_ax == "horizontal":
            # Reverse the order of the lens groups as it is a symmetric expansion
            for lens_id in range(len(self.system)-1, -1, -1):
                lens = self.system[lens_id]

                new_lens_surfaces = lens.surfaces.copy()

                for i in range(len(lens.surfaces)):
                    new_lens_surfaces[i].d = - new_lens_surfaces[i].d
                    new_lens_surfaces[i].reverse()
                new_lens_surfaces.reverse()

                new_surfaces = []
                for ind_s, surface in enumerate(new_lens_surfaces):
                    surface_dict = {}
                    surface_dict['type'] = get_surface_type(surface)
                    if ind_s == 0:
                        surface_dict['d'] = surface.d.item() if isinstance(surface.d, torch.Tensor) else surface.d
                    else:
                        surface_dict['d'] = surface.d.item() - new_lens_surfaces[ind_s-1].d.item() if isinstance(surface.d, torch.Tensor) and isinstance(new_lens_surfaces[ind_s-1].d, torch.Tensor) else surface.d - new_lens_surfaces[ind_s-1].d
                    surface_dict['R'] = surface.r.item() if isinstance(surface.r, torch.Tensor) else surface.r
                    surface_dict['params'] = get_surface_params(surface)

                    new_surfaces.append(surface_dict)

                new_system_surfaces.append(new_surfaces)
                new_mats = self.systems_materials[lens_id].copy()
                new_mats.reverse()
                new_system_materials.append(new_mats)

                new_system_d_sensor.append(self.list_d_sensor[lens_id]) #TODO May need a change, as the values between old and new system should be different
                new_system_r_last.append(self.list_r_last[lens_id])
                new_system_film_size.append(self.list_film_size[lens_id])
                new_system_pixel_size.append(self.list_pixel_size[lens_id])

                new_system_theta_x.append(-self.list_theta_x[lens_id]) #TODO Rotations need to be checked too
                new_system_theta_y.append(-self.list_theta_y[lens_id])
                new_system_theta_z.append(-self.list_theta_z[lens_id]) # Because the z axis w.r.t. the origin is reversed

                new_origin = self.list_origin[lens_id].copy() if self.list_origin[lens_id] is not None else [0., 0., 0.]
                new_origin[2] = 2*ax_position - new_origin[2]
                new_system_origin.append(new_origin)
                
                new_shift = self.list_shift[lens_id].copy() if self.list_shift[lens_id] is not None else [0., 0., 0.]

                new_shift[2] = - new_shift[2]
                new_system_shift.append(new_shift)

                new_system_rotation_order.append(self.list_rotation_order[lens_id])
        
        elif symmetry_ax == "vertical":
            # Reverse the order of the lens groups as it is a symmetric expansion
            for lens_id in range(len(self.system)-1, -1, -1):
                lens = self.system[lens_id]

                new_lens_surfaces = lens.surfaces.copy()

                for i in range(len(lens.surfaces)):
                    new_lens_surfaces[i].d = new_lens_surfaces[i].d  # THIS LINE IS FALSE (well, depends on how we want to do things, will see tomorrow)
                new_lens_surfaces.reverse()

                new_surfaces = []
                for ind_s, surface in enumerate(new_lens_surfaces):
                    surface_dict = {}
                    surface_dict['type'] = get_surface_type(surface)
                    if ind_s == 0:
                        surface_dict['d'] = surface.d.item() if isinstance(surface.d, torch.Tensor) else surface.d
                    else:
                        surface_dict['d'] = surface.d.item() - new_lens_surfaces[ind_s-1].d.item() if isinstance(surface.d, torch.Tensor) and isinstance(new_lens_surfaces[ind_s-1].d, torch.Tensor) else surface.d - new_lens_surfaces[ind_s-1].d
                    surface_dict['R'] = surface.r.item() if isinstance(surface.r, torch.Tensor) else surface.r
                    surface_dict['params'] = get_surface_params(surface)

                    if surface_dict['type'] == "XYPolynomial":
                        if len(surface_dict['params']['ai']) == 3:
                            surface_dict['params']['ai'][2] = - surface_dict['params']['ai'][2]

                    new_surfaces.append(surface_dict)

                new_system_surfaces.append(new_surfaces)
                new_mats = self.systems_materials[lens_id].copy()
                new_mats.reverse()
                new_system_materials.append(new_mats)

                new_system_d_sensor.append(-self.list_d_sensor[lens_id]) #TODO May need a change, as the values between old and new system should be different
                new_system_r_last.append(self.list_r_last[lens_id])
                new_system_film_size.append(self.list_film_size[lens_id])
                new_system_pixel_size.append(self.list_pixel_size[lens_id])

                new_system_theta_x.append(self.list_theta_x[lens_id]) #TODO Rotations need to be checked too
                new_system_theta_y.append(-self.list_theta_y[lens_id])
                new_system_theta_z.append(-self.list_theta_z[lens_id])

                new_origin = self.list_origin[lens_id].copy() if self.list_origin[lens_id] is not None else [0., 0., 0.]

                new_origin[0] = 2*ax_position - new_origin[0]
                #new_origin[2] = lens.surfaces[0].d.item() - new_origin[2]
                new_system_origin.append(new_origin)

                new_shift = self.list_shift[lens_id].copy() if self.list_shift[lens_id] is not None else [0., 0., 0.]
                new_shift[0] = - new_shift[0]
                new_system_shift.append(new_shift)

                new_system_rotation_order.append(self.list_rotation_order[lens_id])
        elif symmetry_ax == "any":
            # ax_position must be a list of 3 values

            def mirror_point(point, ax_position, ax_normal):
                point = np.array(point)
                ax_position = np.array(ax_position)
                ax_normal = np.array(ax_normal)

                mirrored_point = point - 2 * np.dot(point - ax_position, ax_normal) * ax_normal

                return mirrored_point
            
            def mirror_direction(direction, ax_normal):
                direction = np.array(direction)
                ax_normal = np.array(ax_normal)

                mirrored_direction = direction - 2 * np.dot(direction, ax_normal) * ax_normal

                return mirrored_direction
                
            basic_dir = np.array([0., 0., 1])
            mirrored_basic_dir = mirror_direction(basic_dir, ax_normal)

            mirrored_basic_rot = np.zeros(3)
            mirrored_basic_rot[0] = np.arctan2(mirrored_basic_dir[2], mirrored_basic_dir[1])
            mirrored_basic_rot[1] = np.arctan2(mirrored_basic_dir[2], mirrored_basic_dir[0])
            mirrored_basic_rot[2] = np.arctan2(mirrored_basic_dir[1], mirrored_basic_dir[0])

            basic_rot = np.zeros(3)
            basic_rot[0] = np.arctan2(basic_dir[2], basic_dir[1])
            basic_rot[1] = np.arctan2(basic_dir[2], basic_dir[0])
            basic_rot[2] = np.arctan2(basic_dir[1], basic_dir[0])

            # Reverse the order of the lens groups as it is a symmetric expansion
            for lens_id in range(len(self.system)-1, -1, -1):
                lens = self.system[lens_id]

                basic_dir = np.array([0., 0., 1.])

                lens_dir = lens.to_world.transform_vector(basic_dir)
                mirrored_lens_dir = mirror_direction(lens_dir, ax_normal)

                mirrored_basic_dir = mirror_direction(basic_dir, ax_normal)

                new_lens_surfaces = lens.surfaces.copy()

                for i in range(len(lens.surfaces)):
                    #new_lens_surfaces[i].d = np.sign(mirrored_basic_dir[2]) * new_lens_surfaces[i].d #- new_lens_surfaces[i].d
                    #new_lens_surfaces[i].d = - np.sign(np.dot(lens_dir, mirrored_lens_dir)) * new_lens_surfaces[i].d
                    new_lens_surfaces[i].d = new_lens_surfaces[i].d
                    #if np.dot(lens_dir, mirrored_lens_dir) > 0:
                    #    new_lens_surfaces[i].reverse()

                new_lens_surfaces.reverse()

                new_surfaces = []
                for ind_s, surface in enumerate(new_lens_surfaces):
                    surface_dict = {}
                    surface_dict['type'] = get_surface_type(surface)
                    if ind_s == 0:
                        surface_dict['d'] = surface.d.item() if isinstance(surface.d, torch.Tensor) else surface.d
                    else:
                        surface_dict['d'] = surface.d.item() - new_lens_surfaces[ind_s-1].d.item() if isinstance(surface.d, torch.Tensor) and isinstance(new_lens_surfaces[ind_s-1].d, torch.Tensor) else surface.d - new_lens_surfaces[ind_s-1].d
                    surface_dict['R'] = surface.r.item() if isinstance(surface.r, torch.Tensor) else surface.r
                    surface_dict['params'] = get_surface_params(surface)

                    #if (np.dot(lens_dir, mirrored_lens_dir) < 0) and (surface_dict['type'] == "XYPolynomial"):
                    #   if len(surface_dict['params']['ai']) == 3:
                    #        surface_dict['params']['ai'][2] = - surface_dict['params']['ai'][2]
                    
                    if (surface_dict['type'] == "XYPolynomial"):
                        # for i_ai in range(len(surface_dict['params']['ai'])):
                        #     surface_dict['params']['ai'][i_ai] = float(np.array([surface_dict['params']['ai'][i_ai]]))
                        if len(surface_dict['params']['ai']) == 3:
                            _, _, coef = compute_euler_angles(basic_dir, mirrored_lens_dir, rotation_order=self.list_rotation_order[lens_id])
                            coef = 2 * int((np.abs(coef).item() > 90.)) - 1
                            # if np.abs(coef) > 90.:
                            #     coef = 1
                            # else:
                            #     coef = -1
                            surface_dict['params']['ai'][2] = coef * surface_dict['params']['ai'][2]
                        surface_dict['params']['ai'] = surface_dict['params']['ai'].tolist()
                        #print("ai: ", surface_dict['params']['ai'])
                    new_surfaces.append(surface_dict)

                new_system_surfaces.append(new_surfaces)
                new_mats = self.systems_materials[lens_id].copy()
                new_mats.reverse()
                new_system_materials.append(new_mats)

                if lens_id == 0:
                    new_system_d_sensor.append(0.) #TODO May need a change
                else:
                    new_system_d_sensor.append((np.sign(np.dot(lens_dir, mirrored_lens_dir)) * self.list_d_sensor[lens_id]).item()) #TODO May need a change, as the values between old and new system should be different
                new_system_r_last.append(self.list_r_last[lens_id])
                #new_system_film_size.append(self.list_film_size[lens_id])
                new_system_film_size.append([int(self.list_film_size[lens_id][0]), int(self.list_film_size[lens_id][1])])

                new_system_pixel_size.append(self.list_pixel_size[lens_id])

                # new_system_theta_x.append(-np.sign(mirrored_basic_dir[0])*self.list_theta_x[lens_id])
                # new_system_theta_y.append(-np.sign(mirrored_basic_dir[1])*self.list_theta_y[lens_id])
                # new_system_theta_z.append(-np.sign(mirrored_basic_dir[2])*self.list_theta_z[lens_id]) # Because the z axis w.r.t. the origin is reversed

                basic_dir = np.array([0., 0., 1.])

                lens_dir = lens.to_world.transform_vector(basic_dir)
                mirrored_lens_dir = mirror_direction(lens_dir, ax_normal)

                mirrored_basic_dir = mirror_direction(basic_dir, ax_normal)

                # mirrored_basic_dir_12 = mirrored_basic_dir[[1,2]] / (np.linalg.norm(mirrored_basic_dir[[1,2]])) if np.linalg.norm(mirrored_basic_dir[[1,2]]) > 1e-10 else np.array([0., 0.])
                # mirrored_basic_dir_01 = mirrored_basic_dir[[0,1]] / (np.linalg.norm(mirrored_basic_dir[[0,1]])) if np.linalg.norm(mirrored_basic_dir[[0,1]]) > 1e-10 else np.array([0., 0.])
                # mirrored_basic_dir_02 = mirrored_basic_dir[[0,2]] / (np.linalg.norm(mirrored_basic_dir[[0,2]])) if np.linalg.norm(mirrored_basic_dir[[0,2]]) > 1e-10 else np.array([0., 0.])

                # mirrored_lens_dir_12 = mirrored_lens_dir[[1,2]] / (np.linalg.norm(mirrored_lens_dir[[1,2]])) if np.linalg.norm(mirrored_lens_dir[[1,2]]) > 1e-10 else np.array([0., 0.])
                # mirrored_lens_dir_01 = mirrored_lens_dir[[0,1]] / (np.linalg.norm(mirrored_lens_dir[[0,1]])) if np.linalg.norm(mirrored_lens_dir[[0,1]]) > 1e-10 else np.array([0., 0.])
                # mirrored_lens_dir_02 = mirrored_lens_dir[[0,2]] / (np.linalg.norm(mirrored_lens_dir[[0,2]])) if np.linalg.norm(mirrored_lens_dir[[0,2]]) > 1e-10 else np.array([0., 0.])

                # dot_12 = np.dot(mirrored_basic_dir_12, mirrored_lens_dir_12) if np.linalg.norm(mirrored_basic_dir_12) > 1e-10 and np.linalg.norm(mirrored_lens_dir_12) > 1e-10 else 1.
                # dot_02 = np.dot(mirrored_basic_dir_02, mirrored_lens_dir_02) if np.linalg.norm(mirrored_basic_dir_02) > 1e-10 and np.linalg.norm(mirrored_lens_dir_02) > 1e-10 else 1.
                # dot_01 = np.dot(mirrored_basic_dir_01, mirrored_lens_dir_01) if np.linalg.norm(mirrored_basic_dir_01) > 1e-10 and np.linalg.norm(mirrored_lens_dir_01) > 1e-10 else 1.
                
                print("")
                print("Directions: ", mirrored_basic_dir, mirrored_lens_dir)
                #print("Dot products: ", dot_12, dot_02, dot_01)
                #print("Angles: ", np.rad2deg(np.arccos(dot_12)), np.rad2deg(np.arccos(dot_02)), np.rad2deg(np.arccos(dot_01)))
                print("Previous angles: ", self.list_theta_x[lens_id], self.list_theta_y[lens_id], self.list_theta_z[lens_id])

                # dot_12 = np.dot(mirrored_lens_dir_12, np.array([0., 1.])) if np.linalg.norm(mirrored_lens_dir_12) > 1e-10 else 1.
                # dot_02 = np.dot(mirrored_lens_dir_02, np.array([0. , 1.])) if np.linalg.norm(mirrored_lens_dir_02) > 1e-10 else 1.
                # dot_01 = np.dot(mirrored_lens_dir_01, np.array([1. , 0.])) if np.linalg.norm(mirrored_lens_dir_01) > 1e-10 else 1.
                # dot_01 = 1.

                # print("Angles v2: ", np.rad2deg(np.arccos(dot_12)), np.rad2deg(np.arccos(dot_02)), np.rad2deg(np.arccos(dot_01)))

                # dot_basic_12 = np.dot(basic_dir[[1,2]], mirrored_basic_dir_12) if np.linalg.norm(basic_dir[[1,2]]) > 1e-10 else 1.
                # dot_basic_02 = np.dot(basic_dir[[0,2]], mirrored_basic_dir_02) if np.linalg.norm(basic_dir[[0,2]]) > 1e-10 else 1.
                # dot_basic_01 = np.dot(basic_dir[[0,1]], mirrored_basic_dir_01) if np.linalg.norm(basic_dir[[0,1]]) > 1e-10 else 1.

                # print("Angles v3: ", np.rad2deg(np.arccos(dot_basic_12)), np.rad2deg(np.arccos(dot_basic_02)), np.rad2deg(np.arccos(dot_basic_01)))

                # basic_dir = np.array([1., 0., 0.])
                # mirrored_basic_dir = mirror_direction(basic_dir, ax_normal)

                # dot_basic_12 = np.dot(basic_dir[[1,2]], mirrored_basic_dir[[1,2]]) if np.linalg.norm(basic_dir[[1,2]]) > 1e-10 else 1.
                # dot_basic_02 = np.dot(basic_dir[[0,2]], mirrored_basic_dir[[1,2]]) if np.linalg.norm(basic_dir[[0,2]]) > 1e-10 else 1.
                # dot_basic_01 = np.dot(basic_dir[[0,1]], mirrored_basic_dir[[1,2]]) if np.linalg.norm(basic_dir[[0,1]]) > 1e-10 else 1.

                # print("Angles v4: ", np.rad2deg(np.arccos(dot_basic_12)), np.rad2deg(np.arccos(dot_basic_02)), np.rad2deg(np.arccos(dot_basic_01)))

                # theta_x_ = 0. #np.sign(mirrored_basic_dir[2]) * np.rad2deg(np.arccos(dot_12)) + np.rad2deg(np.arccos(dot_basic_12))
                # theta_y_ = - np.sign(np.dot(lens_dir, mirrored_lens_dir)) * np.rad2deg(np.arccos(dot_02)) + np.rad2deg(np.arccos(dot_basic_02))
                # #TODO sign problem on theta_y_. Origins and shifts are correct but there are problems on signs of d and of rotations. Need to be checked
                # theta_z_ = np.rad2deg(np.arccos(dot_01))

                
                # dot_origin_12 = np.dot(mirrored_lens_dir_12, np.array([0., 1.])) if np.linalg.norm(mirrored_lens_dir_12) > 1e-10 else 0.
                # dot_origin_02 = np.dot(mirrored_lens_dir_02, np.array([0., 1.])) if np.linalg.norm(mirrored_lens_dir_02) > 1e-10 else 0.
                # dot_origin_01 = np.dot(mirrored_lens_dir_01, np.array([1., 0.])) if np.linalg.norm(mirrored_lens_dir_01) > 1e-10 else 0.
                
                # det_origin_12 = np.linalg.det(np.array([mirrored_lens_dir_12, np.array([0., 1.])])) if np.linalg.norm(mirrored_lens_dir_12) > 1e-10 else 0.
                # det_origin_02 = np.linalg.det(np.array([mirrored_lens_dir_02, np.array([0., 1.])])) if np.linalg.norm(mirrored_lens_dir_02) > 1e-10 else 0.
                # det_origin_01 = np.linalg.det(np.array([mirrored_lens_dir_01, np.array([1., 0.])])) if np.linalg.norm(mirrored_lens_dir_01) > 1e-10 else 0.

                # theta_x_ = - np.rad2deg(np.arctan2(det_origin_12, dot_origin_12))
                # theta_y_ = np.rad2deg(np.arctan2(det_origin_02, dot_origin_02))
                # theta_z_ = np.rad2deg(np.arctan2(det_origin_01, dot_origin_01))*0.

                # if lens_id == 1:
                #     theta_x_, theta_y_, theta_z_ = [ 175.00031772,  -19.68009774, -151.74373034]

                # if np.abs(theta_x_) == 180.:
                #     theta_x_ = 0.
                # if theta_z_ == 180.:
                #     theta_z_ = 180.

                theta_x_, theta_y_, theta_z_ = compute_euler_angles(basic_dir, mirrored_lens_dir, rotation_order=self.list_rotation_order[lens_id])

                theta_z_ = 0.

                print("New angles: ", theta_x_, theta_y_, theta_z_)

                new_system_theta_x.append(theta_x_.item() if isinstance(theta_x_, np.floating) else theta_x_)
                new_system_theta_y.append(theta_y_.item() if isinstance(theta_y_, np.floating) else theta_y_)
                new_system_theta_z.append(theta_z_)

                # new_system_theta_x.append(np.rad2deg(np.arctan2(mirrored_lens_dir[2], mirrored_lens_dir[1])))
                # new_system_theta_y.append(np.rad2deg(np.arctan2(mirrored_lens_dir[2], mirrored_lens_dir[0])))
                # new_system_theta_z.append(np.rad2deg(np.arctan2(mirrored_lens_dir[1], mirrored_lens_dir[0]))) # Because the z axis w.r.t. the origin is reversed

                # new_system_theta_x.append(coef_rot[0]*self.list_theta_x[lens_id])
                # new_system_theta_y.append(coef_rot[1]*self.list_theta_y[lens_id])
                # new_system_theta_z.append(coef_rot[2]*self.list_theta_z[lens_id]) # Because the z axis w.r.t. the origin is reversed

                new_origin = self.list_origin[lens_id].copy() if self.list_origin[lens_id] is not None else [0., 0., 0.]
                new_origin = mirror_point(new_origin, ax_position, ax_normal)
                new_system_origin.append(new_origin.tolist())
                
                new_shift = self.list_shift[lens_id].copy() if self.list_shift[lens_id] is not None else [0., 0., 0.]
                
                new_shift = (mirrored_basic_dir * new_shift)
                #new_shift = mirror_point(new_shift, ax_position, ax_normal)
                new_system_shift.append(new_shift.tolist())

                new_system_rotation_order.append(self.list_rotation_order[lens_id])

        else:
            raise NotImplementedError("Symmetry axis not recognized")

        self.systems_surfaces = self.systems_surfaces + new_system_surfaces
        self.systems_materials = self.systems_materials + new_system_materials
        self.list_d_sensor = self.list_d_sensor + new_system_d_sensor
        self.list_r_last = self.list_r_last + new_system_r_last
        self.list_film_size = self.list_film_size + new_system_film_size
        self.list_pixel_size = self.list_pixel_size + new_system_pixel_size
        self.list_theta_x = self.list_theta_x + new_system_theta_x
        self.list_theta_y = self.list_theta_y + new_system_theta_y
        self.list_theta_z = self.list_theta_z + new_system_theta_z
        self.list_origin = self.list_origin + new_system_origin
        self.list_shift = self.list_shift + new_system_shift
        self.list_rotation_order = self.list_rotation_order + new_system_rotation_order

        self.create_system(self.list_d_sensor, self.list_r_last, self.list_film_size, self.list_pixel_size, self.list_theta_x, self.list_theta_y, self.list_theta_z, self.list_origin, self.list_shift, self.list_rotation_order)
        self.pos_dispersed, self.pixel_dispersed = self.central_positions_wavelengths(self.wavelengths) # In x,y coordinates for pos, in lines, columns (y, x) for pixel


def get_surface_type(surface):
    """
    Get the type of the surface.

    Args:
        surface (do.Surface): The surface object.

    Returns:
        str: The type of the surface.
    """
    if isinstance(surface, do.Aspheric):
        return "Aspheric"
    elif isinstance(surface, do.XYPolynomial):
        return "XYPolynomial"
    elif isinstance(surface, do.BSpline):
        return "BSpline"
    elif isinstance(surface, do.ThinLens):
        return "ThinLens"
    elif isinstance(surface, do.ThinLenslet):
        return "ThinLenslet"
    elif isinstance(surface, do.FocusThinLens):
        return "FocusThinLens"
    elif isinstance(surface, do.Mirror):
        return "Mirror"
    else:
        raise ValueError("Surface type not recognized")

def get_surface_params(surface):
    """
    Get the parameters of the surface.

    Args:
        surface (do.Surface): The surface object.

    Returns:
        dict: The parameters of the surface.
    """
    if isinstance(surface, do.Aspheric):
        return {'c': surface.c, 'k': surface.k, 'ai': surface.ai, 'is_square': surface.is_square}
    elif isinstance(surface, do.XYPolynomial):
        return {'J': surface.J, 'ai': surface.ai, 'b': float(surface.b), 'is_square': surface.is_square}
    elif isinstance(surface, do.BSpline):
        return {'size': surface.size, 'px': surface.px, 'py': surface.py, 'tx': surface.tx, 'ty': surface.ty, 'c': surface.c, 'is_square': surface.is_square}
    elif isinstance(surface, do.ThinLens):
        return {'f': float(surface.f), 'is_square': surface.is_square}
    elif isinstance(surface, do.ThinLenslet):
        return {'f': float(surface.f), 'r0': surface.r0, 'is_square': surface.is_square}
    elif isinstance(surface, do.FocusThinLens):
        return {'f': float(surface.f), 'is_square': surface.is_square}
    elif isinstance(surface, do.Mirror):
        return {'is_square': surface.is_square}
    else:
        raise ValueError("Surface type not recognized")
    
def compute_euler_angles(vector1, vector2, rotation_order='xyz'):
    """
    Compute Euler angles to rotate vector1 to align with vector2 using the specified rotation order.
    Returns angles in degrees (theta_x, theta_y, theta_z).
    """
    # Normalize input vectors
    v1 = vector1 / np.linalg.norm(vector1)
    v2 = vector2 / np.linalg.norm(vector2)
    
    # Handle edge case: vectors are parallel or anti-parallel
    if np.allclose(v1, v2):
        return 0.0, 0.0, 0.0
    if np.allclose(v1, -v2):
        # 180° rotation around an arbitrary axis (here, first axis in the rotation order)
        return (180.0, 0.0, 0.0) if rotation_order[0] == 'x' else (0.0, 180.0, 0.0) if rotation_order[0] == 'y' else (0.0, 0.0, 180.0)
    
    # Compute rotation matrix using axis-angle
    axis = np.cross(v1, v2)
    axis /= np.linalg.norm(axis)
    angle = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
    
    # Axis-angle to rotation matrix
    c = np.cos(angle)
    s = np.sin(angle)
    t = 1 - c
    x, y, z = axis
    R = np.array([
        [t*x*x + c,    t*x*y - s*z,  t*x*z + s*y],
        [t*x*y + s*z,  t*y*y + c,    t*y*z - s*x],
        [t*x*z - s*y,  t*y*z + s*x,  t*z*z + c]
    ])
    
    # Extract Euler angles based on rotation order
    rotation_order = rotation_order.lower()
    if rotation_order == 'xyz':
        # XYZ order
        theta_y = np.arcsin(-R[2, 0])
        if np.cos(theta_y) > 1e-6:
            theta_x = np.arctan2(R[2, 1], R[2, 2])
            theta_z = np.arctan2(R[1, 0], R[0, 0])
        else:
            theta_x = np.arctan2(-R[0, 1], R[1, 1])
            theta_z = 0.0
    elif rotation_order == 'xzy':
        # XZY order
        theta_z = np.arcsin(R[1, 0])
        if np.cos(theta_z) > 1e-6:
            theta_x = np.arctan2(-R[1, 2], R[1, 1])
            theta_y = np.arctan2(-R[2, 0], R[0, 0])
        else:
            theta_x = np.arctan2(R[2, 1], R[2, 2])
            theta_y = 0.0
    elif rotation_order == 'yxz':
        # YXZ order
        theta_x = np.arcsin(R[2, 1])
        if np.cos(theta_x) > 1e-6:
            theta_y = np.arctan2(-R[2, 0], R[2, 2])
            theta_z = np.arctan2(-R[0, 1], R[1, 1])
        else:
            theta_y = np.arctan2(R[0, 2], R[0, 0])
            theta_z = 0.0
    elif rotation_order == 'yzx':
        # YZX order
        theta_z = np.arcsin(-R[0, 1])
        if np.cos(theta_z) > 1e-6:
            theta_y = np.arctan2(R[0, 2], R[0, 0])
            theta_x = np.arctan2(R[2, 1], R[1, 1])
        else:
            theta_y = np.arctan2(-R[2, 0], R[2, 2])
            theta_x = 0.0
    elif rotation_order == 'zxy':
        # ZXY order
        theta_x = np.arcsin(R[1, 2])
        if np.cos(theta_x) > 1e-6:
            theta_z = np.arctan2(-R[1, 0], R[1, 1])
            theta_y = np.arctan2(-R[0, 2], R[2, 2])
        else:
            theta_z = np.arctan2(R[2, 0], R[0, 0])
            theta_y = 0.0
    elif rotation_order == 'zyx':
        # ZYX order
        theta_y = np.arcsin(-R[0, 2])
        if np.cos(theta_y) > 1e-6:
            theta_z = np.arctan2(R[0, 1], R[0, 0])
            theta_x = np.arctan2(R[1, 2], R[2, 2])
        else:
            theta_z = np.arctan2(-R[1, 0], R[1, 1])
            theta_x = 0.0
    else:
        raise ValueError("Invalid rotation_order. Supported: 'xyz', 'xzy', 'yxz', 'yzx', 'zxy', 'zyx'")
    
    # Convert to degrees and return in order (theta_x, theta_y, theta_z)
    angles_rad = [theta_x, theta_y, theta_z]
    angles_deg = np.degrees(angles_rad)
    return angles_deg