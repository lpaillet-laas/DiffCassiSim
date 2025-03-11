import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
import matplotlib
from tqdm import tqdm

import sys
sys.path.append("../")
import diffoptics as do

import time

import yaml

import cProfile

import h5py

class HSSystem:
    def __init__(self, list_systems_surfaces=None, list_systems_materials=None,
                 list_d_sensor=None, list_r_last=None, list_film_size=None, list_pixel_size=None, list_theta_x=None, list_theta_y=None, list_theta_z=None, list_origin=None, list_shift=None, list_rotation_order=None,
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
            self.list_theta_x = list_theta_x
            self.list_theta_y = list_theta_y
            self.list_theta_z = list_theta_z
            self.list_origin = list_origin
            self.list_shift = list_shift
            self.list_rotation_order = list_rotation_order
        
        self.entry_radius = self.systems_surfaces[0][0]['R']

        self.create_system(self.list_d_sensor, self.list_r_last, self.list_film_size, self.list_pixel_size, self.list_theta_x, self.list_theta_y, self.list_theta_z, self.list_origin, self.list_shift, self.list_rotation_order)
        self.pos_dispersed, self.pixel_dispersed = self.central_positions_wavelengths(self.wavelengths) # In x,y coordinates for pos, in lines, columns (y, x) for pixel

    
    def create_lensgroup(self, surfaces, materials, d_sensor, r_last, film_size=None, pixel_size=None,
                     theta_x=0., theta_y=0., theta_z=0., origin=np.zeros(3), shift=np.zeros(3), rotation_order='xyz'):
        """
        Create a lens group object based on the given parameters. Distances in the surfaces are relative to the last surface.

        Args:
            surfaces (list[dict]): List of dictionaries representing the surfaces. Each dictionary should contain the following keys:
                - type (str): Type of the surface. Can be 'Aspheric', 'XYPolynomial', or 'BSpline'.
                - R (float): Radius of the surface.
                - d (float): Distance from the last surface.
                - params (dict): Parameters of the surface. The allowed fields depend on the surface type:
                    - For 'Aspheric': 'c' (float), 'k' (float), 'ai' (float or None), 'is_square' (bool).
                    - For 'XYPolynomial': 'J' (float), 'ai' (float or None), 'b' (float or None), 'is_square' (bool).
                    - For 'BSpline': 'size' (tuple[int, int]) necessary, 'px' (int), 'py' (int), 'tx' (float or None), 'ty' (float or None), 'c' (float or None), 'is_square' (bool).
                    - For 'ThinLens': 'f' (float)
                    - For 'FocusThinLens': 'f' (float)
            materials (list[str]): List of the names of the desired materials.
            d_sensor (float): Distance from the origin to the sensor.
            r_last (float): Radius of the last surface.
            film_size (tuple[int, int], optional): Film size of the sensor in x, y coordinates. Defaults to None.
            pixel_size (float, optional): Pixel size of the sensor. Defaults to None.
            theta_x (float, optional): Rotation angle (in degrees) along the x axis for the entire lens group. Defaults to 0.
            theta_y (float, optional): Rotation angle (in degrees) along the y axis for the entire lens group. Defaults to 0.
            theta_z (float, optional): Rotation angle (in degrees) along the z axis for the entire lens group. Defaults to 0.
            origin (tuple[float, float, float], optional): Origin position in x, y, z coordinates. Defaults to (0, 0, 0).
            shift (tuple[float, float, float], optional): Shift of the lens group relative to the origin, in x, y, z coordinates. Defaults to (0, 0, 0).
            rotation_order (str, optional): Operation order for the computation of rotation matrices. Defaults to 'xyz'.

        Returns:
            Lensgroup: Lens group object representing the created lens group.
        """
        surfaces_processed = []
        accumulated_d = 0
        for surface in surfaces:
            if surface['type'] == 'Aspheric':
                arguments = ['c', 'k', 'ai', 'is_square']
                for argument in arguments:
                    if argument not in surface['params']:
                        if argument in ['c', 'k']:
                            surface['params'][argument] = 0.
                        elif argument == 'ai':
                            surface['params'][argument] = None
                        elif argument == 'is_square':
                            surface['params'][argument] = False

                surfaces_processed.append(do.Aspheric(
                    surface['R'], surface['d']+accumulated_d, c=surface['params']['c'], ai=surface['params']['ai'], is_square=surface['params']['is_square'], device=self.device
                ))
            elif surface['type'] == 'XYPolynomial':
                arguments = ['J', 'ai', 'b', 'is_square']
                for argument in arguments:
                    if argument not in surface['params']:
                        if argument == 'J':
                            surface['params'][argument] = 0.
                        elif argument in ['ai', 'b']:
                            surface['params'][argument] = None
                        elif argument == 'is_square':
                            surface['params'][argument] = False
                surfaces_processed.append(do.XYPolynomial(
                    surface['R'], surface['d']+accumulated_d, J=surface['params']['J'], ai=surface['params']['ai'], b=surface['params']['b'], is_square=surface['params']['is_square'], device=self.device
                ))
            elif surface['type'] == 'BSpline':
                arguments = ['size', 'px', 'py', 'tx', 'ty', 'c', 'is_square']
                for argument in arguments:
                    if argument not in surface['params']:
                        if argument =='size':
                            raise AttributeError('size must be provided for BSpline surface.')
                        elif argument in ['px', 'py']:
                            surface['params'][argument] = 3
                        elif argument in ['tx', 'ty', 'c']:
                            surface['params'][argument] = None
                        elif argument == 'is_square':
                            surface['params'][argument] = False
                surfaces_processed.append(do.BSpline(
                    surface['R'], surface['d']+accumulated_d, size=surface['params']['size'], px=surface['params']['px'], py=surface['params']['py'], tx=surface['params']['tx'], ty=surface['params']['ty'], c=surface['params']['c'], is_square=surface['params']['is_square'], device=self.device
                ))
            elif surface['type'] == 'ThinLens':
                arguments = ['f', 'is_square']
                for argument in arguments:
                    if argument not in surface['params']:
                        if argument == 'f':
                            surface['params'][argument] = 0.
                        elif argument == 'is_square':
                            surface['params'][argument] = False
                surfaces_processed.append(do.ThinLens(
                    surface['R'], surface['d'] + accumulated_d, surface['params']['f'], is_square=surface['params']['is_square'], device=self.device
                ))
            elif surface['type'] == 'ThinLenslet':
                arguments = ['f', 'is_square', 'r0']
                for argument in arguments:
                    if argument not in surface['params']:
                        if argument == 'f':
                            surface['params'][argument] = 0.
                        elif argument == 'is_square':
                            surface['params'][argument] = False
                        elif argument == 'r0':
                            surface['params'][argument] = 0.
                surfaces_processed.append(do.ThinLenslet(
                    surface['R'], surface['params']['r0'], surface['d'] + accumulated_d, surface['params']['f'], is_square=surface['params']['is_square'], device=self.device
                ))
            elif surface['type'] == 'FocusThinLens':
                arguments = ['f', 'is_square']
                for argument in arguments:
                    if argument not in surface['params']:
                        if argument == 'f':
                            surface['params'][argument] = 0.
                        elif argument == 'is_square':
                            surface['params'][argument] = False
                surfaces_processed.append(do.FocusThinLens(
                    surface['R'], surface['d'] + accumulated_d, surface['params']['f'], device=self.device
                ))
            elif surface['type'] == 'Mirror':
                arguments = ['is_square']
                for argument in arguments:
                    if argument not in surface['params']:
                        if argument == 'is_square':
                            surface['params'][argument] = False
                surfaces_processed.append(do.Mirror(
                    surface['R'], surface['d'] + accumulated_d, is_square=surface['params']['is_square'], device=self.device
                ))

            accumulated_d += surface['d']
        
        materials_processed = []
        for material in materials:
            materials_processed.append(do.Material(material))

        lens = do.Lensgroup(device=self.device, theta_x=theta_x, theta_y=theta_y, theta_z=theta_z, origin=origin, shift=shift, rotation_order=rotation_order)
        lens.load(surfaces_processed, materials_processed)
        lens.d_sensor = d_sensor
        lens.r_last = r_last
        lens.film_size = film_size
        lens.pixel_size = pixel_size

        return lens

    def create_system(self, list_d_sensor, list_r_last, list_film_size, list_pixel_size, list_theta_x, list_theta_y, list_theta_z, list_origin, list_shift, list_rotation_order):
        """
        Create a system of lens groups based on the given parameters.
        Refer to create_lensgroup for the parameters of each lens group.
        """
        n = len(self.systems_surfaces)
        self.size_system = n
        self.system = [None for i in range(n)]

        if list_film_size is None:
            list_film_size = [0. for i in range(n)]
        if list_pixel_size is None:
            list_pixel_size = [0. for i in range(n)]
        if list_theta_x is None:
            list_theta_x = [0. for i in range(n)]
        if list_theta_y is None:
            list_theta_y = [0. for i in range(n)]
        if list_theta_z is None:
            list_theta_z = [0. for i in range(n)]
        if list_origin is None:
            list_origin = [np.zeros(3) for i in range(n)]
        if list_shift is None:
            list_shift = [np.zeros(3) for i in range(n)]
        if list_rotation_order is None:
            list_rotation_order = ['xyz' for i in range(n)]
            self.list_rotation_order = list_rotation_order

        for i in range(n):
            lens_surface = self.systems_surfaces[i]
            lens_material = self.systems_materials[i]
            d_sensor = list_d_sensor[i]
            r_last = list_r_last[i]
            film_size = [int(list_film_size[i][0]), int(list_film_size[i][1])]
            pixel_size = list_pixel_size[i]
            theta_x = list_theta_x[i]
            theta_y = list_theta_y[i]
            theta_z = list_theta_z[i]
            if list_origin[i] is None:
                origin = np.zeros(3)
            else:
                origin = list_origin[i]
            if list_shift[i] is None:
                shift = np.zeros(3)
            else:
                shift = list_shift[i]
            if list_rotation_order[i] is None:
                rotation_order = 'xyz'
            else:
                rotation_order = list_rotation_order[i]
            system = self.create_lensgroup(surfaces=lens_surface, materials=lens_material,
                                           d_sensor=d_sensor, r_last=r_last, film_size=film_size, pixel_size=pixel_size,
                                           theta_x=theta_x, theta_y=theta_y, theta_z=theta_z, origin=origin, shift=shift, rotation_order=rotation_order)
            self.system[i] = system

        self.d_subsystems = torch.zeros(n, device=self.device)
        return self.system
    
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
        system_dict['wavelengths'] = self.wavelengths.tolist()
        system_dict['device'] = str(self.device)
        system_dict['save_dir'] = str(self.save_dir)

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
        self.wavelengths = torch.tensor(system_dict['wavelengths']).float()
        self.device = system_dict['device']
        self.save_dir = system_dict['save_dir']

    def update_system(self, d_subsystems=None):
        """
        Update the system with the provided distances between the subsystems.

        Args:
            d_subsystems (torch.Tensor, optional): The distances between the subsystems. Defaults to None.
        
        Returns:
            list[Lensgroup]: The updated system.
        """
        if d_subsystems is None:
            d_subsystems = self.d_subsystems
        
        d_cum_subsystems = torch.cumsum(d_subsystems, dim=0)

        for s, system in enumerate(self.system):
            for surf in range(len(system.surfaces)):
                system.surfaces[surf].d += d_cum_subsystems[s]
                self.systems_surfaces[s][surf]['d'] += d_cum_subsystems[s]
            system.d_sensor += d_cum_subsystems[s]
            self.list_d_sensor[s] += d_cum_subsystems[s]
        
        return self.system
        
        
    
    def trace_all(self, R=None, M=5, wavelength=550):
        """
        Trace rays through the lenses and calculate the ray positions and optical path lengths (oss).

        Args:
            R (float): The radius of the incoming beam of rays.
            M (int): The number of rays to sample.
            wavelength (float): The wavelength of the light.

        Returns:
            tuple: A tuple containing the ray positions (ps) and optical path lengths (oss).
        """
        if R is None:
            R = 12.7

        # Generate initial ray positions and optical path lengths
        ray_mid = self.system[0].sample_ray_2D(R, wavelength, M=M)
        oss = [0 for i in range(self.size_system)]
        
        # Trace rays through each lens in the system
        for i, lens in enumerate(self.system[:-1]):
            ray_mid, valid, oss_temp = lens.trace_r(ray_mid)
            oss[i] = oss_temp
        
        # Trace rays to the sensor
        ps, oss_last = self.system[-1].trace_to_sensor_r(ray_mid)
        oss[-1] = oss_last
        
        # Return the ray positions and optical path lengths
        return ps[...,:2], oss
    
    
    def prepare_mts(self, lens_id, pixel_size, film_size, R=np.eye(3), t=np.zeros(3), start_distance=0.0):
        """
        Prepares the lens for multi-surface tracing (MTS) by setting the necessary parameters. This function should be called before rendering the lens.
        It is based on the original prepare_mts function in the diffoptics library, with the option to specify the starting distance for the lens surfaces to allow for smoother rendering with several lens groups.

        Args:
            lens_id (int): The index of the lens object to prepare for MTS.
            pixel_size (float): The size of a pixel in millimeters.
            film_size (length 2 int array): The size of the film in pixels.
            R (np.ndarray, optional): The rotation matrix representing the transformation of the lensgroup. Defaults to np.eye(3).
            t (np.ndarray, optional): The translation vector representing the transformation of the lensgroup. Defaults to np.zeros(3).
            start_distance (float, optional): The starting distance for the lens surfaces. Defaults to 0.0.
        """
        lens = self.system[lens_id]
        if lens.mts_prepared:
            print('MTS already prepared for this lensgroup.')
            return
            
        # sensor parameters
        lens.pixel_size = pixel_size # [mm]
        lens.film_size = film_size # [pixel]

        # rendering parameters
        lens.mts_Rt = do.Transformation(R, t) # transformation of the lensgroup
        lens.mts_Rt.to(self.device)

        # for visualization
        lens.r_last = lens.pixel_size * max(lens.film_size) / 2

        # treat the lenspart as a camera; append one more surface to it
        if lens_id == -1:
            lens.surfaces.append(do.Aspheric(lens.r_last, lens.d_sensor, 0.0))
        #lens.surfaces.append(do.Aspheric(lens.r_last, lens.d_sensor, 0.0))
        
        # reverse surfaces
        d_total = torch.tensor([lens.surfaces[-1].d])
        #print("D_total: ", d_total)
        for i in range(len(lens.surfaces)):
            #lens.surfaces[i].d = (d_total - lens.surfaces[i].d + start_distance).item()
            lens.surfaces[i].d = - lens.surfaces[i].d
            lens.surfaces[i].reverse()
        lens.surfaces.reverse()
        if lens_id == -1:
            lens.surfaces.pop(0) # remove sensor plane
        #lens.surfaces.pop(0) # remove sensor plane
        # reverse materials
        lens.materials.reverse()

        # aperture plane
        lens.aperture_radius = lens.surfaces[0].r
        #lens.aperture_distance = torch.tensor([lens.surfaces[0].d])

        #print("d_sensor", lens.d_sensor)
        #lens.aperture_distance = (start_distance - lens.origin[2]) - lens.shift[2] + torch.tensor([lens.surfaces[0].d])
        lens.aperture_distance = lens.d_sensor + lens.surfaces[0].d# + lens.origin[2] + lens.shift[2]
        #print("Lens: ", lens)
        #print("aperture dist: ", lens.aperture_distance)
        lens.mts_prepared = True
        #lens.d_sensor = 0

        #lens.origin = torch.tensor([lens.origin[0], - lens.origin[1], start_distance + d_total - lens.origin[2]]).float().to(device=self.device)
        #print("Sd: ", start_distance)
        #print("Total: ", d_total)

        ######lens.origin = torch.tensor([lens.origin[0], - lens.origin[1], start_distance + d_total*np.cos(lens.theta_y*np.pi/180).item()]).float().to(device=self.device)
        lens.origin = torch.tensor([lens.origin[0], - lens.origin[1], start_distance - lens.origin[2]]).float().to(device=self.device)
        lens.d_sensor = - lens.d_sensor
        #lens.d_sensor = - lens.d_sensor - lens.surfaces[0].d
        lens.shift = torch.tensor([lens.shift[0], - lens.shift[1], - lens.shift[2]]).float().to(device=self.device)
        lens.theta_x *= -1
        lens.theta_y *= -1
        R_, t_ = lens._compute_transformation().R, lens._compute_transformation().t
        lens.mts_Rt = do.Transformation(R_, t_) # transformation of the lensgroup
        lens.mts_Rt.to(self.device)
        #print(lens.rotation_order)
        #print("Pixel size: ", lens.pixel_size)
        lens.update()
        self.system[lens_id] = lens

    def render_single_back(self, wavelength, screen, numerical_aperture = 0.05, save_pos = False):
        """
        Renders back propagation of single ray through a series of lenses onto a screen.

        Args:
            wavelength (float): The wavelength of the light.
            screen (object): The screen object onto which the light is projected.
            numerical_aperture (float, optional): The numerical aperture of the system. Defaults to 0.05.
            save_pos (bool, optional): Whether to save the positions of the resulting rays (True), or return a rendered image (False). Defaults to False.
        Returns:
            tuple: A tuple containing the intensity values (I) and the mask indicating valid pixels on the screen.
        """
        # Sample rays from the sensor
        ####valid, ray_mid = self.system[-1].sample_ray_sensor(wavelength.item(), numerical_aperture = numerical_aperture)
        valid, ray_mid = self.system[0].sample_ray_sensor(wavelength.item(), numerical_aperture = numerical_aperture)
        #valid_start = valid.clone()

        # ray_mid.o = ray_mid.o[valid, :]
        # ray_mid.d = ray_mid.d[valid, :]
        #print("Ray 1: ", ray_mid)
        #print("Nb valid 1: ", torch.sum(valid))
        # Trace rays through each lens in the system
        if len(self.system) > 1:
            ####for lens in self.system[::-1][1:-1]:
            for lens in self.system[1:-1]:
                ray_mid = lens.to_object.transform_ray(ray_mid)
                valid_1, ray_mid = lens._trace(ray_mid)
                #ray_mid.o = ray_mid.o[valid_1, :]
                #ray_mid.d = ray_mid.d[valid_1, :]
                #ray_mid = lens.mts_Rt.transform_ray(ray_mid)
                ray_mid = lens.to_world.transform_ray(ray_mid)
                #print(f"Ray mid: ", ray_mid)
                #print("Nb valid mid: ", torch.sum(valid_1))
                valid = valid & valid_1

            # Trace rays to the first lens
            ####ray_mid = self.system[0].to_object.transform_ray(ray_mid)
            ray_mid = self.system[-1].to_object.transform_ray(ray_mid)
            ####valid_last, ray_last = self.system[0]._trace(ray_mid)
            valid_last, ray_last = self.system[-1]._trace(ray_mid)
            #print("Nb valid last: ", torch.sum(valid_last))
            valid_last = valid & valid_last
            #print("Ray last before transform: ", ray_last)

            #ray_last = self.system[0].mts_Rt.transform_ray(ray_last)
            ####ray_last = self.system[0].to_world.transform_ray(ray_last)
            ray_last = self.system[-1].to_world.transform_ray(ray_last)
            #print("Ray last: ", ray_last)
        else:
            ray_last = ray_mid
            valid_last = valid
        

        # Intersect rays with the screen
        uv, valid_screen = screen.intersect(ray_last)[1:]
        # Apply mask to filter out invalid rays
        mask = valid_last & valid_screen
        
        print("Ratio valid rays: ", (torch.sum(mask)/(screen.texture.shape[0]*screen.texture.shape[1])).item())

        if save_pos:
            return uv, mask
        else:
            # Calculate intensity values on the screen
            I = screen.shading(uv, mask)
            
            return I, mask

    def render_single_back_save_pos(self, wavelength, screen, numerical_aperture = 0.05):
        """
        Renders back propagation of single ray through a series of lenses onto a screen. Used to save the positions of the resulting rays.

        Args:
            wavelength (float): The wavelength of the light.
            screen (object): The screen object onto which the light is projected.
            numerical_aperture (float, optional): The numerical aperture of the system. Defaults to 0.05.

        Returns:
            tuple: A tuple containing the rays positions (uv) and the mask indicating valid pixels on the screen.
        """
        uv, mask = self.render_single_back(wavelength, screen, numerical_aperture, save_pos = True)

        return uv, mask
    
    def save_pos_render(self, texture = None, nb_rays=20, wavelengths = [656.2725, 587.5618, 486.1327], z0=0, offsets=None, numerical_aperture = 0.05):
        """
        Renders a dummy image to save the positions of rays passing through the optical system.
        Args:
            texture (ndarray, optional): The texture image to be rendered. Defaults to None.
            nb_rays (int, optional): The number of rays to be rendered per pixel. Defaults to 20.
            wavelengths (list, optional): The wavelengths of the rays to be rendered. Defaults to [656.2725, 587.5618, 486.1327].
            z0 (int, optional): The z-coordinate of the screen. Defaults to 0.
            offsets (list, optional): The offsets for each lens in the system. Defaults to None.
            numerical_aperture (int, optional): The reduction factor for the aperture. Defaults to 0.05.
        Returns:
            tuple: A tuple containing the positions of the rays (big_uv) and the corresponding mask (big_mask).
        """            

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
        print("Texture nonzero: ", texture_torch.count_nonzero())
        # render
        ray_counts_per_pixel = nb_rays
        time_start = time.time()
        big_uv = torch.zeros((nb_wavelengths, ray_counts_per_pixel, self.system[0].film_size[0]*self.system[0].film_size[1], 2), dtype=torch.float,  device=self.device)
        big_mask = torch.zeros((nb_wavelengths, ray_counts_per_pixel, self.system[0].film_size[0]*self.system[0].film_size[1]), dtype=torch.bool, device=self.device)
        screen.update_texture(texture_torch[..., 0])

        for wavelength_id, wavelength in enumerate(wavelengths):
            # multi-pass rendering by sampling the aperture
            for i in tqdm(range(ray_counts_per_pixel)):
                uv, mask = self.render_single_back_save_pos(wavelength, screen, numerical_aperture=numerical_aperture)
                big_uv[wavelength_id, i, :, :] = uv
                big_mask[wavelength_id, i, :] = mask

            print(f"Elapsed rendering time: {time.time()-time_start:.3f}s for wavelength {wavelength_id+1}")
        return big_uv, big_mask

    def render_based_on_saved_pos(self, big_uv = None, big_mask = None, texture = None, nb_rays=20, wavelengths = [656.2725, 587.5618, 486.1327],
                                  z0=0, save=False, plot = False):
        """
        Renders an image based on the saved ray positions and validity.

        Args:
            big_uv (ndarray, optional): Array of UV coordinates. Defaults to None.
            big_mask (ndarray, optional): Array of masks. Defaults to None.
            texture (ndarray, optional): Texture array. Defaults to None.
            nb_rays (int, optional): Number of rays. Defaults to 20.
            wavelengths (list, optional): List of wavelengths. Defaults to [656.2725, 587.5618, 486.1327].
            z0 (int, optional): Z-coordinate. Defaults to 0.
            save (bool, optional): Whether to save the rendered image. Defaults to False.
            plot (bool, optional): Whether to plot the rendered image. Defaults to False.

        Returns:
            ndarray: Rendered image.
        """
        # create a dummy screen
        pixelsize = self.system[0].pixel_size # [mm]

        nb_wavelengths = len(wavelengths)

        # default texture
        if texture is None:
            size_pattern = tuple(self.system[0].film_size)
            texture = np.zeros(size_pattern + (nb_wavelengths,)).astype(np.float32)
            texture[int(0.1*size_pattern[0]):-int(0.3*size_pattern[0]), size_pattern[1]//2 + 0*4, :] = 1                # Big vertical
            texture[size_pattern[0]//2, int(0.2*size_pattern[1]):int(0.8*size_pattern[1]), :] = 1                 # Horizontal
            texture[int(0.4*size_pattern[0]):int(0.6*size_pattern[0]), int(0.7*size_pattern[1]) + 0*4, :] = 1           # Small vertical

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
        print("Texture nonzero: ", texture_torch.count_nonzero())

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

    def render_batch_based_on_saved_pos(self, big_uv = None, big_mask = None, texture = None, nb_rays=20, wavelengths = [656.2725, 587.5618, 486.1327],
                                  z0=0):
        """
        Renders an image based on the saved ray positions and validity.

        Args:
            big_uv (ndarray, optional): Array of UV coordinates. Defaults to None.
            big_mask (ndarray, optional): Array of masks. Defaults to None.
            texture (ndarray, optional): Texture array. Defaults to None.
            nb_rays (int, optional): Number of rays. Defaults to 20.
            wavelengths (list, optional): List of wavelengths. Defaults to [656.2725, 587.5618, 486.1327].
            z0 (int, optional): Z-coordinate. Defaults to 0.

        Returns:
            ndarray: Rendered image.
        """
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
        ####print("Texture nonzero: ", texture_torch.count_nonzero())

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
    
    def render_all_based_on_saved_pos(self, big_uv = None, big_mask = None, texture = None, nb_rays=20, wavelengths = [656.2725, 587.5618, 486.1327],
                                  z0=0):
        """
        Renders an image based on the saved ray positions and validity.

        Args:
            big_uv (ndarray, optional): Array of UV coordinates. Defaults to None.
            big_mask (ndarray, optional): Array of masks. Defaults to None.
            texture (ndarray, optional): Texture array. Defaults to None.
            nb_rays (int, optional): Number of rays. Defaults to 20.
            wavelengths (list, optional): List of wavelengths. Defaults to [656.2725, 587.5618, 486.1327].
            z0 (int, optional): Z-coordinate. Defaults to 0.

        Returns:
            ndarray: Rendered image.
        """
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
        ####print("Texture nonzero: ", texture_torch.count_nonzero())

        # render
        print("Simulating acquisition")
        screen.update_texture_all(texture_torch.permute(0, 3, 1, 2))

        # multi-pass rendering by sampling the aperture
        nb_cut = 4
        I = 0
        for cut in range(nb_cut):
            Imid = screen.shading_all(big_uv[:, big_uv.shape[1]//nb_cut*cut:big_uv.shape[1]//nb_cut*(cut+1), :, :], big_mask[:, big_uv.shape[1]//nb_cut*cut:big_uv.shape[1]//nb_cut*(cut+1), :]) # [batchsize, nC, nb_rays, N]
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
            #print("Surface ", i)
            #print("Origin: ", lens.origin)
            #print("Shift: ", lens.shift)
            #print("D final: ", lens.surfaces[-1].d)
        if plot:
            self.combined_plot_setup(with_sensor=False)

        self.system = self.system[::-1] # The system is reversed
        # create a dummy screen
        pixelsize = self.system[-1].pixel_size # [mm]

        nb_wavelengths = len(wavelengths)

        # default texture
        if texture is None:
            size_pattern = tuple(self.system[0].film_size)
            texture = np.zeros(size_pattern + (nb_wavelengths,)).astype(np.float32)
            texture[int(0.1*size_pattern[0]):-int(0.3*size_pattern[0]), size_pattern[1]//2 + 0*4, :] = 1                # Big vertical
            texture[size_pattern[0]//2, int(0.2*size_pattern[1]):int(0.8*size_pattern[1]), :] = 1                 # Horizontal
            texture[int(0.4*size_pattern[0]):int(0.6*size_pattern[0]), int(0.7*size_pattern[1]) + 0*4, :] = 1           # Small vertical

        if plot and (nb_wavelengths == 3):
            plt.figure()
            plt.plot()
            plt.imshow(texture.detach().cpu().numpy())
            plt.show()

        texture_torch = torch.Tensor(texture).float().to(device=self.device)
        # texture_torch = torch.permute(texture_torch, (1,0,2)) # Permute
        # texture_torch = texture_torch.flip(dims=[0]) # Flip
        texture_torch = texture_torch.rot90(1, dims=[0, 1])

        texturesize = np.array(texture_torch.shape[0:2])
        
        screen = do.Screen(
            do.Transformation(np.eye(3), np.array([0, 0, z0])),
            texturesize * pixelsize, texture_torch, device=self.device
        )
        #print("Texture nonzero: ", texture_torch.count_nonzero())

        # render
        ray_counts_per_pixel = nb_rays
        time_start = time.time()
        Is = []

        #shift_value = shift_value.int()
        for wavelength_id, wavelength in enumerate(wavelengths):
            screen.update_texture(texture_torch[..., wavelength_id])

            # multi-pass rendering by sampling the aperture
            I = 0
            M = 0
            for i in tqdm(range(ray_counts_per_pixel)):
                I_current, mask = self.render_single_back(wavelength, screen, numerical_aperture=numerical_aperture)
                I = I + I_current
                M = M + mask
            I = I / (M + 1e-10)
            # reshape data to a 2D image
            #print(f"Image {wavelength_id} nonzero count: {I.count_nonzero()}")

            I = I.reshape(*np.flip(np.asarray(self.system[0].film_size)))
            Is.append(I.cpu())
        # show image
        I_rendered = torch.stack(Is, axis=-1)
        I_rendered_plot = I_rendered.detach().cpu().numpy()#.flip(0)#.astype(np.uint8)
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

    def sample_rays_pos(self, wavelength, angles, x_pos, y_pos, z_pos, d = None):
        """
        Samples rays with given wavelength, angles, and positions or direction. Giving the direction instead of the angles is the preferred method.

        Args:
            wavelength (float): The wavelength of the rays.
            angles (list): A list of angles in degrees, where each angle is represented as a tuple (phi, psi).
            x_pos (float): The x-coordinate of the position.
            y_pos (float): The y-coordinate of the position.
            z_pos (float): The z-coordinate of the position.
            d (ndarray, optional): The direction of the rays. Preferred method. Defaults to None.

        Returns:
            ray (do.Ray): A Ray object representing the sampled rays.
        """
        
        if d is None:
            # Convert angles from degrees to radians
            angles_rad = torch.tensor(np.radians(np.asarray(angles)))
            angles_phi = angles_rad[:,0]
            angles_psi = angles_rad[:,1]
            
            # Create position tensor
            pos = torch.tensor([x_pos, y_pos, z_pos]).repeat(angles_rad.shape[0], 1).float()

            # Calculate ray origin and direction
            o = pos
            d = torch.stack((
                torch.sin(angles_phi),
                torch.sin(angles_psi),
                torch.cos(angles_phi)*torch.cos(angles_psi)), axis=-1
            ).float()
        # If d is provided, use it to create the rays, preferred method
        else:
            pos = torch.tensor([x_pos, y_pos, z_pos]).repeat(d.shape[0], 1).float()

            o = pos
            d = torch.Tensor(d).float() if type(d) == np.ndarray else d.float()

        # Normalize the direction vectors
        d = d/torch.norm(d, p=2, dim=1)[:, None]

        return do.Ray(o, d, wavelength, device=self.device)
    
    def extract_hexapolar_dir(self, nb_centric_circles, source_pos, max_angle):
        """
        Extracts the direction of rays based on a hexapolar distribution with nb_centric_cirles circles.

        Parameters:
            nb_centric_circles (int): The number of hexapolars.
            source_pos (torch.Tensor): The position of the source.
            max_angle (float): The maximum angle in degrees from the center of the circle to the generated points.

        Returns:
            dist (torch.Tensor): The extracted hexapolar direction.
        """
        z = max(torch.norm(source_pos - torch.tensor([-self.entry_radius, 0])), torch.norm(source_pos - torch.tensor([self.entry_radius, 0])),
                            torch.norm(source_pos - torch.tensor([0, -self.entry_radius])), torch.norm(source_pos - torch.tensor([0, self.entry_radius])))
        
        z = self.system[0].surfaces[0].d + self.system[0].origin[-1] + self.system[0].shift[-1]#TODO: Check if this is correct
        x, y = draw_circle_hexapolar(nb_centric_circles, z, max_angle, center_x = 0, center_y = 0)

        #dist = torch.stack((x, y, self.system[0].surfaces[0].d*torch.ones(x.shape[0])), dim=-1)
        dist = torch.stack((x, y, z*torch.ones(x.shape[0])), dim=-1)

        return dist
    
    def trace_psf_from_point_source(self, angles=[[0, 0]], x_pos=0, y_pos=0, z_pos=0, wavelength = 550,
                 show_rays = True, d = None, ignore_invalid = False, show_res = True):
        """
        Traces the point spread function (PSF) from a point source through a series of lenses.

        Args:
            angles (list): angles (list): A list of angles in degrees, where each angle is represented as a tuple (phi, psi) (default: [[0,0]]).
            x_pos (int): X position of the point source (default: 0)
            y_pos (int): Y position of the point source (default: 0)
            z_pos (int): Z position of the point source (default: 0)
            wavelength (int): Wavelength of the rays in nanometers (default: 550)
            show_rays (bool): Flag indicating whether to show the plots with rays (default: True)
            d (ndarray, optional): Direction of the rays. Defaults to None.
            ignore_invalid (bool): Flag indicating whether to ignore invalid rays (default: False)
            show_res (bool): Flag indicating whether to show the results (default: True)

        Returns:
            ps (array): Array containing the x and y coordinates of the PSF
        """
        
        if self.system is None:
            raise AttributeError('lenses must be provided')
        
        # sample wavelengths in [nm]
        wavelength = torch.Tensor([wavelength]).float().to(self.device)

        ray = self.sample_rays_pos(wavelength, angles, x_pos, y_pos, z_pos, d = d)
        """ # Plot the position of the rays when they arrive on the first lens
        pos = ray(torch.tensor([self.system[0].surfaces[0].d]))
        if show_res:
            plt.scatter(pos[:,1], pos[:,0])
            plt.axis('scaled')
            plt.title('Entry of first lens')
        plt.show() """
        
        oss = [None for i in range(self.size_system)]

        # Trace rays through each lens in the system
        for i, lens in enumerate(self.system[:-1]):
            if show_rays:
                ray, valid, oss_mid = lens.trace_r(ray)
                oss[i] = oss_mid
            else:
                ray, valid = lens.trace(ray)
            if not ignore_invalid:
                ray.o = ray.o[valid, :]
                ray.d = ray.d[valid, :]
        """ # Plot the position of the rays when they arrive on the second lens
        pos = ray(torch.tensor([np.cos(-9.088)*self.F]))
        if show_res:
            plt.figure()
            plt.scatter(pos[:,1], pos[:,0])
            plt.axis('scaled')
            plt.title('Entry of second lens') """
        
        # Trace rays to the sensor
        if show_rays:
            ps, oss_final = self.system[-1].trace_to_sensor_r(ray, ignore_invalid=True)
            oss[-1] = oss_final
            ax, fig = self.plot_setup_with_rays(oss)
            if self.save_dir is not None:
                fig.savefig(os.path.join(self.save_dir, "setup_with_rays.svg"), format="svg")
        else:
            ps = self.system[-1].trace_to_sensor(ray, ignore_invalid=True)

        if not show_res:
            plt.close()
        return ps[...,:2]
    
    def plot_spot_diagram(self, wavelength, nb_pixels, size_pixel, normalize=False, show=True, nb_pts_x=3, nb_pts_y=3):
        """
        Generates a spot diagram for the given set of lenses.

        Args:
            wavelength (float): The wavelength of the light.
            nb_pixels (int): The number of pixels in the real grid.
            size_pixel (float): The size of each pixel in the real grid.
            normalize (bool, optional): Whether to normalize the spot diagram. Defaults to False.
            show (bool, optional): Whether to display the spot diagram. Defaults to True.

        Returns:
            numpy.ndarray: The spot diagram coordinates.
        """
        ray = self.system[0].sample_ray(wavelength, M=nb_pixels, R=nb_pixels*size_pixel/2, sampling='grid')

        # CASSI mesh 
        x, y = torch.meshgrid(
            torch.arange(-(nb_pixels-1)*size_pixel/2, (nb_pixels+1)*size_pixel/2, size_pixel, device=self.device),
            torch.arange(-(nb_pixels-1)*size_pixel/2, (nb_pixels+1)*size_pixel/2, size_pixel, device=self.device),
            indexing='xy' # ij in original code
        )

        x, y = torch.meshgrid(
            torch.linspace(-2.5, 2.5, nb_pts_x, device=self.device),
            torch.linspace(-2.5, 2.5, nb_pts_y, device=self.device),
            indexing = 'xy'
        )

        d = torch.zeros((1,1,3), device=self.device)
        d[0,0,-1] = 1
        d = d.repeat(x.shape[0], x.shape[1], 1)

        if self.save_dir is not None:
            np.save(self.save_dir + "grid.npy", np.stack((x.cpu().detach().numpy().flatten(),y.cpu().detach().numpy().flatten()), axis=-1))

        o = torch.stack((x,y,torch.zeros_like(x, device=self.device)), axis=2)
        ray.o = o
        ray.d = d
        for i, lens in enumerate(self.system[:-1]):
            ray, valid = lens.trace(ray)
        ps = self.system[-1].trace_to_sensor(ray)
        #self.system[-1].spot_diagram(ps, xlims=[-nb_pixels*size_pixel/2*1.5, nb_pixels*size_pixel/2*1.5], ylims=[-nb_pixels*size_pixel/2*1.5, nb_pixels*size_pixel/2*1.5], savepath=self.save_dir + "spotdiagram.png", normalize=normalize, show=show)
        ps = ps.cpu().detach().numpy()
        return ps[...,:2]
    
    def plot_spot_less_points(self, nb_pixels, size_pixel, opposite = [False, False], shift = [0., 0.], wavelengths = [450., 550., 650.]):
        """
        Compare the spot diagram of the lens system with a reference spot diagram.

        Args:
            nb_pixels (int): The number of pixels in the real grid. Has to be a square number.
            size_pixel (float): The size of each pixel in the real grid.
            opposite (list, optional): Whether to compute the opposite of the spot diagram along the two dimensions. Defaults to [True, False].
            shift (list, optional): The shift to apply to the spot diagram along the two dimensions. Defaults to [0., 0.].
            wavelengths (list, optional): The wavelengths to compare. Defaults to [450., 550., 650.].
        """

        sq = int(np.sqrt(nb_pixels))
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.rcParams['text.usetex'] = True
        plt.rcParams['font.family'] = 'serif'
        for w_id, w in enumerate(wavelengths):
            ps = self.plot_spot_diagram(w, nb_pixels, size_pixel, show=False, nb_pts_x=3, nb_pts_y=3)

            """ps = ps[sq-1::sq,:]
            new_ps = np.zeros((ps.shape[0]//sq, 2))

            for i in range(sq):
                new_ps[i*sq:(i+1)*sq,:] = np.flip(ps[nb_pixels*i:nb_pixels*i+sq,:], axis=0)
            ps = new_ps """

            if opposite[0]:
                ps[..., 0] = - ps[..., 0]
            if opposite[1]:
                ps[..., 1] = - ps[..., 1]
            ps[..., 0] += shift[0]
            ps[..., 1] += shift[1]

            ps = self.plot_spot_diagram(w, nb_pixels, size_pixel, show=False, nb_pts_x=20, nb_pts_y=3)

            if w_id==0:
                for i in range(3):
                    plt.plot(ps[i*20:(i+1)*20,0], ps[i*20:(i+1)*20,1], color='b', label = '_nolegend_')
            elif w_id==1:
                for i in range(3):
                    plt.plot(ps[i*20:(i+1)*20,0], ps[i*20:(i+1)*20,1], color='g', label = '_nolegend_')
            elif w_id==2:
                for i in range(3):
                    plt.plot(ps[i*20:(i+1)*20,0], ps[i*20:(i+1)*20,1], color='r', label = '_nolegend_')
            
            ps = self.plot_spot_diagram(w, nb_pixels, size_pixel, show=False, nb_pts_x=3, nb_pts_y=20)
            if w_id==0:
                for i in range(3):
                    plt.plot(ps[i::3,0], ps[i::3,1], color='b', label = '_nolegend_')
            elif w_id==1:
                for i in range(3):
                    plt.plot(ps[i::3,0], ps[i::3,1], color='g', label = '_nolegend_')
            elif w_id==2:
                for i in range(3):
                    plt.plot(ps[i::3,0], ps[i::3,1], color='r', label = '_nolegend_')
        for w_id, w in enumerate(wavelengths):
            ps = self.plot_spot_diagram(w, nb_pixels, size_pixel, show=False, nb_pts_x=3, nb_pts_y=3)

            if w_id==0:
                plt.scatter(ps[...,0], ps[...,1], color='b')
            elif w_id==1:
                plt.scatter(ps[...,0], ps[...,1], color='g')
            elif w_id==2:
                plt.scatter(ps[...,0], ps[...,1], color='r')
        
        plt.xticks([-3, -2, -1, 0, 1, 2, 3])
        plt.yticks([-2, -1, 0, 1, 2])
        ax.tick_params(axis='both', which='major', width=5/2.5, length=20/2.5)
        plt.grid(True)
        plt.legend([f'{wavelengths[0]} nm', f'{wavelengths[1]} nm', f'{wavelengths[2]} nm'])
        if self.save_dir is not None:
            plt.savefig(self.save_dir + "spotdiagram.svg", format="svg", bbox_inches='tight', pad_inches = 0)
        plt.show()
    
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
        
        # Plot the setup of the last lens with the sensor
        self.system[-1].plot_setup2D(ax=ax, fig=fig, with_sensor=True, show=True)
        
        # Return the figure and axes objects
        return fig, ax
    
    def plot_setup_basic_rays(self, radius = None):
        """
        Plot the setup with a beam of rays.

        Args:
            radius (float, optional): The radius of the beam of rays. Defaults to None.
        Returns:
            ax (matplotlib.axes.Axes): The matplotlib axes object.
            fig (matplotlib.figure.Figure): The matplotlib figure object.
        """
        if radius is None:
            radius = self.entry_radius
        ps, oss = self.trace_all(R=radius)
        return self.plot_setup_with_rays(oss)

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
        
        # Plot the raytraces for the last lens with the sensor
        ax, fig = self.system[-1].plot_raytraces(oss[-1], ax=ax, fig=fig, color=color, linewidth=linewidth, show=show, with_sensor=True)
        
        return ax, fig

    def compare_spot_zemax(self, path_compare='./'):
        """
        Compare the spot diagram of the lens system with a reference spot diagram.

        Args:
            path_compare (str, optional): The path to the reference spot diagram model. Defaults to './'.
        
        Returns:
            torch.Tensor: The mean difference between the spot of the system and of Zemax for the system's wavelengths.
        """
        plt.rcParams['text.usetex'] = True
        plt.rcParams['font.family'] = 'serif'
        #params = {'text.latex.preamble': r'\usepackage{siunitx} \usepackage{sfmath} \sisetup{detect-family = true} \usepackage{amsmath}'}
        params = {'axes.labelsize': 90/2.5,'axes.titlesize':90/2.5, 'legend.fontsize': 90/2.5, 'xtick.labelsize': 70/2.5, 'ytick.labelsize': 70/2.5}
        matplotlib.rcParams.update(params)
        plt.rcParams.update(params)

        wavelengths = torch.tensor([450., 520., 650.])
        mean_diff = torch.zeros((len(wavelengths)))
        list_diff = []
        list_diff_field = []
        
        shift_per_w = self.central_positions_wavelengths(wavelengths)[0]

        for w_id, w in enumerate(wavelengths):
            file_path = path_compare + f'grid_distorsions_{int(w)}nm.txt'
            _, x_field, y_field, x_real, y_real = parse_grid_distortion_data(file_path)
            prop_x_ref = x_real.flatten()
            prop_y_ref = y_real.flatten()
            prop_ref = np.stack((prop_x_ref, prop_y_ref), axis=-1)

            x_field = x_field.flatten()
            y_field = y_field.flatten()

            o = torch.stack((x_field, y_field, torch.zeros_like(x_field, device=self.device)), axis=-1).float()
            d = torch.stack((torch.zeros_like(x_field, device=self.device),
                             torch.zeros_like(x_field, device=self.device),
                             torch.ones_like(x_field, device=self.device)), axis=-1).float()
            ray = do.Ray(o, d, w, device=self.device)
            for i, lens in enumerate(self.system[:-1]):
                ray, valid = lens.trace(ray)
            ps = self.system[-1].trace_to_sensor(ray)
            ps = ps.cpu().detach().numpy()
            ps= ps[...,:2]

            plt.figure()
            plt.scatter(prop_x_ref, prop_y_ref, color='r', s=15)
            plt.scatter(ps[...,0], ps[...,1], color='b', s=10, marker='*')

            plt.title(f'Spot diagram at {w.int().item()}nm')
            plt.xlabel('x [mm]')
            plt.ylabel('y [mm]')
            plt.legend(['dO', 'Zemax'])
            if self.save_dir is not None:
                plt.savefig(self.save_dir + f'spot_diagram_{w.int().item()}nm.svg', format='svg')
            field_test = y_field.reshape((int(np.sqrt(x_real.shape).item()), int(np.sqrt(x_real.shape).item()))).numpy()

            diff = np.linalg.norm(ps - prop_ref, axis=-1).reshape((int(np.sqrt(x_real.shape).item()), int(np.sqrt(x_real.shape).item())))
            diff = np.rot90(diff, k=1, axes=(0, 1))
            list_diff.append(diff)
            

            mean_diff[w_id] = np.mean(diff)
            print(f"Average distance at {w.int().item()}nm: {mean_diff[w_id]:.4f}mm")

            x_field_shifted = x_field -shift_per_w[w_id, 0]
            y_field_shifted = y_field -shift_per_w[w_id, 1]
            diff_field = np.linalg.norm(ps - torch.stack((-x_field_shifted, -y_field_shifted), axis=-1).numpy(), axis=-1).reshape((int(np.sqrt(x_real.shape).item()), int(np.sqrt(x_real.shape).item())))
            diff_field = np.rot90(diff_field, k=1, axes=(0, 1))
            list_diff_field.append(diff_field)
            
        list_diff = 1000*np.stack(list_diff, axis=-1)
        list_diff_field = 1000*np.stack(list_diff_field, axis=-1)

        for i, w in enumerate(wavelengths):
            diff = list_diff[..., i]
            diff_field = list_diff_field[..., i]
            fig = plt.figure(figsize=(32/2.5, 18/2.5), dpi=60*2.5)
            ax = fig.add_subplot(111)
            #plt.rcParams.update({'font.size': 90})
            #vmin_zemax = list_diff.min()
            #vmax_zemax = list_diff.max()
            vmin_zemax = 0.0017079563141009224
            vmax_zemax = 4.64173844285655
            plt.imshow(diff, extent=[x_field.min()*1000, x_field.max()*1000, y_field.min()*1000, y_field.max()*1000], vmin=vmin_zemax, vmax=vmax_zemax)
            cb = plt.colorbar(label=r'Distance [µm]')
            plt.title(f'Distortion map with respect to Zemax at {w.int().item()}nm')
            ax.set_xlabel(r"x [µm]")
            ax.set_ylabel(r"y [µm]")
            ax.tick_params(axis='both', which='major', width=5/2.5, length=20/2.5)
            cb.ax.tick_params(axis='both', which='major', width=5/2.5, length=20/2.5)
            plt.xticks([-2000, -1000, 0, 1000, 2000])
            plt.yticks([-2000, -1000, 0, 1000, 2000])
            if self.save_dir is not None:
                plt.savefig(self.save_dir + f'distortion_map_wrt_zemax_{w.int().item()}nm.svg', format='svg', bbox_inches = 'tight', pad_inches = 0)

            fig = plt.figure(figsize=(32/2.5, 18/2.5), dpi=60*2.5)
            ax = fig.add_subplot(111)
            #plt.rcParams.update({'font.size': 90})
            
            plt.imshow(diff_field, extent=[x_field.min()*1000, x_field.max()*1000, y_field.min()*1000, y_field.max()*1000], vmin=list_diff_field.min(), vmax=list_diff_field.max())
            cb = plt.colorbar(label=r'Distance [µm]')
            plt.title(f'Distortion map relative to the grid at {w.int().item()}nm')
            ax.set_xlabel(r"x [µm]")
            ax.set_ylabel(r"y [µm]")
            ax.tick_params(axis='both', which='major', width=5/2.5, length=20/2.5)
            cb.ax.tick_params(axis='both', which='major', width=5/2.5, length=20/2.5)
            plt.xticks([-2000, -1000, 0, 1000, 2000])
            plt.yticks([-2000, -1000, 0, 1000, 2000])
            if self.save_dir is not None:
                plt.savefig(self.save_dir + f'distortion_map_wrt_field_{w.int().item()}nm.svg', format='svg', bbox_inches = 'tight', pad_inches = 0)
        print(f"Diff min: {list_diff.min()} diff max: {list_diff.max()}")
        plt.show()
        return mean_diff

    def get_mapping_scene_detector(self, wavelengths, shape_scene = [512, 512]):
        """
        Create the mapping cube from the scene to the detector for a given set of wavelengths.

        Args:
            wavelengths (list): The wavelengths to consider.
            shape_scene (list, optional): The shape of the scene. Defaults to [512, 512].
        
        Returns:
            torch.Tensor: The mapping cube from the scene to the detector.
        """
        x_field = np.linspace(-1/2, 1/2, shape_scene[0])*shape_scene[0]*self.system[-1].pixel_size
        y_field = np.linspace(-1/2, 1/2, shape_scene[1])*shape_scene[1]*self.system[-1].pixel_size

        x, y = np.meshgrid(x_field, y_field)
        x = torch.from_numpy(x).float().to(self.device)
        y = torch.from_numpy(y).float().to(self.device)

        o = torch.stack((x, y, torch.zeros_like(x, device=self.device)), axis=-1).float()
        d = torch.stack((torch.zeros_like(x, device=self.device),
                            torch.zeros_like(x, device=self.device),
                            torch.ones_like(x, device=self.device)), axis=-1).float()
                        
        mapping_cube = torch.zeros((shape_scene[0], shape_scene[1], len(wavelengths), 2), device=self.device, dtype=torch.int)

        film_size = torch.tensor(self.system[-1].film_size, device=self.device)
        for w_id, w in enumerate(wavelengths):
            ray = do.Ray(o, d, w, device=self.device)
            for i, lens in enumerate(self.system[:-1]):
                ray, valid = lens.trace(ray)
            ps = self.system[-1].trace_to_sensor(ray)
            ps = (ps[:, :2]/self.system[-1].pixel_size + film_size[None, :]//2).flip(1).reshape(shape_scene[0], shape_scene[1], 2) # Flip because x, y -> column, line
            ps = torch.stack((torch.clamp(ps[..., 0], min=0, max=self.system[-1].film_size[1]-1), torch.clamp(ps[..., 1], min=0, max=self.system[-1].film_size[0]-1)), dim=-1)
            mapping_cube[:, :, w_id, :] = ps.int()

        return mapping_cube
    
    def create_simple_mapping(self, wavelengths, shape_scene = [512, 512], remove_y = False):
        """
        Create a simplified mapping from the scene to the detector for a given set of wavelengths, without taking into account the potential distortions.

        Args:
            wavelengths (list): The wavelengths to consider.
            shape_scene (list, optional): The shape of the scene. Defaults to [512, 512].
            remove_y (bool, optional): Whether to remove the y-coordinate spectral spreading. Defaults to False.
        
        Returns:
            torch.Tensor: The mapping cube from the scene to the detector.
        """
        x_field = np.linspace(-1/2, 1/2, shape_scene[0])*shape_scene[0]*self.system[-1].pixel_size
        y_field = np.linspace(-1/2, 1/2, shape_scene[1])*shape_scene[1]*self.system[-1].pixel_size

        x, y = np.meshgrid(x_field, y_field)
        x = torch.from_numpy(x).float().to(self.device)
        y = torch.from_numpy(y).float().to(self.device)

        positions = torch.stack((x,y), axis=-1).reshape(-1, 2).float()

        pos_dispersed = self.central_positions_wavelengths(torch.linspace(450, 650, 28))[0]

        if remove_y:
            pos_dispersed[:, 1] = 0.
                        
        mapping_cube = torch.zeros((shape_scene[0], shape_scene[1], len(wavelengths), 2), device=self.device, dtype=torch.int)

        film_size = torch.tensor(self.system[-1].film_size, device=self.device)
        for w_id, w in enumerate(wavelengths):
            #ps = positions + pos_dispersed[-1-w_id, :].unsqueeze(0)
            ps = positions - pos_dispersed[w_id, :].unsqueeze(0)
            ps = (ps[:, :2]/self.system[-1].pixel_size + film_size[None, :]//2).flip(1).reshape(shape_scene[0], shape_scene[1], 2) # Flip because x, y -> column, line
            ps = torch.stack((torch.clamp(ps[..., 0], min=0, max=self.system[-1].film_size[1]-1), torch.clamp(ps[..., 1], min=0, max=self.system[-1].film_size[0]-1)), dim=-1)
            mapping_cube[:, :, w_id, :] = ps.int()
        
        mapping_cube[:,:,:, 0] = film_size[1]-1 - mapping_cube[:,:,:, 0]
        mapping_cube[:,:,:, 1] = film_size[0]-1 - mapping_cube[:,:,:, 1]
        return mapping_cube

    def render(self, wavelengths=[450., 550., 650.], nb_rays=1, z0=0, texture=None, offsets=None, numerical_aperture = 0.05, plot=True):
        """
        Renders the texture with the optical system.
        Args:
            wavelengths (list, optional): List of wavelengths to propagate. Defaults to [450., 550., 650.].
            nb_rays (int, optional): Number of rays to propagate. Defaults to 1.
            z0 (int, optional): Initial z-coordinate. Should be approximately equal to the distance from the sensor to the origin. Defaults to 0.
            texture (ndarray, optional): Texture array. Defaults to None.
            offsets (ndarray, optional): Offsets array. Defaults to None.
            numerical_aperture (int, optional): Aperture reduction factor. Defaults to 1.
            plot (bool, optional): Whether to plot the rendered image. Defaults to True.
        """

        return self.propagate(wavelengths = wavelengths, nb_rays = nb_rays, z0 = z0,
                  texture = texture, offsets = offsets, numerical_aperture=numerical_aperture, plot = plot)
    
    def central_positions_wavelengths(self, wavelengths):
        """
        Calculates where the central position of the scene is traced on the sensor for given wavelengths.

        Args:
            wavelengths (list): A list of wavelengths for which to calculate the central dispersion.

        Returns:
            torch.Tensor: A tensor containing the positions of the central rays for each wavelength.
                          The tensor has shape (len(wavelengths), 2).
        """
        # Initialize a chief ray
        o = torch.zeros((1,1,3), device=self.device)
        d = torch.zeros((1,1,3), device=self.device)
        d[0,0,-1] = 1

        results_pos = torch.empty((len(wavelengths), 2), device=self.device)
        for w, wav in enumerate(wavelengths):
            ray = do.Ray(o, d, wav, device=self.device)
            for i, lens in enumerate(self.system[:-1]):
                ray, valid = lens.trace(ray)
            ps = self.system[-1].trace_to_sensor(ray)
            results_pos[w, :] = ps[...,:2]
        return results_pos, torch.mul(results_pos, 1/self.system[-1].pixel_size).flip(1) # Flip because x, y -> column, line

    def compare_positions_trace(self, nb_centric_circles, list_source_pos, max_angle, wavelength, colors=None):
        """
        Compare the positions of rays traced from different source positions.
        Parameters:
            nb_centric_circles (int): Number of hexapolars.
            list_source_pos (list): List of source positions.
            max_angle (float): Maximum angle.
            wavelength (float): Wavelength of the rays.
            colors (list, optional): List of colors for plotting. Defaults to None.
        Returns:
            None
        """

        if colors is None:
            colors = ['b-' for i in range(len(list_source_pos))]
        ax, fig = None, None
        for ind, source_pos in enumerate(list_source_pos):
            d = self.extract_hexapolar_dir(nb_centric_circles, source_pos, max_angle) 

            wavelength = torch.Tensor([wavelength]).float().to(self.device)

            ray = self.sample_rays_pos(wavelength, None, source_pos[0], source_pos[1], 0., d = d)
            #ray = do.Ray(o = torch.tensor([2.5, 0., 0.]).repeat(2, 1).float(), d = torch.stack((torch.tensor([0., 0.]), torch.tensor([0., 0.]), torch.tensor([1., 1.])), dim=-1), wavelength = wavelength)
            # ray.o[..., 2] = ray.o[..., 2]-50
            # ray.d[..., 2] *= -1
            # print(ray)
            oss = [None for i in range(self.size_system)]
            #print(ray)

            # Trace rays through each lens in the system
            for i, lens in enumerate(self.system[:-1]):
                ray, valid, oss_mid = lens.trace_r(ray)
                oss[i] = oss_mid

                ray.o = ray.o[valid, :]
                ray.d = ray.d[valid, :]
            
            # Trace rays to the sensor
            ps, oss_final = self.system[-1].trace_to_sensor_r(ray, ignore_invalid=True)
            oss[-1] = oss_final
            ax, fig = self.plot_setup_with_rays(oss, ax=ax, fig=fig, show=False, color=colors[ind])
            #print(oss)
        if self.save_dir is not None:
            fig.savefig(os.path.join(self.save_dir, "setup_with_rays.svg"), format="svg")
        plt.show()

    def compare_wavelength_trace(self, nb_centric_circles, list_source_pos, max_angle, wavelengths, colors=None, linewidth=1.0):
        """
        Compare the wavelength trace for different wavelengths.
        Args:
            nb_centric_circles (int): Number of hexapolars.
            list_source_pos (list): List of source positions.
            max_angle (float): Maximum angle.
            wavelengths (list): List of wavelengths to compare.
            colors (list, optional): List of colors for each wavelength for plotting. Defaults to None.
            linewidth (float, optional): Width of the rays. Defaults to 1.0.
        Returns:
            None
        """

        if colors is None:
            colors = ['b-' for i in range(len(wavelengths))]
        colors = colors[::-1]
        ax, fig = None, None
        for ind, source_pos in enumerate(list_source_pos):
            d = self.extract_hexapolar_dir(nb_centric_circles, source_pos, max_angle) 
            for w_id, w in enumerate(wavelengths[::-1]):
                wavelength = torch.Tensor([w]).float().to(self.device)

                ray = self.sample_rays_pos(wavelength, None, source_pos[0], source_pos[1], 0., d = d)
                
                oss = [0 for i in range(self.size_system)]

                # Trace rays through each lens in the system
                for i, lens in enumerate(self.system[:-1]):
                    ray, valid, oss_mid = lens.trace_r(ray)
                    oss[i] = oss_mid

                    ray.o = ray.o[valid, :]
                    ray.d = ray.d[valid, :]
                
                # Trace rays to the sensor
                ps, oss_final = self.system[-1].trace_to_sensor_r(ray, ignore_invalid=True)
                oss[-1] = oss_final
                ax, fig = self.plot_setup_with_rays(oss, ax=ax, fig=fig, show=False, color=colors[w_id], linewidth=linewidth)
                if self.save_dir is not None:
                    fig.savefig(os.path.join(self.save_dir, "setup_with_rays.svg"), format="svg")
            if ind ==0:
                plt.legend([f"{w:.0f} nm" for w in wavelengths[::-1]], labelcolor=[color.replace('-','') for color in colors])
        plt.show()

    def trace_psf(self, nb_centric_circles, source_pos, max_angle, wavelength):
        """
        Plot the point spread function (PSF) for a given set of parameters.
        Args:
            nb_centric_circles (int): Number of centric circles in the hexapolar grid. Will define the number of rays.
            source_pos (numpy.ndarray): Position of the source in the format [x, y].
            max_angle (float): Maximum angle for the hexapolar grid.
            wavelength (float): Wavelength of the light source.
        Returns:
            None
        """
        if not torch.is_tensor(source_pos):
            source_pos = torch.from_numpy(source_pos).float()
        d = self.extract_hexapolar_dir(nb_centric_circles, source_pos, max_angle) 

        ps = self.trace_psf_from_point_source(angles = None, x_pos = source_pos[0], y_pos = source_pos[1], z_pos = 0., wavelength = wavelength,
                        show_rays = False, d = d, ignore_invalid = False, show_res = False)

        return ps
        
    def plot_psf(self, nb_centric_circles, source_pos, max_angle, wavelength, show_rays = False, show_res = False):
        """
        Plot the point spread function (PSF) for a given set of parameters.
        Args:
            nb_centric_circles (int): Number of centric circles in the hexapolar grid. Will define the number of rays.
            source_pos (numpy.ndarray): Position of the source in the format [x, y].
            max_angle (float): Maximum angle for the hexapolar grid.
            wavelength (float): Wavelength of the light source.
            show_rays (bool, optional): Whether to show the rays in the plot. Defaults to False.
            show_res (bool, optional): Whether to show the different lenses rays go through in the plot. Defaults to False.
        Returns:
            None
        """

        d = self.extract_hexapolar_dir(nb_centric_circles, source_pos, max_angle) 

        time_start = time.time()
        ps = self.trace_psf_from_point_source(angles = None, x_pos = source_pos[0], y_pos = source_pos[1], z_pos = 0., wavelength = wavelength,
                        show_rays = show_rays, d = d, ignore_invalid = False, show_res = show_res)

        print(f"Time elapsed: {time.time() - time_start:.3f} seconds")
        ps_plot = ps.clone().detach().cpu()
        plt.figure()
        plt.scatter(ps_plot[..., 1], ps_plot[...,0], s=0.1)
        plt.show()
        
        fig, ax = plt.subplots()
        hist = torch.histogramdd(ps_plot.flip(1), bins=11, density=False)
        hist, edges = hist.hist.numpy(), hist.bin_edges

        ax.hist2d(ps_plot[...,1], ps_plot[...,0], bins=11)
        ax.axis('equal')
        plt.show()
        return ps[...,:2]
    
    def compare_psf(self, nb_centric_circles, params, max_angle, pixel_size, kernel_size=11, show_rays = False, show_res = False):
        """
        Plot the point spread function (PSF) for a given set of parameters.
        Args:
            nb_centric_circles (int): Number of centric circles in the hexapolar grid. Will define the number of rays.
            params (list): List of parameters to compare. Each parameter is a tuple (source_pos, wavelength, ps_zemax).
            max_angle (float): Maximum angle for the hexapolar grid.
            pixel_size (float): Size of the pixels in the sensor.
            kernel_size (int, optional): Size of the kernel for the histogram. Defaults to 11.
            show_rays (bool, optional): Whether to show the rays in the plot. Defaults to False.
            show_res (bool, optional): Whether to show the different lenses rays go through in the plot. Defaults to False.
        Returns:
            None
        """
        plt.rcParams['text.usetex'] = True
        plt.rcParams['font.family'] = 'serif'
        matplotlib.rcParams['axes.linewidth'] = 1
        
        for k in range(len(params)):
            d = self.extract_hexapolar_dir(nb_centric_circles, params[k][0], max_angle)
            ps = self.trace_psf_from_point_source(angles = None, x_pos = params[k][0][0], y_pos = params[k][0][1], z_pos = 0., wavelength = params[k][1],
                    show_rays = show_rays, d = d, ignore_invalid = False, show_res = show_res)
            bins_i, bins_j, centroid_ps = find_bins(ps, pixel_size, kernel_size)
            hist_ps = torch.histogramdd(ps, bins=(bins_i, bins_j), density=False).hist
            hist_ps /= hist_ps.sum()

            ps_zemax = params[k][2]
            ps_zemax = torch.stack(ps_zemax, dim=-1).float()
            bins_i, bins_j, centroid_zemax = find_bins(ps_zemax, pixel_size, kernel_size)
            hist_zemax = torch.histogramdd(ps_zemax, bins=(bins_i, bins_j), density=False).hist
            hist_zemax /= hist_zemax.sum()
            # ax, fig = plt.subplots(2, 2, figsize=(10, 5))
            # fig[0,0].imshow(hist_ps.reshape(kernel_size, kernel_size), origin='lower', interpolation='nearest')
            # fig[0,0].set_title('dO')
            # fig[0,1].imshow(hist_zemax.reshape(kernel_size, kernel_size), origin='lower', interpolation='nearest')
            # fig[0,1].set_title('Zemax')
            # fig[1,0].scatter(ps[..., 1], ps[...,0], s=0.1)
            # fig[1,0].set_aspect('equal')
            # fig[1,1].scatter(ps_zemax[..., 1], ps_zemax[...,0], s=0.1)
            # fig[1,1].set_aspect('equal')
            ax, fig = plt.subplots(2, 1, figsize=(32/2, 18/2), dpi=120)
            #fig[0].scatter(ps[..., 1], ps[...,0], s=2.5)
            fig[0].scatter(ps[..., 0], ps[...,1], s=2.5/(4*4))
            fig[0].set_aspect('equal', adjustable='box')
            fig[0].set_ylabel('y [mm]')
            #fig[1].scatter(ps_zemax[..., 1], ps_zemax[...,0], s=2.5)
            fig[1].scatter(ps_zemax[..., 0], ps_zemax[...,1], s=2.5/(4*4))
            fig[1].set_aspect('equal', adjustable='box')
            fig[1].set_xlabel('x [mm]')
            fig[1].set_ylabel('y [mm]')
            
            fig[0].set_xticks([])
            fig[0].set_yticks([])
            fig[1].set_xticks([])
            fig[1].set_yticks([])
            
            fig[0].set_ylabel('')
            fig[1].set_xlabel('')
            fig[1].set_ylabel('')

            ylims0 = fig[0].get_ylim()
            y_center0 = (ylims0[0]+ylims0[1])/2
            xlims0 = fig[0].get_xlim()
            x_center0 = (xlims0[0]+xlims0[1])/2
            ylims1 = fig[0].get_ylim()
            y_center1 = (ylims1[0]+ylims1[1])/2
            xlims1 = fig[0].get_xlim()
            x_center1 = (xlims1[0]+xlims1[1])/2
            ylims = (min(ylims0[0], ylims1[0]), max(ylims0[1], ylims1[1]))
            xlims = (min(xlims0[0], xlims1[0]), max(xlims0[1], xlims1[1]))
            y_center = (ylims[0]+ylims[1])/2
            x_center = (xlims[0]+xlims[1])/2
            d_y = (ylims[1]-ylims[0])/2
            d_x = (xlims[1]-xlims[0])/2
            d_max = max(d_y, d_x)

            #if k <= 2 or k==5:
            #    y_center1 += 0.0025

            fig[0].set_ylabel('dO', fontsize = 70/2)
            fig[1].set_ylabel('Zemax', fontsize = 70/2)
            fig[0].set_ylim((y_center0-d_max*1.5, y_center0+d_max*1.5))
            fig[0].set_xlim((x_center0-d_max*1.5, x_center0+d_max*1.5))
            fig[1].set_ylim((y_center1-d_max*1.5, y_center1+d_max*1.5))
            fig[1].set_xlim((x_center1-d_max*1.5, x_center1+d_max*1.5))

            rms_ps = torch.sqrt(torch.mean((ps - centroid_ps)**2))*1000
            rms_zemax = torch.sqrt(torch.mean((ps_zemax- centroid_zemax)**2))*1000


            def add_interval(ax, xdata, ydata, caps="||"):
                """
                Add an interval to the plot for size reference.

                Args:
                    ax (matplotlib.axes.Axes): The axes to add the interval to.
                    xdata (list): The x-coordinates of the interval.
                    ydata (list): The y-coordinates of the interval.
                    caps (str, optional): The text to add at the ends of the interval. Defaults to "||".
                
                Returns:
                    tuple: The line and the annotations.
                """
                line = ax.add_line(lines.Line2D(xdata, ydata, color='black', linewidth=3))
                anno_args = {
                    'ha': 'center',
                    'va': 'center',
                    'size': 30/2,
                    'color': line.get_color()
                }
                plt.rcParams['text.usetex'] = False
                a0 = ax.annotate(caps[0], xy=(xdata[0], ydata[0]), **anno_args)
                a1 = ax.annotate(caps[1], xy=(xdata[1], ydata[1]), **anno_args)
                plt.rcParams['text.usetex'] = True
                
                a2 = ax.annotate(r'1\,px', xy=((xdata[0]+xdata[1])/2, ydata[0]), xytext=(0, 30), textcoords='offset pixels', ha='center', va='center', size=60/2)
                return (line,(a0,a1))
            
            circle_ps=patches.Circle((centroid_ps[0].item(), centroid_ps[1].item()), radius=rms_ps.item()/1000,facecolor=None,
                    edgecolor='black',linestyle='dotted',linewidth=3., fill=False)
            fig[0].add_patch(circle_ps)

            
            add_interval(fig[0], [fig[0].get_xlim()[0]+2*d_max*1.5*0.95 - pixel_size, fig[0].get_xlim()[0]+2*d_max*1.5*0.95], [fig[0].get_ylim()[0] + 2*d_max*1.5*0.05, fig[0].get_ylim()[0] + 2*d_max*1.5*0.05], "||")

            circle_zemax=patches.Circle((centroid_zemax[0].item(), centroid_zemax[1].item()), radius=rms_zemax.item()/1000,facecolor=None,
                    edgecolor='black',linestyle='dotted',linewidth=3., fill=False)
            fig[1].add_patch(circle_zemax)

            add_interval(fig[1], [fig[1].get_xlim()[0]+2*d_max*1.5*0.95 - pixel_size, fig[1].get_xlim()[0]+2*d_max*1.5*0.95], [fig[1].get_ylim()[0] + 2*d_max*1.5*0.05, fig[1].get_ylim()[0] + 2*d_max*1.5*0.05], "||")

            fig[0].text(0.2, 0.95, f'RMS = {rms_ps:.1f}' + r'\,µm', transform=fig[0].transAxes,
                fontsize=70/2, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

            fig[1].text(0.2, 0.95, f'RMS = {rms_zemax:.1f}' + r'\,µm', transform=fig[1].transAxes,
                fontsize=70/2, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
            
            inset_ax0 = fig[0].inset_axes([0.02, 0.02, 0.15, 0.15])
            inset_ax0.scatter([-params[k][0][0]*0.85], [-params[k][0][1]*0.85], color='red', s=120/4)
            inset_ax0.set_ylim((-2.7, 2.7))
            inset_ax0.set_xlim((-2.7, 2.7))
            inset_ax0.set_xticks([])
            inset_ax0.set_yticks([])

            inset_ax1 = fig[1].inset_axes([0.02, 0.02, 0.15, 0.15])
            inset_ax1.scatter([-params[k][0][0]*0.85], [-params[k][0][1]*0.85], color='red', s=120/4)
            inset_ax1.set_ylim((-2.7, 2.7))
            inset_ax1.set_xlim((-2.7, 2.7))
            inset_ax1.set_xticks([])
            inset_ax1.set_yticks([])

            #plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            #                hspace = 0, wspace = 0)
            plt.margins(0,0)

            plt.suptitle(f'PSF at x={params[k][0][0]}mm, y={params[k][0][1]}mm, w={params[k][1]:.1f}nm', fontsize = 70/2)
            if self.save_dir is not None:
                plt.savefig(self.save_dir + f'psf_posx_{params[k][0][0]}_posy_{params[k][0][1]}_w_{params[k][1]:.1f}.png', format='png', bbox_inches = 'tight', pad_inches = 0)
        plt.show()

    def fit_psf(self, nb_centric_circles, params, depth_list, angle_list, start_dist, pixel_size, kernel_size=11, show_psf=False):
        """
        Manually fit the PSF for a given set of parameters. Shows the rmse between the PSF and the Zemax PSF for each set of parameters.
        Args:
            nb_centric_circles (int): Number of circles in the hexapolar grid. Will define the number of rays.
            params (list): List of parameters to compare. Each parameter is a tuple (source_pos, wavelength, ps_zemax).
            depth_list (list): List of depths to test.
            angle_list (list): List of angles to test.
            start_dist (float): Starting distance.
            pixel_size (float): Size of the pixels in the sensor.
            kernel_size (int, optional): Size of the kernel for the histogram. Defaults to 11.
            show_psf (bool, optional): Whether to show the PSF. Defaults to False.
        
        Returns:
            torch.Tensor: The rmse between the PSF and the Zemax PSF for each set of parameters.
        """
        results = torch.zeros(len(depth_list), len(angle_list))
        for i, depth in enumerate(tqdm(depth_list)):
            self.system[-1].d_sensor = depth + start_dist
            for j, angle in enumerate(angle_list):
                for k in range(len(params)):
                    d = self.extract_hexapolar_dir(nb_centric_circles, params[k][0], angle) 
                    ps = self.trace_psf_from_point_source(angles = None, x_pos = params[k][0][0], y_pos = params[k][0][1], z_pos = 0., wavelength = params[k][1],
                            show_rays = False, d = d, ignore_invalid = False, show_res = False)
                    bins_i, bins_j, centroid_ps = find_bins(ps, pixel_size, kernel_size)
                    hist_ps = torch.histogramdd(ps, bins=(bins_i, bins_j), density=False).hist
                    hist_ps /= hist_ps.sum()

                    ps_zemax = params[k][2]
                    ps_zemax = torch.stack(ps_zemax, dim=-1).float()
                    bins_i, bins_j, centroid_zemax = find_bins(ps_zemax, pixel_size, kernel_size)
                    hist_zemax = torch.histogramdd(ps_zemax, bins=(bins_i, bins_j), density=False).hist
                    hist_zemax /= hist_zemax.sum()
                    if show_psf:
                        ax, fig = plt.subplots(2, 2, figsize=(10, 5))
                        fig[0,0].imshow(hist_ps.reshape(kernel_size, kernel_size), origin='lower', interpolation='nearest')
                        fig[0,0].set_title('dO')
                        fig[0,1].imshow(hist_zemax.reshape(kernel_size, kernel_size), origin='lower', interpolation='nearest')
                        fig[0,1].set_title('Zemax')
                        fig[1,0].scatter(ps[..., 1], ps[...,0], s=0.1)
                        fig[1,0].set_aspect('equal')
                        fig[1,1].scatter(ps_zemax[..., 1], ps_zemax[...,0], s=0.1)
                        fig[1,1].set_aspect('equal')
                        plt.savefig(f'psf_posx_{params[k][0][0]}_posy_{params[k][0][1]}_w_{params[k][1]}.png')
                        plt.show()

                    rmse = torch.sqrt(torch.mean(((hist_ps - hist_zemax))**2))
                    rms_ps = torch.sqrt(torch.mean((ps - centroid_ps)**2))
                    rms_zemax = torch.sqrt(torch.mean((ps_zemax - centroid_zemax)**2))
                    results[i, j] += rmse
                    #results[i, j] += torch.abs(rms_ps-rms_zemax)
                results[i, j] /= len(params)
        return results

    def create_train_database(self, nb_centric_circles, line_pos_list, col_pos_list, max_angle, wavelength_list, lim):
        """
        Creates a PSF training database regularly sampled.
        Args:
            nb_centric_circles (int): Number of centric circles in the hexapolar grid. Will define the number of rays.
            line_pos_list (ndarray): Array of sample's line positions.
            col_pos_list (ndarray): Array of sample's column positions.
            max_angle (float): Maximum angle towards the first lens.
            wavelength_list (list): List of wavelengths.
            lim (float): Grid's limit.
        Returns:
            None
        """

        nb_pts = line_pos_list.shape[0]
        nb_ray = 1 + 3*nb_centric_circles*(nb_centric_circles-1)
        time_start = time.time()
        for wavelength in wavelength_list:
            saved_data = np.empty((nb_pts, nb_pts, nb_ray, 2))
            for i, col_pos in enumerate(col_pos_list):
                for j, line_pos in enumerate(line_pos_list):
                    source_pos = np.array([col_pos, line_pos])
                    d = self.extract_hexapolar_dir(nb_centric_circles, source_pos, max_angle)                   

                    ps = self.trace_psf_from_point_source(angles = None, x_pos = line_pos, y_pos = col_pos, z_pos = 0., wavelength = wavelength,
                            show_rays = False, d = d, ignore_invalid = False, show_res = False)
            
                    saved_data[i, j, :ps.shape[0], :] = ps
                print(f"Line {i} done after {time.time() - time_start:.1f} seconds")
            print(f"Wavelength {wavelength} done after {time.time() - time_start:.1f} seconds")

            np.save(self.save_dir +'/psfs/' + f'psf_field_w_{round(wavelength, 2)}_lim_{lim}_pts_{nb_pts}.npy', saved_data)

    def create_test_database(self, nb_centric_circles, line_pos_list, col_pos_list, max_angle, wavelength_list, lim, index_save = True):
        """
        Creates a PSF test database randomly sampled.
        Args:
            nb_centric_circles (int): Number of centric circles in the hexapolar grid. Will define the number of rays.
            line_pos_list (ndarray): Array of sample's line positions.
            col_pos_list (ndarray): Array of sample's column positions.
            max_angle (float): Maximum angle towards the first lens.
            wavelength_list (list): List of wavelengths.
            lim (float): Grid's limit.
            index_save (bool): Flag indicating whether to save the PSFs with the index or the wavelength. Defaults to True.
        Returns:
            None
        """
        database_size = line_pos_list.shape[0]

        time_start = time.time()
        for i in range(database_size):
            line_pos = line_pos_list[i]
            col_pos = col_pos_list[i]
            wavelength = self.wavelengths[wavelength_list[i]]
            source_pos = np.array([col_pos, line_pos])
            d = self.extract_hexapolar_dir(nb_centric_circles, source_pos, max_angle) 

            ps = self.trace_psf_from_point_source(angles = None, x_pos = line_pos, y_pos = col_pos, z_pos = 0., wavelength = wavelength,
                            show_rays = False, d = d, ignore_invalid = False, show_res = False)
            
            if index_save:
                np.save(self.save_dir + f"/psfs/{lim}_database_for_comparison/" f'x_{-col_pos}_y_{-line_pos}_w_{wavelength_list[i]}.npy', ps)
            else:
                np.save(self.save_dir + f"/psfs/{lim}_database_for_comparison/" f'x_{-col_pos}_y_{-line_pos}_interp_w_{self.wavelengths[wavelength_list[i]]}.npy', ps)
            if i % 100==0:
                print(f"{i} samples generated after {time.time() - time_start:.1f} seconds")
    
    def create_test_regular_database(self, nb_centric_circles, line_pos_list, col_pos_list, max_angle, wavelength_list, lim,
                                     reduced = True, index_save = True):
        """
        Creates a PSF test database regularly sampled at the middle of the training grid's positions.
        Args:
            nb_centric_circles (int): Number of centric circles in the hexapolar grid. Will define the number of rays.
            line_pos_list (ndarray): Array of sample's line positions.
            col_pos_list (ndarray): Array of sample's column positions.
            max_angle (float): Maximum angle towards the first lens.
            wavelength_list (list): List of wavelengths.
            lim (float): Grid's limit.
            reduced (bool): Flag indicating whether to reduce the number of wavelengths to lowest, central, highest. Defaults to True.
            index_save (bool): Flag indicating whether to save the PSFs with the index or the wavelength. Defaults to True.
        Returns:
            None
        """
        if reduced:
            wavelength_list = [wavelength_list[0], wavelength_list[19], wavelength_list[-1]] # 450, 520.37, 650
        
        time_start = time.time()
        for wav, wavelength in enumerate(wavelength_list):
            if reduced:
                if wav==0:
                    w = 0
                elif wav==1:
                    w = 19
                elif wav==2:
                    w = 54
            else:
                w = wav
            for i, col_pos in enumerate(col_pos_list):
                for j, line_pos in enumerate(line_pos_list):
                    source_pos = np.array([col_pos, line_pos])
                    d = self.extract_hexapolar_dir(nb_centric_circles, source_pos, max_angle)                    

                    ps = self.trace_psf_from_point_source(angles = None, x_pos = line_pos, y_pos = col_pos, z_pos = 0., wavelength = wavelength,
                            show_rays = False, d = d, ignore_invalid = False, show_res = False)
                    if index_save:
                        np.save(self.save_dir + f"/psfs/nb_pts_{lim}_regular_database_for_comparison/" f'x_{-col_pos}_y_{-line_pos}_w_{w}.npy', ps)
                    else:
                        np.save(self.save_dir + f"/psfs/nb_pts_{lim}_regular_database_for_comparison/" f'x_{-col_pos}_y_{-line_pos}_interp_w_{wavelength_list[w]}.npy', ps)
                print(f"Line {i} done after {time.time() - time_start:.1f} seconds")
                
            print(f"Wavelength {wavelength} done after {time.time() - time_start:.1f} seconds")

    def psf_field_view(self, nb_centric_circles, depth_list, line_pos_list, col_pos_list, max_angle, wavelength_list):
        """
        Shows a field of several PSFs at several wavelengths. Used to track the visual evolution of PSFs with depth, spatial and spectral positions.
        Args:
            nb_centric_circles (int): Number of centric circles in the hexapolar grid. Will define the number of rays.
            depth_list (ndarray): Array of sample's depths.
            line_pos_list (ndarray): Array of sample's line positions.
            col_pos_list (ndarray): Array of sample's column positions.
            max_angle (float): Maximum angle towards the first lens.
            wavelength_list (list): List of wavelengths.
        Returns:
            None
        """

        nb_ray = 1 + 3*nb_centric_circles*(nb_centric_circles-1)

        base_value = self.system[-1].d_sensor

        saved_data = np.zeros((len(depth_list), len(wavelength_list), len(line_pos_list),len(col_pos_list),nb_ray,2))
        for d, depth in enumerate(depth_list):
            self.system[-1].d_sensor = base_value + depth
            for w, wavelength in enumerate(wavelength_list):
                for l, line_pos in enumerate(line_pos_list):
                    for c, col_pos in enumerate(col_pos_list):
                        source_pos = np.array([col_pos, line_pos])

                        dist = self.extract_hexapolar_dir(nb_centric_circles, source_pos, max_angle)   

                        ps = self.trace_psf_from_point_source(angles = None, x_pos = line_pos, y_pos = col_pos, z_pos = 0., wavelength = wavelength,
                            show_rays = False, d = dist, ignore_invalid = False, show_res = False)
                        saved_data[d, w, l, c, :ps.shape[0], :] = ps
            print(f"Depth {depth:.2f} done")
        
        np.save(self.save_dir +'/psfs_optim/' + f'psf_field.npy', saved_data)

        plt.close()
        for w, wavelength in enumerate(wavelength_list):
            for de, depth in enumerate(depth_list):
                plt.figure()
                for i in range(len(line_pos_list)):
                    for j in range(len(col_pos_list)):
                        ps = saved_data[de, w, i, j, :, :]
                        ps = ps[~np.any(ps == 0.0, axis=1), :].reshape(-1,2)
                        plt.scatter(ps[:, 1], ps[:, 0], s=0.1)
                plt.axis('equal')
                plt.title(f"Wavelength {wavelength:.2f}, depth {depth:.2f}")
                plt.savefig(self.save_dir + '/psfs_optim/' + f"w_{wavelength:.2f}_depth_{depth:.2f}.png", format='png')

        plt.show()
        for w, wavelength in enumerate(wavelength_list):
            for de, depth in enumerate(depth_list):
                fig, ax = plt.subplots(len(line_pos_list),len(col_pos_list))
                pixel_size = 0.005
                kernel_size = 11
                for i in range(len(line_pos_list)):
                    for j in range(len(col_pos_list)):
                        ps = saved_data[de, w, i, j, :, :]
                        ps = ps[~np.any(ps == 0.0, axis=1), :].reshape(-1,2)
                        bins_i, bins_j = find_bins(ps, pixel_size, kernel_size)
                        hist = torch.histogramdd(torch.from_numpy(ps).float().flip(1), bins=(bins_j, bins_i), density=False).hist
                        hist = hist.T.numpy()
                        ax[i,j].imshow(hist.reshape(kernel_size, kernel_size), origin='lower', interpolation='nearest')

                plt.axis('equal')
                plt.title(f"Wavelength {wavelength:.2f}, depth {depth:.2f}")
                plt.savefig(self.save_dir + '/psfs_optim/' + f"heatmap_w_{wavelength:.2f}_depth_{depth:.2f}.png", format='png')

                plt.close()
    
    def minimize_psf(self, nb_centric_circles, depth_list, line_pos_list, col_pos_list, max_angle, wavelength_list):
        """
        Plot a curve to find the minimum sized PSFs w.r.t the depth. Used to find the optimal depth for the sensor. 
        This is done by computing the PSFs for a range of depths at several regular positions and then computing the average size of the PSFs.
        Args:
            nb_centric_circles (int): Number of centric circles in the hexapolar grid. Will define the number of rays.
            depth_list (ndarray): Array of sample's depths.
            line_pos_list (ndarray): Array of sample's line positions.
            col_pos_list (ndarray): Array of sample's column positions.
            max_angle (float): Maximum angle towards the first lens.
            wavelength_list (list): List of wavelengths.
        Returns:
            None
        """

        base_value = self.system[-1].d_sensor

        saved_data = np.zeros((len(depth_list), len(line_pos_list)*len(col_pos_list)*len(wavelength_list)))
        for d, depth in enumerate(depth_list):
            ind = 0
            for wavelength in wavelength_list:
                self.system[-1].d_sensor = base_value + depth
                for l, line_pos in enumerate(line_pos_list):
                    for c, col_pos in enumerate(col_pos_list):
                        source_pos = np.array([col_pos, line_pos])
                        dist = self.extract_hexapolar_dir(nb_centric_circles, source_pos, max_angle)   

                        ps = self.trace_psf_from_point_source(angles = None, x_pos = line_pos, y_pos = col_pos, z_pos = 0., wavelength = wavelength,
                            show_rays = False, d = dist, ignore_invalid = False, show_res = False)

                        saved_data[d, ind] = np.mean(np.var(ps.numpy(), axis=0))
                        ind+=1
            print(f"Depth {depth:.2f} done")
        plt.plot(depth_list, np.mean(saved_data, axis=-1))
        plt.show()

    def minimize_psf_random(self, nb_centric_circles, depth_list, max_angle, wavelength_list, lim, nb_pts, n_run):
        """
        Plot a curve to find the minimum sized PSFs w.r.t the depth. Used to find the optimal depth for the sensor. 
        This is done by computing the PSFs for a range of depths at several random positions and then computing the average size of the PSFs.
        Args:
            nb_centric_circles (int): Number of centric circles in the hexapolar grid. Will define the number of rays.
            depth_list (ndarray): Array of sample's depths.
            line_pos_list (ndarray): Array of sample's line positions.
            col_pos_list (ndarray): Array of sample's column positions.
            max_angle (float): Maximum angle towards the first lens.
            wavelength_list (list): List of wavelengths.
            lim (float): Grid's limit.
            nb_pts (int): Number of random points to generate.
            n_run (int): Number of runs to average the results. 1 can be enough if sampled high enough.
        Returns:
            None
        """
        base_value = self.system[-1].d_sensor

        for n in range(n_run):
            line_pos_list = np.random.uniform(-lim, lim, size=nb_pts)
            col_pos_list = np.random.uniform(-lim, lim, size=nb_pts)
            
            saved_data = np.zeros((len(depth_list), len(line_pos_list)*len(wavelength_list), 2))

            for d, depth in enumerate(depth_list):
                ind = 0
                for wavelength in wavelength_list:
                    self.system[-1].d_sensor = base_value + depth
                    for l in range(len(line_pos_list)):
                        line_pos = line_pos_list[l]
                        col_pos = col_pos_list[l]
                        source_pos = np.array([col_pos, line_pos])
                        dist = self.extract_hexapolar_dir(nb_centric_circles, source_pos, max_angle)   

                        ps = self.trace_psf_from_point_source(angles = None, x_pos = line_pos, y_pos = col_pos, z_pos = 0., wavelength = wavelength,
                            show_rays = False, d = dist, ignore_invalid = False, show_res = False)

                        saved_data[d, ind, :] = np.var(ps.numpy(), axis=0)
                        ind+=1
                print(f"Depth {depth:.3f} done")
            np.save(self.save_dir +'/psfs_optim_random/' + f'vars_{n}.npy', saved_data)
            plt.figure()
            plt.plot(depth_list, np.mean(saved_data[:,:,0], axis=-1))
            plt.title("Variance w.r.t. z for y dimension")
            #plt.savefig(save_dir + '/psfs_optim_random/' + f"vars_y_graph_{n}.png", format='png')
            plt.figure()
            
            plt.plot(depth_list, np.mean(saved_data[:,:,1], axis=-1))
            plt.title("Variance w.r.t. z for x dimension")
            #plt.savefig(save_dir + '/psfs_optim_random/' + f"vars_x_graph_{n}.png", format='png')
            #plt.close()
            plt.show()
            print(f"Run {n+1} done")
    
def draw_circle(N, z, max_angle, center_x=0, center_y=0):
    """
    Generate random points on a circle.
    Args:
        N (int): Number of points to generate.
        z (float): Distance from the center of the circle to the plane.
        max_angle (float): Maximum angle in degrees from the center of the circle to the generated points.
        center_x (float, optional): x-coordinate of the center of the circle. Default is 0.
        center_y (float, optional): y-coordinate of the center of the circle. Default is 0.
    Returns:
        x (ndarray): Array of x-coordinates of the generated points.
        y (ndarray): Array of y-coordinates of the generated points.
    """
    

    theta = np.random.uniform(0, 2*np.pi, N)
    radius = np.random.uniform(0, (z*np.tan(max_angle*np.pi/180))**2, N)**0.5

    x = radius*np.cos(theta) + center_x
    y = radius*np.sin(theta) + center_y

    return x, y

def draw_circle_hexapolar(N, z, max_angle, center_x=0, center_y=0):
    """
    Generate coordinates for a hexapolar pattern.
    Args:
        N (int): Number of points to generate.
        z (float): Distance from the center of the circle to the plane.
        max_angle (float): Maximum angle in degrees from the center of the circle to the generated points.
        center_x (float, optional): X-coordinate of the center of the pattern. Default is 0.
        center_y (float, optional): Y-coordinate of the center of the pattern. Default is 0.
    Returns:
        x (ndarray): Array of x-coordinates of the generated points..
        y (ndarray): Array of y-coordinates of the generated points..
    """
    max_angle = torch.tensor([max_angle]).float() if not torch.is_tensor(max_angle) else max_angle
    radius = torch.linspace(0, z*torch.tan(max_angle*np.pi/180).item(), N)
    nb_rays = 1 + 3*N*(N-1)
    x = torch.zeros(nb_rays)
    y = torch.zeros(nb_rays)
    ind = 1
    for i, r in enumerate(radius):
        if i ==0:
            continue
        theta = torch.linspace(0, 2*np.pi, 6*i+1)[:-1] # 6*i points, remove endpoint
        x[ind:ind+6*i] = r*torch.cos(theta)
        y[ind:ind+6*i] = r*torch.sin(theta)
        ind += 6*i
    x += center_x
    y += center_y

    return x, y

def compute_centroid(points):
    """
    Compute the centroid of a set of points.
    Args:
        points: A tensor or numpy array representing the points.
    Returns:
        centroid: A tensor representing the centroid of the points.
    """

    points = points if type(points) is torch.Tensor else torch.from_numpy(points)
    return torch.mean(points, dim=0)

def find_bins(points, size_bin, nb_bins=11, same_grid=False, centroid=None, absolute_grid = False,):
    """
    Compute the bins for a given set of points. The bins can be computed independently or on a given grid

    Args:
        points (torch.Tensor): The input points.
        size_bin (float): The size of each bin.
        nb_bins (int, optional): The number of bins. Default is 11.
        same_grid (bool, optional): Whether to use the same grid for all points. Default is False.
        centroid (torch.Tensor, optional): The centroid of the points. Default is None.
        absolute_grid (bool, optional): Whether to use an absolute grid, defined by the size and number of pixels. Default is False.

    Returns:
        bins_i (torch.Tensor): The bins along the x-axis.
        bins_j (torch.Tensor): The bins along the y-axis.
        centroid (torch.Tensor): The centroid of the points.
    """

    if centroid is None:
        centroid = compute_centroid(points)
    if same_grid:
        in_grid_pos = torch.floor(centroid/size_bin) + 0.5
        centroid = in_grid_pos * size_bin
    
    if absolute_grid:
        bins_i = torch.linspace(-size_bin*nb_bins[0]/2, size_bin*nb_bins[0]/2, nb_bins[0]+1)
        bins_j = torch.linspace(-size_bin*nb_bins[1]/2, size_bin*nb_bins[1]/2, nb_bins[1]+1)
    else:
        bins_i = torch.zeros((nb_bins+1))
        bins_j = torch.zeros((nb_bins+1))
    
        for i in range(nb_bins+1):
            bins_i[i] = centroid[0] - size_bin/2 - (nb_bins//2)*size_bin + i*size_bin
            bins_j[i] = centroid[1] - size_bin/2 - (nb_bins//2)*size_bin + i*size_bin
    
    

    return bins_i, bins_j, centroid

def extract_positions(file_path):
    """
    Extract the x and y positions of a PSF from Zemax from a file.

    Args:
        file_path (str): The path to the file.
    
    Returns:
        tuple: The x and y positions.
    """
    if '.h5' in file_path:
        # Open the H5 file
        with h5py.File(file_path, 'r') as h5f:
            # Read X and Y positions
            x_positions = h5f['X Position'][:]
            y_positions = h5f['Y Position'][:]
    elif '.txt' in file_path:
        data = []
    
        with open(file_path, 'r', encoding='utf-16') as file:
            lines = file.readlines()

        for line in lines:
            if '(' in line:
                parts = line.split()
                try:
                    if len(parts) >= 5:
                        x_pos = float(parts[-2].replace(',', '.'))
                        y_pos = float(parts[-1].replace(',', '.'))
                        data.append([x_pos, y_pos])
                except ValueError:
                    continue
        data = np.array(data)
        x_positions = data[:, 0]
        y_positions = data[:, 1]

    return torch.from_numpy(x_positions), torch.from_numpy(y_positions)

import re

def parse_grid_distortion_data(file_path):
    """
    Parse the spot diagram of a grid from a system in Zemax from a file.

    Args:
        file_path (str): The path to the file.
    
    Returns:
        tuple: The data, the x-field (input), y-field, x-real (output), y-real
    """
    with open(file_path, 'r', encoding='utf-16') as file:
        lines = file.readlines()

    data = {
        "file": "",
        "title": "",
        "date": "",
        "units": "",
        "wavelength": "",
        "reference_coordinates": {},
        "grid_data": [],
        "maximum_distortion": "",
        "predicted_coordinate_abcd_matrix": {}
    }
    # Parse grid data
    ref_coords = re.search(r'Reference Coordinates:\s*Xref = (.*), Yref = (.*)', lines[9])
    x_ref, y_ref = float(ref_coords.group(1).strip().replace(',', '.')), float(ref_coords.group(2).strip().replace(',', '.'))
    grid_data_start = 12
    grid_data_end = len(lines) - 10  # Assuming footer starts 10 lines from the end
    xy_data = []
    for line in lines[grid_data_start:grid_data_end]:
        if line.strip():
            parts = re.split(r'\s+', line.strip())
            dic = {
                "i": int(parts[0]),
                "j": int(parts[1]),
                "X-Field": float(parts[2].replace(',', '.')),
                "Y-Field": float(parts[3].replace(',', '.')),
                "R-Field": float(parts[4].replace(',', '.')),
                "Predicted X": float(parts[5].replace(',', '.')),
                "Predicted Y": float(parts[6].replace(',', '.')),
                "Real X": float(parts[7].replace(',', '.')),
                "Real Y": float(parts[8].replace(',', '.')),
                "Distortion": float(parts[9].replace(',', '.').replace('%', ''))
            }
            data["grid_data"].append(dic)
            xy_data.append([dic['X-Field'], dic['Y-Field'], dic['Predicted X'], dic['Predicted Y'], dic['Real X'], dic['Real Y']])
    xy_data = np.array(xy_data)
    x_field = xy_data[:, 0]
    y_field = xy_data[:, 1]
    x_pred = xy_data[:, 2]
    y_pred = xy_data[:, 3]
    x_real = xy_data[:, 4] + x_ref
    y_real = xy_data[:, 5] + y_ref

    return data, torch.from_numpy(x_field), torch.from_numpy(y_field), torch.from_numpy(x_real), torch.from_numpy(y_real)