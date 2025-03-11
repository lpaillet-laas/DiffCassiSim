from CASSI_class import HSSystem
import diffoptics as do
import matplotlib.pyplot as plt
import numpy as np
from diffoptics.basics import InterpolationMode
import yaml
import torch


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

            self.mask = mask
            self.mask_pixelsize = mask_pixelsize if mask_pixelsize is not None else 0.
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

        system_dict['mask_pixelsize'] = list(self.mask_pixelsize)
        system_dict['mask_d'] = self.mask_d
        system_dict['mask_theta'] = list(self.mask_theta)
        system_dict['mask_origin'] = list(self.mask_origin)
        system_dict['mask_shift'] = list(self.mask_shift)
        system_dict['mask_index'] = self.mask_index

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

        


    def apply_mask(self, z0, rays, valid, R_mask = None, t_mask = None, shading_type = "single"):
        """
        Applies the mask to the rays.

        Args:
            z0 (float): z Position of the mask.
            rays (do.Rays): Rays to apply the mask to.
            valid (torch.Tensor): Valid rays.
            R_mask (np.ndarray, optional): Rotation matrix of the mask. Defaults to None.
            t_mask (np.ndarray, optional): Translation vector of the mask. Defaults to None.
            shading_type (str, optional): Type of shading to apply. Defaults to "single". Available options are "all", "batch" and "single".
        
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
        screen = do.Screen(
            do.Transformation(R_mask, t_mask),
            texturesize * self.mask_pixelsize, texture, device=self.device
        )

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

        return valid_rays
    
    def render_single_back(self, wavelength, screen, numerical_aperture = 0.05, z0_mask = None, save_pos = False):
        """
        Renders back propagation of single ray through a series of lenses onto a screen.

        Args:
            wavelength (float): The wavelength of the light.
            screen (object): The screen object onto which the light is projected.
            numerical_aperture (float, optional): The numerical aperture of the system. Defaults to 0.05.
            z0_mask (float, optional): z Position of the mask. Defaults to 0..
            save_pos (bool, optional): Whether to save the positions of the resulting rays (True), or return a rendered image (False). Defaults to False.
        Returns:
            tuple: A tuple containing the intensity values (I) and the mask indicating valid pixels on the screen.
        """
        # Sample rays from the sensor
        valid, ray_mid = self.system[0].sample_ray_sensor(wavelength.item(), numerical_aperture = numerical_aperture)

        if z0_mask is None:
            z0_mask = self.mask_origin[2] + self.mask_shift[2] + self.mask_d

        if self.mask_index == 0:
            valid = self.apply_mask(z0_mask, ray_mid, valid, self.mask_R, self.mask_t, shading_type = "single")   

        #print("Ray 1: ", ray_mid)
        #print("Nb valid 1: ", torch.sum(valid))
        # Trace rays through each lens in the system
        if len(self.system) > 1:
            for ind, lens in enumerate(self.system[1:-1]):
                ray_mid = lens.to_object.transform_ray(ray_mid)
                valid_1, ray_mid = lens._trace(ray_mid)

                ray_mid = lens.to_world.transform_ray(ray_mid)

                if self.mask_index == ind + 1:
                    valid_1 = self.apply_mask(z0_mask, ray_mid, valid_1, self.mask_R, self.mask_t, shading_type = "single")

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
                valid_last = self.apply_mask(z0_mask, ray_last, valid_last, self.mask_R, self.mask_t, shading_type = "single")
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
    
    def expand_system_symmetrically(self, symmetry_ax = "horizontal", ax_position = 0., ax_theta = np.zeros(3)):
        """
        Expands the system symmetrically along a given axis. The expansion will be set at the end of the current system.

        Args:
            symmetry_ax (str, optional): The axis of symmetry. Can be either horizontal, vertical or any. If any, ax_position needs to be a 3D vector. Defaults to "horizontal".
            ax_position (float, optional): The position of the axis of symmetry. It is a z position if the symmetry is "horizontal" and a
                                        x position if the symmetry is "vertical". Defaults to 0.
        """
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
                        start_distance = surface.d.item() if isinstance(surface.d, torch.Tensor) else surface.d
                    else:
                        surface_dict['d'] = surface.d.item() - start_distance if isinstance(surface.d, torch.Tensor) else surface.d - start_distance
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
                        start_distance = surface.d.item() if isinstance(surface.d, torch.Tensor) else surface.d
                    else:
                        surface_dict['d'] = surface.d.item() - start_distance if isinstance(surface.d, torch.Tensor) else surface.d - start_distance
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

            def mirror_point(point, ax_position, ax_theta):
                point = np.array(point)
                ax_position = np.array(ax_position)
                ax_theta = np.array(ax_theta)
                transfo_ax = do.Lensgroup(origin = ax_position, theta_x = ax_theta[0], theta_y = ax_theta[1], theta_z = ax_theta[2], device = self.device)._compute_transformation()
                e1 = transfo_ax.transform_point(np.array([1, 0, 0]))
                e2 = transfo_ax.transform_point(np.array([0, 1, 0]))

                normal = np.cross(e1 - ax_position, e2 - ax_position)
                normal = normal / np.linalg.norm(normal)

                mirrored_point = point - 2 * np.dot(point - ax_position, normal) * normal

                return mirrored_point
            
            def mirror_direction(direction, ax_theta):
                direction = np.array(direction)
                ax_theta = np.array(ax_theta)
                transfo_ax = do.Lensgroup(theta_x = ax_theta[0], theta_y = ax_theta[1], theta_z = ax_theta[2], device = self.device)._compute_transformation()
                e1 = transfo_ax.transform_point(np.array([1, 0, 0]))
                e2 = transfo_ax.transform_point(np.array([0, 1, 0]))

                normal = np.cross(e1, e2)
                normal = normal / np.linalg.norm(normal)

                mirrored_direction = direction - 2 * np.dot(direction, normal) * normal

                return mirrored_direction
                
            basic_dir = np.array([0., 0., 1])
            mirrored_basic_dir = mirror_direction(basic_dir, ax_theta)

            mirrored_basic_rot = np.zeros(3)
            mirrored_basic_rot[0] = np.arctan2(mirrored_basic_dir[2], mirrored_basic_dir[1])
            mirrored_basic_rot[1] = np.arctan2(mirrored_basic_dir[2], mirrored_basic_dir[0])
            mirrored_basic_rot[2] = np.arctan2(mirrored_basic_dir[1], mirrored_basic_dir[0])

            basic_rot = np.zeros(3)
            basic_rot[0] = np.arctan2(basic_dir[2], basic_dir[1])
            basic_rot[1] = np.arctan2(basic_dir[2], basic_dir[0])
            basic_rot[2] = np.arctan2(basic_dir[1], basic_dir[0])

            coef_rot = mirrored_basic_rot / basic_rot
            print(basic_dir, mirrored_basic_dir)
            print(mirrored_basic_rot, basic_rot)

            # Reverse the order of the lens groups as it is a symmetric expansion
            for lens_id in range(len(self.system)-1, -1, -1):
                lens = self.system[lens_id]

                basic_dir = np.array([0., 0., 1.])

                lens_dir = lens.to_world.transform_vector(basic_dir)
                mirrored_lens_dir = mirror_direction(lens_dir, ax_theta)

                mirrored_basic_dir = mirror_direction(basic_dir, ax_theta)

                new_lens_surfaces = lens.surfaces.copy()

                for i in range(len(lens.surfaces)):
                    #new_lens_surfaces[i].d = np.sign(mirrored_basic_dir[2]) * new_lens_surfaces[i].d #- new_lens_surfaces[i].d
                    new_lens_surfaces[i].d = - np.sign(np.dot(lens_dir, mirrored_lens_dir)) * new_lens_surfaces[i].d
                    #new_lens_surfaces[i].d = new_lens_surfaces[i].d
                    if np.dot(lens_dir, mirrored_lens_dir) > 0:
                        new_lens_surfaces[i].reverse()
                new_lens_surfaces.reverse()

                new_surfaces = []
                for ind_s, surface in enumerate(new_lens_surfaces):
                    surface_dict = {}
                    surface_dict['type'] = get_surface_type(surface)
                    if ind_s == 0:
                        surface_dict['d'] = surface.d.item() if isinstance(surface.d, torch.Tensor) else surface.d
                        start_distance = surface.d.item() if isinstance(surface.d, torch.Tensor) else surface.d
                    else:
                        surface_dict['d'] = surface.d.item() - start_distance if isinstance(surface.d, torch.Tensor) else surface.d - start_distance
                    surface_dict['R'] = surface.r.item() if isinstance(surface.r, torch.Tensor) else surface.r
                    surface_dict['params'] = get_surface_params(surface)

                    if (np.dot(lens_dir, mirrored_lens_dir) < 0) and (surface_dict['type'] == "XYPolynomial"):
                        if len(surface_dict['params']['ai']) == 3:
                            surface_dict['params']['ai'][2] = - surface_dict['params']['ai'][2]

                    new_surfaces.append(surface_dict)

                new_system_surfaces.append(new_surfaces)
                new_mats = self.systems_materials[lens_id].copy()
                new_mats.reverse()
                new_system_materials.append(new_mats)

                new_system_d_sensor.append(np.sign(np.dot(lens_dir, mirrored_lens_dir)) * self.list_d_sensor[lens_id]) #TODO May need a change, as the values between old and new system should be different
                new_system_r_last.append(self.list_r_last[lens_id])
                new_system_film_size.append(self.list_film_size[lens_id])
                new_system_pixel_size.append(self.list_pixel_size[lens_id])

                # new_system_theta_x.append(-np.sign(mirrored_basic_dir[0])*self.list_theta_x[lens_id])
                # new_system_theta_y.append(-np.sign(mirrored_basic_dir[1])*self.list_theta_y[lens_id])
                # new_system_theta_z.append(-np.sign(mirrored_basic_dir[2])*self.list_theta_z[lens_id]) # Because the z axis w.r.t. the origin is reversed

                basic_dir = np.array([0., 0., 1.])

                lens_dir = lens.to_world.transform_vector(basic_dir)
                mirrored_lens_dir = mirror_direction(lens_dir, ax_theta)

                mirrored_basic_dir = mirror_direction(basic_dir, ax_theta)

                mirrored_basic_dir_12 = mirrored_basic_dir[[1,2]] / (np.linalg.norm(mirrored_basic_dir[[1,2]])) if np.linalg.norm(mirrored_basic_dir[[1,2]]) > 1e-10 else np.array([0., 0.])
                mirrored_basic_dir_01 = mirrored_basic_dir[[0,1]] / (np.linalg.norm(mirrored_basic_dir[[0,1]])) if np.linalg.norm(mirrored_basic_dir[[0,1]]) > 1e-10 else np.array([0., 0.])
                mirrored_basic_dir_02 = mirrored_basic_dir[[0,2]] / (np.linalg.norm(mirrored_basic_dir[[0,2]])) if np.linalg.norm(mirrored_basic_dir[[0,2]]) > 1e-10 else np.array([0., 0.])

                mirrored_lens_dir_12 = mirrored_lens_dir[[1,2]] / (np.linalg.norm(mirrored_lens_dir[[1,2]])) if np.linalg.norm(mirrored_lens_dir[[1,2]]) > 1e-10 else np.array([0., 0.])
                mirrored_lens_dir_01 = mirrored_lens_dir[[0,1]] / (np.linalg.norm(mirrored_lens_dir[[0,1]])) if np.linalg.norm(mirrored_lens_dir[[0,1]]) > 1e-10 else np.array([0., 0.])
                mirrored_lens_dir_02 = mirrored_lens_dir[[0,2]] / (np.linalg.norm(mirrored_lens_dir[[0,2]])) if np.linalg.norm(mirrored_lens_dir[[0,2]]) > 1e-10 else np.array([0., 0.])

                dot_12 = np.dot(mirrored_basic_dir_12, mirrored_lens_dir_12) if np.linalg.norm(mirrored_basic_dir_12) > 1e-10 and np.linalg.norm(mirrored_lens_dir_12) > 1e-10 else 1.
                dot_02 = np.dot(mirrored_basic_dir_02, mirrored_lens_dir_02) if np.linalg.norm(mirrored_basic_dir_02) > 1e-10 and np.linalg.norm(mirrored_lens_dir_02) > 1e-10 else 1.
                dot_01 = np.dot(mirrored_basic_dir_01, mirrored_lens_dir_01) if np.linalg.norm(mirrored_basic_dir_01) > 1e-10 and np.linalg.norm(mirrored_lens_dir_01) > 1e-10 else 1.
                
                print("")
                print("Directions: ", mirrored_basic_dir, mirrored_lens_dir)
                print("Dot products: ", dot_12, dot_02, dot_01)
                print("Angles: ", np.rad2deg(np.arccos(dot_12)), np.rad2deg(np.arccos(dot_02)), np.rad2deg(np.arccos(dot_01)))
                print("Previous angles: ", self.list_theta_x[lens_id], self.list_theta_y[lens_id], self.list_theta_z[lens_id])

                # dot_12 = np.dot(mirrored_lens_dir_12, np.array([0., 1.])) if np.linalg.norm(mirrored_lens_dir_12) > 1e-10 else 1.
                # dot_02 = np.dot(mirrored_lens_dir_02, np.array([0. , 1.])) if np.linalg.norm(mirrored_lens_dir_02) > 1e-10 else 1.
                # dot_01 = np.dot(mirrored_lens_dir_01, np.array([1. , 0.])) if np.linalg.norm(mirrored_lens_dir_01) > 1e-10 else 1.
                # dot_01 = 1.

                print("Angles v2: ", np.rad2deg(np.arccos(dot_12)), np.rad2deg(np.arccos(dot_02)), np.rad2deg(np.arccos(dot_01)))

                dot_basic_12 = np.dot(basic_dir[[1,2]], mirrored_basic_dir_12) if np.linalg.norm(basic_dir[[1,2]]) > 1e-10 else 1.
                dot_basic_02 = np.dot(basic_dir[[0,2]], mirrored_basic_dir_02) if np.linalg.norm(basic_dir[[0,2]]) > 1e-10 else 1.
                dot_basic_01 = np.dot(basic_dir[[0,1]], mirrored_basic_dir_01) if np.linalg.norm(basic_dir[[0,1]]) > 1e-10 else 1.

                print("Angles v3: ", np.rad2deg(np.arccos(dot_basic_12)), np.rad2deg(np.arccos(dot_basic_02)), np.rad2deg(np.arccos(dot_basic_01)))

                # basic_dir = np.array([1., 0., 0.])
                # mirrored_basic_dir = mirror_direction(basic_dir, ax_theta)

                # dot_basic_12 = np.dot(basic_dir[[1,2]], mirrored_basic_dir[[1,2]]) if np.linalg.norm(basic_dir[[1,2]]) > 1e-10 else 1.
                # dot_basic_02 = np.dot(basic_dir[[0,2]], mirrored_basic_dir[[1,2]]) if np.linalg.norm(basic_dir[[0,2]]) > 1e-10 else 1.
                # dot_basic_01 = np.dot(basic_dir[[0,1]], mirrored_basic_dir[[1,2]]) if np.linalg.norm(basic_dir[[0,1]]) > 1e-10 else 1.

                # print("Angles v4: ", np.rad2deg(np.arccos(dot_basic_12)), np.rad2deg(np.arccos(dot_basic_02)), np.rad2deg(np.arccos(dot_basic_01)))

                theta_x_ = 0. #np.sign(mirrored_basic_dir[2]) * np.rad2deg(np.arccos(dot_12)) + np.rad2deg(np.arccos(dot_basic_12))
                theta_y_ = - np.sign(np.dot(lens_dir, mirrored_lens_dir)) * np.rad2deg(np.arccos(dot_02)) + np.rad2deg(np.arccos(dot_basic_02))
                #TODO sign problem on theta_y_. Origins and shifts are correct but there are problems on signs of d and of rotations. Need to be checked
                theta_z_ = np.rad2deg(np.arccos(dot_01))

                print("New angles: ", theta_x_, theta_y_, theta_z_)

                new_system_theta_x.append(theta_x_)
                new_system_theta_y.append(theta_y_)
                new_system_theta_z.append(theta_z_) # Because the z axis w.r.t. the origin is reversed

                # new_system_theta_x.append(np.rad2deg(np.arctan2(mirrored_lens_dir[2], mirrored_lens_dir[1])))
                # new_system_theta_y.append(np.rad2deg(np.arctan2(mirrored_lens_dir[2], mirrored_lens_dir[0])))
                # new_system_theta_z.append(np.rad2deg(np.arctan2(mirrored_lens_dir[1], mirrored_lens_dir[0]))) # Because the z axis w.r.t. the origin is reversed

                # new_system_theta_x.append(coef_rot[0]*self.list_theta_x[lens_id])
                # new_system_theta_y.append(coef_rot[1]*self.list_theta_y[lens_id])
                # new_system_theta_z.append(coef_rot[2]*self.list_theta_z[lens_id]) # Because the z axis w.r.t. the origin is reversed

                new_origin = self.list_origin[lens_id].copy() if self.list_origin[lens_id] is not None else [0., 0., 0.]
                new_origin = mirror_point(new_origin, ax_position, ax_theta)
                new_system_origin.append(new_origin)
                
                new_shift = self.list_shift[lens_id].copy() if self.list_shift[lens_id] is not None else [0., 0., 0.]
                
                new_shift = mirrored_basic_dir * new_shift
                new_system_shift.append(new_shift)

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
        start_distance = self.system[0].d_sensor*torch.cos(self.system[0].theta_y*np.pi/180) + self.system[0].origin[-1] + self.system[0].shift[-1]
        self.prepare_mts_mask(start_distance=start_distance)
        return super().save_pos_render(texture, nb_rays, wavelengths, z0, offsets, numerical_aperture)
    
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
        start_distance = self.system[0].d_sensor*torch.cos(self.system[0].theta_y*np.pi/180) + self.system[0].origin[-1] + self.system[0].shift[-1]
        self.prepare_mts_mask(start_distance=start_distance)
        return super().propagate(texture, nb_rays, wavelengths, z0, offsets, numerical_aperture, save, plot)


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
        return {'f': surface.f, 'is_square': surface.is_square}
    elif isinstance(surface, do.ThinLenslet):
        return {'f': surface.f, 'r0': surface.r0, 'is_square': surface.is_square}
    elif isinstance(surface, do.FocusThinLens):
        return {'f': surface.f, 'is_square': surface.is_square}
    elif isinstance(surface, do.Mirror):
        return {'is_square': surface.is_square}
    else:
        raise ValueError("Surface type not recognized")