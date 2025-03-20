from .basics import *


class Endpoint(PrettyPrinter):
    """
    Abstract class for objects.
    """
    def __init__(self, transformation, device=torch.device('cpu')):
        self.to_world  = transformation
        self.to_object = transformation.inverse()
        self.to(device)
        self.device = device

    def intersect(self, ray):
        raise NotImplementedError()

    def sample_ray(self, position_sample=None):
        raise NotImplementedError()

    def draw_points(self, ax, options, seq=range(3)):
        raise NotImplementedError()

    def update_Rt(self, R, t):
        self.to_world = Transformation(R, t)
        self.to_object = self.to_world.inverse()
        self.to(self.device)


class Screen(Endpoint):
    """
    A screen obejct, useful for image rendering.
    
    Local frame centers at [-w, w]/2 x [-h, h]/2.
    """
    def __init__(self, transformation, size, texture, device=torch.device('cpu')):
        self.size = torch.tensor(np.float32(size), device=device)  # screen dimension [mm]
        self.halfsize  = self.size/2                # screen half-dimension [mm]
        self.texture_shift = torch.zeros(2)         # screen image shift [mm]
        self.device = device
        if len(texture.shape) == 2 or len(texture.shape) == 3:
            self.update_texture(texture)
        elif len(texture.shape) == 4:
            self.update_texture_batch(texture)
        Endpoint.__init__(self, transformation, device)
        self.to(device)

    def update_texture(self, texture: torch.Tensor):
        self.texture = texture # screen image [h, w]
        self.texturesize = torch.Tensor(np.array(texture.shape[0:2])).long().to(self.device) # screen image dimension [pixel]

    def update_texture_batch(self, texture: torch.Tensor):
        self.texture = texture # screen image [batch_size, h, w]
        self.texturesize = torch.Tensor(np.array(texture.shape[1:3])).long().to(self.device) # screen image dimension [pixel]

    def update_texture_all(self, texture: torch.Tensor):
        self.texture = texture # screen image [batch_size, nC, h, w]
        self.texturesize = torch.Tensor(np.array(texture.shape[2:4])).long().to(self.device) # screen image dimension [pixel]
        
    def intersect(self, ray):
        ray_in = self.to_object.transform_ray(ray)
        t = - ray_in.o[..., 2] / (1e-10 + ray_in.d[..., 2]) # (TODO: potential NaN grad)
        local = ray_in(t)

        # Is intersection within ray segment and rectangle?
        valid = (
            (t >= ray_in.mint) &
            (t <= ray_in.maxt) &
            (torch.abs(local[..., 0] - self.texture_shift[0]) <= self.halfsize[0]) &
            (torch.abs(local[..., 1] - self.texture_shift[1]) <= self.halfsize[1])
        )
        
        # UV coordinate
        uv = (local[..., 0:2] + self.halfsize - self.texture_shift) / self.size

        # Force uv to be valid in [0,1]^2 (just a sanity check: uv should be in [0,1]^2)
        uv = torch.clamp(uv, min=0.0, max=1.0)

        return local, uv, valid

    def shading(self, uv, valid, bmode=BoundaryMode.replicate, lmode=InterpolationMode.linear):
        # p = uv * (self.texturesize[None, None, ...]-1)
        p = uv * (self.texturesize-1)
        p_floor = torch.floor(p).long()

        def tex(x, y):
            """
            Texture indexing function, handle various boundary conditions.
            """
            if bmode is BoundaryMode.zero:
                raise NotImplementedError()
            elif bmode is BoundaryMode.replicate:
                x = torch.clamp(x, min=0, max=self.texturesize[0].item()-1)
                y = torch.clamp(y, min=0, max=self.texturesize[1].item()-1)
            elif bmode is BoundaryMode.symmetric:
                raise NotImplementedError()
            elif bmode is BoundaryMode.periodic:
                raise NotImplementedError()
            img = self.texture[x.flatten(), y.flatten()]
            return img.reshape(x.shape)

        # Texture fetching, requires interpolation to compute fractional pixel values.
        if lmode is InterpolationMode.nearest:
            val = tex(p_floor[...,0], p_floor[...,1])
        elif lmode is InterpolationMode.linear:
            x0, y0 = p_floor[...,0], p_floor[...,1]
            s00 = tex(  x0,   y0)
            s01 = tex(  x0, 1+y0)
            s10 = tex(1+x0,   y0)
            s11 = tex(1+x0, 1+y0)
            w1 = p - p_floor
            w0 = 1. - w1
            val = (
                w0[...,0] * (w0[...,1] * s00 + w1[...,1] * s01) + 
                w1[...,0] * (w0[...,1] * s10 + w1[...,1] * s11)
            )
        
        # val = val * valid
        # val[torch.isnan(val)] = 0.0

        # TODO: should be added;
        # but might cause RuntimeError: Function 'MulBackward0' returned nan values in its 0th output.
        val[~valid] = 0.0
        return val
    
    def shading_batch(self, uv, valid, bmode=BoundaryMode.replicate, lmode=InterpolationMode.linear):
        # p = uv * (self.texturesize[None, None, ...]-1)
        p = uv * (self.texturesize-1) # [N, 2]
        p_floor = torch.floor(p).long()

        def tex_batch(x, y):
            """
            Texture indexing function, handle various boundary conditions.
            """
            if bmode is BoundaryMode.zero:
                raise NotImplementedError()
            elif bmode is BoundaryMode.replicate:
                x = torch.clamp(x, min=0, max=self.texturesize[0].item()-1)
                y = torch.clamp(y, min=0, max=self.texturesize[1].item()-1)
            elif bmode is BoundaryMode.symmetric:
                raise NotImplementedError()
            elif bmode is BoundaryMode.periodic:
                raise NotImplementedError()
            img = self.texture[:, x.flatten(), y.flatten()] # [batchsize, N]
            return img.reshape((self.texture.shape[0], x.shape[0])) # [batchsize, N]

        # Texture fetching, requires interpolation to compute fractional pixel values.
        if lmode is InterpolationMode.nearest:
            val = tex_batch(p_floor[...,0], p_floor[...,1])
        elif lmode is InterpolationMode.linear:
            x0, y0 = p_floor[...,0], p_floor[...,1]
            s00 = tex_batch(  x0,   y0)
            s01 = tex_batch(  x0, 1+y0)
            s10 = tex_batch(1+x0,   y0)
            s11 = tex_batch(1+x0, 1+y0)
            w1 = p - p_floor
            w0 = 1. - w1
            val = (
                w0[...,0] * (w0[...,1] * s00 + w1[...,1] * s01) + 
                w1[...,0] * (w0[...,1] * s10 + w1[...,1] * s11)
            )
        
        # val = val * valid
        # val[torch.isnan(val)] = 0.0

        # TODO: should be added;
        # but might cause RuntimeError: Function 'MulBackward0' returned nan values in its 0th output.
        val[:, ~valid] = 0.0
        return val
    
    def shading_all(self, uv, valid, bmode=BoundaryMode.replicate, lmode=InterpolationMode.linear):
        # uv shape [nC, nb_rays, N, 2]
        # valid shape [nC, nb_rays, N]
        # texture shape [batch_size, nC, h, w]
        # p = uv * (self.texturesize[None, None, ...]-1)
        p = uv * (self.texturesize-1) # [nC, nb_rays, N, 2]
        p_floor = torch.floor(p).long()

        def tex_all(p):
            """
            Texture indexing function, handle various boundary conditions.
            """
            if bmode is BoundaryMode.zero:
                raise NotImplementedError()
            elif bmode is BoundaryMode.replicate:
                p[..., 0] = torch.clamp(p[..., 0], min=0, max=self.texturesize[0].item()-1)
                p[..., 1] = torch.clamp(p[..., 1], min=0, max=self.texturesize[1].item()-1)
            elif bmode is BoundaryMode.symmetric:
                raise NotImplementedError()
            elif bmode is BoundaryMode.periodic:
                raise NotImplementedError()
            
            p_h = p[:, :, :, 0]
            p_w = p[:, :, :, 1]

            batch_idx = torch.arange(self.texture.shape[0]).view(-1, 1, 1, 1).to(self.device)  # Shape [batch_size, 1, 1, 1]
            channel_idx = torch.arange(self.texture.shape[1]).view(1, -1, 1, 1).to(self.device)        # Shape [1, nC, 1, 1]
            p_h = p_h.unsqueeze(0).expand(self.texture.shape[0], -1, -1, -1)  # [batch_size, nC, nb_rays, N]
            p_w = p_w.unsqueeze(0).expand(self.texture.shape[0], -1, -1, -1)  # [batch_size, nC, nb_rays, N]

            img = self.texture[batch_idx, channel_idx, p_h, p_w]  # [batch_size, nC, nb_rays, N]

            return img # [batchsize, nC, nb_rays, N]

        # Texture fetching, requires interpolation to compute fractional pixel values.
        if lmode is InterpolationMode.nearest:
            val = tex_all(p_floor)
        elif lmode is InterpolationMode.linear:
            # p_floor [nC, nb_rays, N, 2]
            x_add = torch.tensor([1, 0]).unsqueeze(0).unsqueeze(0).unsqueeze(0).to(self.device)
            y_add = torch.tensor([0, 1]).unsqueeze(0).unsqueeze(0).unsqueeze(0).to(self.device)
            xy_add = torch.tensor([1, 1]).unsqueeze(0).unsqueeze(0).unsqueeze(0).to(self.device) 
            s00 = tex_all(p_floor)
            s01 = tex_all(p_floor + y_add)
            s10 = tex_all(p_floor + x_add)
            s11 = tex_all(p_floor + xy_add)
            w1 = (p - p_floor).unsqueeze(0) # [1, nC, nb_rays, N, 2]
            w0 = 1. - w1
            val = (
                w0[...,0] * (w0[...,1] * s00 + w1[...,1] * s01) + 
                w1[...,0] * (w0[...,1] * s10 + w1[...,1] * s11)
            ) # [batchsize, nC, nb_rays, N]
        # val = val * valid
        # val[torch.isnan(val)] = 0.0

        # TODO: should be added;
        # but might cause RuntimeError: Function 'MulBackward0' returned nan values in its 0th output.
        val[~valid.unsqueeze(0).expand(self.texture.shape[0], -1, -1, -1)] = 0.0
        return val # [batchsize, nC, nb_rays, N]

    def draw_points(self, ax, options, seq=range(3)):
        """
        Visualization function.
        """
        coeffs = np.array([
            [ 1, 1, 1],
            [-1, 1, 1],
            [-1,-1, 1],
            [ 1,-1, 1],
            [ 1, 1, 1]
        ])
        points_local = torch.Tensor(coeffs * np.append(self.halfsize.cpu().detach().numpy(), 0)).to(self.device)
        points_world = self.to_world.transform_point(points_local).T.cpu().detach().numpy()
        ax.plot(points_world[seq[0]], points_world[seq[1]], points_world[seq[2]], options)

