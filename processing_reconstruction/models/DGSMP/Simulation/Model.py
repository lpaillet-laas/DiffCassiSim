import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F


class Resblock(nn.Module):
    def __init__(self, HBW):
        super(Resblock, self).__init__()
        self.block1 = nn.Sequential(nn.Conv2d(HBW, HBW, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(HBW, HBW, kernel_size=3, stride=1, padding=1))
        self.block2 = nn.Sequential(nn.Conv2d(HBW, HBW, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(HBW, HBW, kernel_size=3, stride=1, padding=1))

    def forward(self, x):
        tem = x
        r1 = self.block1(x)
        out = r1 + tem
        r2 = self.block2(out)
        out = r2 + out
        return out

class Encoding(nn.Module):
    def __init__(self):
        super(Encoding, self).__init__()
        self.E1 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                                nn.ReLU()
                                )
        self.E2 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                                nn.ReLU()
                                )
        self.E3 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                                nn.ReLU()
                                )
        self.E4 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                                nn.ReLU()
                                )
        self.E5 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                                nn.ReLU()
                                )

    def forward(self, x):
        ## encoding blocks
        E1 = self.E1(x)
        E2 = self.E2(F.avg_pool2d(E1, kernel_size=2, stride=2))
        E3 = self.E3(F.avg_pool2d(E2, kernel_size=2, stride=2))
        E4 = self.E4(F.avg_pool2d(E3, kernel_size=2, stride=2))
        E5 = self.E5(F.avg_pool2d(E4, kernel_size=2, stride=2))
        return E1, E2, E3, E4, E5

class Decoding(nn.Module):
    def __init__(self, Ch=28, kernel_size=[7,7,7]):
        super(Decoding, self).__init__()
        self.upMode = 'bilinear'
        self.Ch = Ch
        out_channel1 = Ch * kernel_size[0]
        out_channel2 = Ch * kernel_size[1]
        out_channel3 = Ch * kernel_size[2]
        self.D1 = nn.Sequential(nn.Conv2d(in_channels=128+128, out_channels=128, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                                nn.ReLU()
                                )
        self.D2 = nn.Sequential(nn.Conv2d(in_channels=128+64, out_channels=64, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                                nn.ReLU()
                                )
        self.D3 = nn.Sequential(nn.Conv2d(in_channels=64+64, out_channels=64, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                                nn.ReLU()
                                )
        self.D4 = nn.Sequential(nn.Conv2d(in_channels=64+32, out_channels=32, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                                nn.ReLU()
                                )

        self.w_generator = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                                         nn.ReLU(),
                                         nn.Conv2d(in_channels=32, out_channels=self.Ch, kernel_size=3, stride=1, padding=1),
                                         nn.ReLU(),
                                         nn.Conv2d(in_channels=self.Ch, out_channels=self.Ch, kernel_size=1, stride=1, padding=0)
                                         )

        self.filter_g_1      = nn.Sequential(nn.Conv2d(64 + 32, out_channel1, kernel_size=3, stride=1, padding=1),
                                             nn.ReLU(),
                                             nn.Conv2d(out_channel1, out_channel1, kernel_size=3, stride=1, padding=1),
                                             nn.ReLU(),
                                             nn.Conv2d(out_channel1, out_channel1, 1, 1, 0)
                                             )

        self.filter_g_2      = nn.Sequential(nn.Conv2d(64 + 32, out_channel2, kernel_size=3, stride=1, padding=1),
                                             nn.ReLU(),
                                             nn.Conv2d(out_channel2, out_channel2, kernel_size=3, stride=1, padding=1),
                                             nn.ReLU(),
                                             nn.Conv2d(out_channel2, out_channel2, 1, 1, 0)
                                             )

        self.filter_g_3      = nn.Sequential(nn.Conv2d(64 + 32, out_channel3, kernel_size=3, stride=1, padding=1),
                                             nn.ReLU(),
                                             nn.Conv2d(out_channel3, out_channel3, kernel_size=3, stride=1, padding=1),
                                             nn.ReLU(),
                                             nn.Conv2d(out_channel3, out_channel3, 1, 1, 0)
                                             )


    def forward(self, E1, E2, E3, E4, E5):
        ## decoding blocks
        D1 = self.D1(torch.cat([E4, F.interpolate(E5, scale_factor=2, mode=self.upMode)], dim=1))
        D2 = self.D2(torch.cat([E3, F.interpolate(D1, scale_factor=2, mode=self.upMode)], dim=1))
        D3 = self.D3(torch.cat([E2, F.interpolate(D2, scale_factor=2, mode=self.upMode)], dim=1))
        D4 = self.D4(torch.cat([E1, F.interpolate(D3, scale_factor=2, mode=self.upMode)], dim=1))

        ## estimating the regularization parameters w
        w = self.w_generator(D4)

        ## generate 3D filters
        f1 = self.filter_g_1(torch.cat([E1, F.interpolate(D3, scale_factor=2, mode=self.upMode)], dim=1))
        f2 = self.filter_g_2(torch.cat([E1, F.interpolate(D3, scale_factor=2, mode=self.upMode)], dim=1))
        f3 = self.filter_g_3(torch.cat([E1, F.interpolate(D3, scale_factor=2, mode=self.upMode)], dim=1))
        return w, f1, f2, f3


class HSI_CS(nn.Module):
    def __init__(self, Ch, stages, dispersion_pixels, mapping_cube):
        super(HSI_CS, self).__init__()
        self.Ch = Ch
        self.s  = stages
        self.filter_size = [7,7,7]  ## 3D filter size
        self.dispersion_pixels = dispersion_pixels
        self.mapping_cube = mapping_cube

        ## The modules for learning the measurement matrix A and A^T
        self.AT = nn.Sequential(nn.Conv2d(Ch, 64, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(),
                                Resblock(64), Resblock(64),
                                nn.Conv2d(64, Ch, kernel_size=3, stride=1, padding=1), nn.LeakyReLU())
        self.A  = nn.Sequential(nn.Conv2d(Ch, 64, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(),
                                Resblock(64), Resblock(64),
                                nn.Conv2d(64, Ch, kernel_size=3, stride=1, padding=1), nn.LeakyReLU())

        ## Encoding blocks
        self.Encoding = Encoding()

        ## Decoding blocks
        self.Decoding   = Decoding(Ch=self.Ch, kernel_size=self.filter_size)

        ## Dense connection
        self.conv  = nn.Conv2d(Ch, 32, kernel_size=3, stride=1, padding=1)
        self.Den_con1 = nn.Conv2d(32    , 32, kernel_size=1, stride=1, padding=0)
        self.Den_con2 = nn.Conv2d(32 * 2, 32, kernel_size=1, stride=1, padding=0)
        self.Den_con3 = nn.Conv2d(32 * 3, 32, kernel_size=1, stride=1, padding=0)
        self.Den_con4 = nn.Conv2d(32 * 4, 32, kernel_size=1, stride=1, padding=0)
        # self.Den_con5 = nn.Conv2d(32 * 5, 32, kernel_size=1, stride=1, padding=0)
        # self.Den_con6 = nn.Conv2d(32 * 6, 32, kernel_size=1, stride=1, padding=0)


        self.delta_0 = Parameter(torch.ones(1), requires_grad=True)
        self.delta_1 = Parameter(torch.ones(1), requires_grad=True)
        self.delta_2 = Parameter(torch.ones(1), requires_grad=True)
        self.delta_3 = Parameter(torch.ones(1), requires_grad=True)
        # self.delta_4 = Parameter(torch.ones(1), requires_grad=True)
        # self.delta_5 = Parameter(torch.ones(1), requires_grad=True)

        self._initialize_weights()
        torch.nn.init.normal_(self.delta_0, mean=0.1, std=0.01)
        torch.nn.init.normal_(self.delta_1, mean=0.1, std=0.01)
        torch.nn.init.normal_(self.delta_2, mean=0.1, std=0.01)
        torch.nn.init.normal_(self.delta_3, mean=0.1, std=0.01)
        # torch.nn.init.normal_(self.delta_4, mean=0.1, std=0.01)
        # torch.nn.init.normal_(self.delta_5, mean=0.1, std=0.01)


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                nn.init.constant_(m.bias.data, 0.0)


    def Filtering_1(self, cube, core):
        batch_size, bandwidth, height, width = cube.size()
        cube_pad = F.pad(cube, [self.filter_size[0] // 2, self.filter_size[0] // 2, 0, 0], mode='replicate')
        img_stack = []
        for i in range(self.filter_size[0]):
            img_stack.append(cube_pad[:, :, :, i:i + width])
        img_stack = torch.stack(img_stack, dim=1)
        out = torch.sum(core.mul_(img_stack), dim=1, keepdim=False)
        return out

    def Filtering_2(self, cube, core):
        batch_size, bandwidth, height, width = cube.size()
        cube_pad = F.pad(cube, [0, 0, self.filter_size[1] // 2, self.filter_size[1] // 2], mode='replicate')
        img_stack = []
        for i in range(self.filter_size[1]):
            img_stack.append(cube_pad[:, :, i:i + height, :])
        img_stack = torch.stack(img_stack, dim=1)
        out = torch.sum(core.mul_(img_stack), dim=1, keepdim=False)
        return out

    def Filtering_3(self, cube, core):
        batch_size, bandwidth, height, width = cube.size()
        cube_pad = F.pad(cube.unsqueeze(0).unsqueeze(0), pad=(0, 0, 0, 0, self.filter_size[2] // 2, self.filter_size[2] // 2)).squeeze(0).squeeze(0)
        img_stack = []
        for i in range(self.filter_size[2]):
            img_stack.append(cube_pad[:, i:i + bandwidth, :, :])
        img_stack = torch.stack(img_stack, dim=1)
        out = torch.sum(core.mul_(img_stack), dim=1, keepdim=False)
        return out

    def recon(self, res1, res2, Xt, i):
        if i == 0 :
            delta = self.delta_0
        elif i == 1:
            delta = self.delta_1
        elif i == 2:
            delta = self.delta_2
        elif i == 3:
            delta = self.delta_3
        # elif i == 4:
        #     delta = self.delta_4
        # elif i == 5:
        #     delta = self.delta_5

        Xt     =   Xt - 2 * delta * (res1 + res2)
        return Xt

    def y2x_init(self, y):
        ##  Spilt operator
        return self.shift_back(y, self.dispersion_pixels, n_lambda=self.Ch)

        """ sz = y.size()
        if len(sz) == 3:
            y = y.unsqueeze(0)
            bs = 1
        else:
            bs = sz[0]
        sz = y.size()
        x = torch.zeros([bs, 28, sz[2], sz[2]]).cuda()
        for t in range(28):
            temp = y[:, :, :, 0 + 2 * t : sz[2] + 2 * t]
            x[:, t, :, :] = temp.squeeze(1)
        return x """
    
    def shift_back(self, inputs, dispersion_pixels, n_lambda = 28):
        """
            Input [bs, H + disp[0], W + disp[1]]
                Mapping cube [n_wav, H, W, 2]
            Output [bs, n_wav, H, W]
        """
        bs,row,col = inputs.shape        
        normed_disp = (torch.norm(dispersion_pixels, dim=1)*torch.sign(dispersion_pixels[:, 1])).round().int() # Pixels to indices
        max_min = (normed_disp.max() - normed_disp.min()).round().int().item()
        abs_shift = normed_disp - normed_disp.min()

        output = torch.zeros(self.mapping_cube.shape[:-1], device=inputs.device, dtype=inputs.dtype).unsqueeze(0).repeat(bs, 1, 1, 1) # [bs, n_lambda, H, W]

        row_indices = self.mapping_cube[..., 0]
        col_indices = self.mapping_cube[..., 1]

        for b in range(bs):
            for i in range(n_lambda):
                output[b, i] = inputs[b, row_indices[i], col_indices[i]].rot90(2, (0, 1)).flip(0)
        
        shifted_output = torch.zeros(bs, n_lambda, output.shape[2], output.shape[3]+ max_min, device=inputs.device, dtype=inputs.dtype) # [bs, n_lambda, H, W + +-||disp||]

        for i in range(n_lambda):
            shifted_output[:, i, :, abs_shift[i]: output.shape[-1] + abs_shift[i]] = output[:, i, :, :]
        
        return output, shifted_output
    
    def y2x(self, y):
        ##  Spilt operator
        return self.shift_back_wo_mapping(y, self.dispersion_pixels, n_lambda=self.Ch)

        """ sz = y.size()
        if len(sz) == 3:
            y = y.unsqueeze(0)
            bs = 1
        else:
            bs = sz[0]
        sz = y.size()
        x = torch.zeros([bs, 28, sz[2], sz[2]]).cuda()
        for t in range(28):
            temp = y[:, :, :, 0 + 2 * t : sz[2] + 2 * t]
            x[:, t, :, :] = temp.squeeze(1)
        return x """

    def shift_back_wo_mapping(self, inputs, dispersion_pixels, n_lambda = 28):
        """
            Input [bs, H, W + disp[1]]
            Output [bs, n_wav, H, W]
        """
        bs,H,W = inputs.shape        
        normed_disp = (torch.norm(dispersion_pixels, dim=1)*torch.sign(dispersion_pixels[:, 1])).round().int() # Pixels to indices
        max_min = (normed_disp.max() - normed_disp.min()).round().int().item()
        abs_shift = normed_disp - normed_disp.min()

        output = torch.zeros(self.mapping_cube.shape[:-1], device=inputs.device, dtype=inputs.dtype).unsqueeze(0).repeat(bs, 1, 1, 1) # [bs, n_lambda, H, W]

        for i in range(n_lambda):
             output[:, i, :, :] = inputs[:, :, abs_shift[i]: W - max_min + abs_shift[i]]
        
        return output
    
    # def shift_back(self, inputs, dispersion_pixels, n_lambda = 28): 
    # #TODO Previous shift_back
    #     """
    #         Input [bs, H + disp[0], W + disp[1]]
    #         Output [bs, n_wav, H, W]
    #     """
    #     bs, H, W = inputs.shape
    #     rounded_disp = dispersion_pixels.round().int() # Pixels to indices
    #     max_min = rounded_disp.max(dim=0).values - rounded_disp.min(dim=0).values
    #     output = torch.zeros(bs, n_lambda, H - max_min[0], W - max_min[1], device=inputs.device)
    #     abs_shift = rounded_disp - rounded_disp.min(dim=0).values
    #     for i in range(n_lambda):
    #         output[:, i, :, :] = inputs[:, abs_shift[i, 0]: H - max_min[0] + abs_shift[i, 0], abs_shift[i, 1]: W - max_min[1] + abs_shift[i, 1]]
    #     return output

    def x2y(self, x):
        return self.shift(x, self.dispersion_pixels, n_lambda=self.Ch).sum(1)
        """ ##  Shift and Sum operator
        sz = x.size()
        if len(sz) == 3:
            x = x.unsqueeze(0).unsqueeze(0)
            bs = 1
        else:
            bs = sz[0]
        sz = x.size()
        y = torch.zeros([bs, 1, sz[2], sz[2]+2*27]).cuda()
        for t in range(28):
            y[:, :, :, 0 + 2 * t : sz[2] + 2 * t] = x[:, t, :, :].unsqueeze(1) + y[:, :, :, 0 + 2 * t : sz[2] + 2 * t]
        return y """

    def shift(self, inputs, dispersion_pixels, n_lambda = 28):
        """
            Input [bs, n_wav, H, W]
            Output [bs, n_wav, H, W + +-||disp||]
        """
        bs, n_lambda, H, W = inputs.shape
        normed_disp = (torch.norm(dispersion_pixels, dim=1)*torch.sign(dispersion_pixels[:, 1])).round().int() # Pixels to indices
        max_min = (normed_disp.max() - normed_disp.min()).round().int().item()
        abs_shift = normed_disp - normed_disp.min()
        output = torch.zeros(bs, n_lambda, H, W + max_min, device=inputs.device, dtype=inputs.dtype)
        for i in range(n_lambda):
            output[:, i, :, abs_shift[i]: W + abs_shift[i]] = inputs[:, i, :, :]
        return output
    
    # def shift(self, inputs, dispersion_pixels, n_lambda = 28):
    # #TODO Previous shift
    #     """
    #         Input [bs, n_wav, H, W]
    #         Output [bs, n_wav, H + disp[0], W + disp[1]]
    #     """
    #     bs, n_lambda, H, W = inputs.shape
    #     rounded_disp = dispersion_pixels.round().int() # Pixels to indices
    #     max_min = rounded_disp.max(dim=0).values - rounded_disp.min(dim=0).values
    #     output = torch.zeros(bs, n_lambda, H + max_min[0], W + max_min[1], device=inputs.device)
    #     abs_shift = rounded_disp - rounded_disp.min(dim=0).values
    #     for i in range(n_lambda):
    #         output[:, i, abs_shift[i, 0]: H + abs_shift[i, 0], abs_shift[i, 1]: W + abs_shift[i, 1]] = inputs[:, i, :, :]
    #     return output
    

    def forward(self, y):
        ## The measurements y is split into a 3D data cube of size H × W × L to initialize x.
        Xt, expanded_y = self.y2x_init(y)
        y = expanded_y.sum(1)

        feature_list = []

        for i in range(0, self.s):
            AXt = self.x2y(self.A(Xt))  # y = Ax
            Res1 = self.AT(self.y2x(AXt - y))   # A^T * (Ax − y)

            fea = self.conv(Xt)

            if i == 0:
                feature_list.append(fea)
                fufea = self.Den_con1(fea)
            elif i == 1:
                feature_list.append(fea)
                fufea = self.Den_con2(torch.cat(feature_list, 1))
            elif i == 2:
                feature_list.append(fea)
                fufea = self.Den_con3(torch.cat(feature_list, 1))
            elif i == 3:
                feature_list.append(fea)
                fufea = self.Den_con4(torch.cat(feature_list, 1))
            # elif i == 4:
            #     feature_list.append(fea)
            #     fufea = self.Den_con5(torch.cat(feature_list, 1))
            # elif i == 5:
            #     feature_list.append(fea)
            #     fufea = self.Den_con6(torch.cat(feature_list, 1))

            E1, E2, E3, E4, E5 = self.Encoding(fufea)
            W, f1, f2, f3 = self.Decoding(E1, E2, E3, E4, E5)

            batch_size, p, height, width = f1.size()
            f1                           = F.normalize(f1.view(batch_size, self.filter_size[0], self.Ch, height, width),dim=1)
            batch_size, p, height, width = f2.size()
            f2                           = F.normalize(f2.view(batch_size, self.filter_size[1], self.Ch, height, width),dim=1)
            batch_size, p, height, width = f3.size()
            f3                           = F.normalize(f3.view(batch_size, self.filter_size[2], self.Ch, height, width),dim=1)

            ## Estimating the local means U
            u1 = self.Filtering_1(Xt, f1)
            u2 = self.Filtering_2(u1, f2)
            U = self.Filtering_3(u2, f3)

            ## w * (x − u)
            Res2 = (Xt - U).mul(W)

            ## Reconstructing HSIs
            Xt = self.recon(Res1, Res2, Xt, i)

        return Xt
