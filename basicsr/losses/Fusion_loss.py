import torch
from torch import nn as nn
from torch.nn import functional as F
import kornia
from basicsr.utils.registry import LOSS_REGISTRY
from .loss_util import weighted_loss
from .basic_loss import L1Loss

@LOSS_REGISTRY.register()
class ColorLoss(L1Loss):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        if reduction not in ['mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: mean | sum')
        super(ColorLoss, self).__init__(loss_weight=loss_weight, reduction=reduction)
    def forward(self, fuse,target, weight=None):
        YCbCr_Fuse = RGB2YCrCb(fuse)
        Cr_Fuse = YCbCr_Fuse[:,1:2,:,:]
        Cb_Fuse = YCbCr_Fuse[:,2:,:,:]
        YCbCr_R_vis = RGB2YCrCb(target)
        Cr_vis = YCbCr_R_vis[:,1:2,:,:]
        Cb_vis = YCbCr_R_vis[:,2:,:,:]
        cr_loss=F.l1_loss(Cr_Fuse, Cr_vis)
        cb_loss=F.l1_loss(Cb_Fuse, Cb_vis)
        loss=cr_loss+cb_loss
        return self.loss_weight*loss
@LOSS_REGISTRY.register()
class MaxFusionLoss(L1Loss):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        if reduction not in ['mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: mean | sum')
        super(MaxFusionLoss, self).__init__(loss_weight=loss_weight, reduction=reduction)
        self.sobelconv=Sobelxy()
        # self.eg=EightGrad()
    def forward(self,generate_img,image_vis_en,image_ir):
        B, C, W, H = image_vis_en.shape
        image_ir = image_ir.expand(B, C, W, H)
        x_in_max=torch.max(image_vis_en,image_ir)
        loss_in_=F.l1_loss(x_in_max, generate_img)
        loss_vis=F.l1_loss(image_vis_en,generate_img)
        loss_in=0.6*loss_in_+0.4*loss_vis
        vi_grad_en=self.sobelconv(image_vis_en)
        ir_grad=self.sobelconv(image_ir)
        # vi_grad_en=self.eg(image_vis_en)
        # ir_grad=self.eg(image_ir)
        B, C, K, W, H = vi_grad_en.shape
        ir_grad = ir_grad.expand(B, C, K, W, H)
        generate_img_grad=self.sobelconv(generate_img)
        # generate_img_grad=self.eg(generate_img)
        x_grad_joint=torch.max(ir_grad,vi_grad_en)
        loss_grad=F.l1_loss(x_grad_joint,generate_img_grad)
        loss_total = 2*loss_in + 1*loss_grad
        return self.loss_weight*loss_total
class EightGrad(nn.Module):
        def __init__(self):
            super(EightGrad, self).__init__()
        def forward(self,input):
            B,C,H,W=input.shape
            input_center = input[:, :, 1:-1, 1:-1]
            left = input[:, :, 0:-2, 1:-1]
            right = input[:, :, 2:, 1:-1]
            top = input[:, :, 1:-1, 0:-2]
            bottom = input[:, :, 1:-1, 2:]
            left_top = input[:, :, 0:-2, 0:-2]
            right_top = input[:, :, 2:, 0:-2]
            left_bottom = input[:, :, 0:-2, 2:]
            right_bottom = input[:, :, 2:, 2:]
            input1 = F.interpolate(input_center-left, size=(H, W))
            input2 = F.interpolate(input_center - right, size=(H, W))
            input3 = F.interpolate(input_center - top, size=(H, W))
            input4 = F.interpolate(input_center - bottom, size=(H, W))
            input5 = F.interpolate(input_center - left_top, size=(H, W))
            input6 = F.interpolate(input_center - right_top, size=(H, W))
            input7 = F.interpolate(input_center - left_bottom, size=(H, W))
            input8 = F.interpolate(input_center - right_bottom, size=(H, W))
            input_list=[input1,input2,input3,input4,input5,input6,input7,input8]
            input_list=[ip.unsqueeze(2) for ip in input_list]
            output=torch.concat(input_list,dim=2)
            return output
class Sobelxy(nn.Module):
        def __init__(self):
            super(Sobelxy, self).__init__()
            kernelx = [[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]]
            kernely = [[1, 2, 1],
                       [0, 0, 0],
                       [-1, -2, -1]]
            kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
            kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
            self.weightx = nn.Parameter(data=kernelx, requires_grad=False).to('cuda:0')
            self.weighty = nn.Parameter(data=kernely, requires_grad=False).to('cuda:0')

        def forward(self, x):
            b, c, w, h = x.shape
            batch_list = []
            for i in range(b):
                tensor_list = []
                for j in range(c):
                    sobelx_0 = F.conv2d(torch.unsqueeze(torch.unsqueeze(x[i, j, :, :], 0), 0), self.weightx, padding=1)
                    sobely_0 = F.conv2d(torch.unsqueeze(torch.unsqueeze(x[i, j, :, :], 0), 0), self.weighty, padding=1)
                    add_0 = torch.abs(sobelx_0) + torch.abs(sobely_0)
                    tensor_list.append(add_0)
                batch_list.append(torch.stack(tensor_list, dim=1))
            return torch.cat(batch_list, dim=0)
def RGB2YCrCb(input_im):
    im_flat = input_im.transpose(1, 3).transpose(
        1, 2).reshape(-1, 3)  # (nhw,c)
    R = im_flat[:, 0]
    G = im_flat[:, 1]
    B = im_flat[:, 2]

    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5
    Y = torch.unsqueeze(Y, 1)
    Cr = torch.unsqueeze(Cr, 1)
    Cb = torch.unsqueeze(Cb, 1)
    temp = torch.cat([Y, Cr, Cb], dim=1).to('cuda:0')
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
        .transpose(1, 3)
        .transpose(2, 3)
    )
    return out
def charbonnier_loss(pred, target, eps=1e-12):
    return torch.mean(torch.sqrt((pred - target)**2 + eps))
