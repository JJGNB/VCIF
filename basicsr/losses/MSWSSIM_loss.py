import torch
from torch import nn as nn
from torch.nn import functional as F
import kornia
from math import exp
from basicsr.utils.registry import LOSS_REGISTRY
_reduction_modes = ['none', 'mean', 'sum']
@LOSS_REGISTRY.register()
class MSWSSIMLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSWSSIMLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')
        self.loss_weight = loss_weight
        self.reduction = reduction
    def forward(self, fuse,vi,ir, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        Windows=[11,9,7,5,3]
        MSWSSIM_loss=0
        if ir.shape[1] !=3:
            ir_3=torch.concat([ir,ir,ir],dim=1)
        for s in Windows:
            ssim=SSIM(window_size=s,size_average=False)
            # loss1,sigma1=ssim(vis,fuse)
            loss1,sigma1=ssim(vi,fuse)
            loss2, sigma2 = ssim(ir_3, fuse)
            r=sigma1/(sigma1+sigma2+0.0000001)
            temp=1-torch.mean(r*loss1)-torch.mean((1-r)*loss2)
            MSWSSIM_loss=MSWSSIM_loss+temp
        if self.reduction=='mean':
            loss=MSWSSIM_loss/5.0
        else:
            loss=MSWSSIM_loss
        return self.loss_weight*loss
class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
            window = window.to('cuda:0')
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            window = window.to('cuda:0')
            self.window = window
            self.channel = channel

        return ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)
def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()
def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        # ret = ssim_map.mean(1).mean(1).mean(1)
        ret = ssim_map
    v=torch.zeros_like(sigma1_sq)+0.0001
    sigma1=torch.where(sigma1_sq<0.0001,v,sigma1_sq)
    if full:
        return ret, cs
    return ret,sigma1