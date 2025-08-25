import numpy as np
from skimage import exposure
import cv2
from skimage.filters import gaussian
import torch
def AAB(img):
    if img.shape[0]==3:
        img=img.unsqueeze(dim=0)
        img=rgb_to_ycbcr(img)
        img_v=img[0,0,:,:]
    else:
        img_v=img
    img_numpy=img_v.detach().cpu().numpy()
    p2, p98 = np.percentile(img_numpy, (5, 95))
    img_numpy[img_numpy>=p98]=np.NAN
    ab=np.nanmean(img_numpy)
    ab=torch.tensor(ab).cuda()
    return ab
def get_rgb_img(img_Y,img_rgb):
    y1 = img_Y
    ycbcr_image2 = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
    chans = cv2.split(ycbcr_image2)
    m = cv2.merge([y1, chans[1], chans[2]])
    rgb_image = cv2.cvtColor(m, cv2.COLOR_YCrCb2RGB)
    return rgb_image
def ycbcr_to_rgb(image: torch.Tensor) -> torch.Tensor:
    r"""Convert an YCbCr image to RGB.

    The image data is assumed to be in the range of (0, 1).

    Args:
        image (torch.Tensor): YCbCr Image to be converted to RGB with shape :math:`(*, 3, H, W)`.

    Returns:
        torch.Tensor: RGB version of the image with shape :math:`(*, 3, H, W)`.

>> Meta_results = ycbcr_to_rgb(input)  # 2x3x4x5
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}"
                         .format(image.shape))

    y: torch.Tensor = image[..., 0, :, :]
    cb: torch.Tensor = image[..., 1, :, :]
    cr: torch.Tensor = image[..., 2, :, :]

    delta: float = 0.5
    cb_shifted: torch.Tensor = cb - delta
    cr_shifted: torch.Tensor = cr - delta

    r: torch.Tensor = y + 1.403 * cr_shifted
    g: torch.Tensor = y - 0.714 * cr_shifted - 0.344 * cb_shifted
    b: torch.Tensor = y + 1.773 * cb_shifted
    return torch.stack([r, g, b], -3)
def rgb_to_ycbcr(image: torch.Tensor) -> torch.Tensor:
    r"""Convert an RGB image to YCbCr.

    Args:
        image (torch.Tensor): RGB Image to be converted to YCbCr.

    Returns:
        torch.Tensor: YCbCr version of the image.
    """

    if not torch.is_tensor(image):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}"
                         .format(image.shape))

    r: torch.Tensor = image[..., 0, :, :]
    g: torch.Tensor = image[..., 1, :, :]
    b: torch.Tensor = image[..., 2, :, :]

    delta = .5
    y: torch.Tensor = .299 * r + .587 * g + .114 * b
    cb: torch.Tensor = (b - y) * .564 + delta
    cr: torch.Tensor = (r - y) * .713 + delta
    return torch.stack((y, cb, cr), -3)
def red(i1,a=1):
    r_ = np.mean(i1[:,:,0])
    g_ = np.mean(i1[:,:,1])
    i1[:,:,0] = i1[:,:,0] + a*(1-i1[:,:,0])*(g_-r_)*i1[:,:,1]
    return i1
def sat(img, saturation=0.3):
    img = (img *255).astype(np.uint8)
    hlsImg = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    hlsImg=hlsImg.astype("float")
    hlsImg[:, :, 2] = (1.0 + saturation / float(1)) * hlsImg[:, :, 2]
    hlsImg[:, :, 2]=np.clip(hlsImg[:, :, 2],0,255)
    # hlsImg[:, :, 2][hlsImg[:, :, 2] > 255] = 255
    hlsImg = hlsImg.astype("uint8")
    # hlsImg[:, :, 2][hlsImg[:, :, 2] < 0] = 0
    lsImg = cv2.cvtColor(hlsImg, cv2.COLOR_HLS2RGB)/255.0
    # hlsImg[:, :, 2]=scale(hlsImg[:, :, 2])
    return lsImg

def scale(img):
    return (img - np.min(img)) / (np.max(img) - np.min(img))

def LAB(img):
    img = (img *255).astype(np.uint8)
    lab = cv2.cvtColor(img,cv2.COLOR_RGB2LAB)
    A= np.mean(np.mean(lab[:,:,1]))
    B= np.mean(np.mean(lab[:,:,2]))
    if A>B:
        lab[:,:,2] = A/B*lab[:,:,2]
    else:
        lab[:, :, 1] = B/A * lab[:, :, 1]
    out = cv2.cvtColor(lab,cv2.COLOR_LAB2RGB)/255
    return out
def gray_world(img):
    out = np.zeros(np.shape(img))
    # p5, p95 = np.percentile(img, (4, 96))
    img_copy=np.copy(img)
    # img_copy[img >= p95] = np.NAN
    # avg = np.nanmean(img_copy)
    avg = np.mean(np.mean(img))
    for j in range(3):
        m = np.nansum(np.nansum(img_copy[:, :, j], axis=0), axis=0)
        n = np.size(img[:, :, j])
        scale = n/m
        g_weight = (avg*scale)
        out[:, :, j] = img[:, :, j]*g_weight
    return out
def ACES(img,k=0.7):
    img = img * k
    a = 2.51
    b = 0.03
    c = 2.43
    d = 0.59
    e = 0.14
    img[:,:,0] = (img[:,:,0]*(a*img[:,:,0]+b))/(img[:,:,0]*(c*img[:,:,0]+d)+e)
    img[:, :, 1] = (img[:, :, 1] * (a * img[:, :, 1] + b)) / (img[:, :, 1] * (c * img[:, :, 1] + d) + e)
    img[:,:,2] = (img[:,:,2]*(a*img[:,:,2]+b))/(img[:,:,2]*(c*img[:,:,2]+d)+e)
    return img
def gamma_correction(image_rgb,gamma=0.7):
    image_rgb[:, :, 0] = exposure.adjust_gamma(image_rgb[:, :, 0], gamma=gamma)
    image_rgb[:, :, 1] = exposure.adjust_gamma(image_rgb[:, :, 1], gamma=gamma)
    image_rgb[:, :, 2] = exposure.adjust_gamma(image_rgb[:, :, 2], gamma=gamma)
    return  image_rgb
def Rescale_intensity(image_rgb,x=1):
    p2, p98 = np.percentile(image_rgb[:, :, 0], (x, 100-x))
    image_rgb[:, :, 0] = exposure.rescale_intensity(image_rgb[:, :, 0], in_range=(p2, p98))
    p2, p98 = np.percentile(image_rgb[:, :, 1], (x, 100-x))
    image_rgb[:, :, 1] = exposure.rescale_intensity(image_rgb[:, :, 1], in_range=(p2, p98))
    p2, p98 = np.percentile(image_rgb[:, :, 2], (x, 100-x))
    image_rgb[:, :, 2] = exposure.rescale_intensity(image_rgb[:, :, 2], in_range=(p2, p98))
    return  image_rgb
def High_pass(img,sigma=0.5):
    gauss_out = gaussian(img, sigma=sigma, channel_axis=-1)
    img_out = img - gauss_out + 0.5
    mask_1 = img_out  < 0
    mask_2 = img_out  > 1
    img_out = img_out * (1-mask_1)
    img_out = img_out * (1-mask_2) + mask_2
    return img_out

def Overlay(img_1, img_2):
    mask = img_2 < 0.5
    img = 2 * img_1 * img_2 * mask + (1-mask) * (1- 2 * (1-img_1)*(1-img_2))
    return img
def bluegreen(i1,a=1):
    b_ = np.mean(i1[:,:,2])
    g_ = np.mean(i1[:,:,1])
    if g_>b_:
        i1[:,:,2] = i1[:,:,2] + a*(1-i1[:,:,2])*(g_-b_)*i1[:,:,1]
    else:
        i1[:, :, 1] = i1[:, :, 1] + a * (1 - i1[:, :, 1]) * (b_ - g_) * i1[:, :, 2]
    return i1
def  run_pipeline(i1,a1,a2,a3,a4,a5,a6,a7):
    if a1==1:
        i1 = gamma_correction(i1)
        i1 = np.minimum(np.maximum(i1, 0), 1)
    if a2==1:
        i1 = sat(i1)
        i1 = np.minimum(np.maximum(i1, 0), 1)
    if a3==1:
        i1 = Rescale_intensity(i1)
        i1 = np.minimum(np.maximum(i1, 0), 1)
    if a4==1:
        i1 = gray_world(i1)
        i1 = np.minimum(np.maximum(i1, 0), 1)
    if a7==1:
        i2 = High_pass(i1)
        i3 = Overlay(i1, i2)
        i1 = np.minimum(np.maximum(i3, 0), 1)
    return i1