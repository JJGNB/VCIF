from torch.utils import data as data
from torchvision.transforms.functional import normalize
from os import path as osp
from basicsr.data.data_util import paired_paths_from_folder, paired_paths_from_lmdb, paired_paths_from_meta_info_file,pairedIVIF_paths_from_folder, pairedIVIF_paths_from_lmdb
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, bgr2ycbcr, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY
import torchvision

@DATASET_REGISTRY.register()
class IVIFImageDataset(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and GT image pairs.

    There are three modes:

    1. **lmdb**: Use lmdb files. If opt['io_backend'] == lmdb.
    2. **meta_info_file**: Use meta information file to generate paths. \
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. **folder**: Scan folders to generate paths. The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
        dataroot_gt (str): Data root path for gt.
        dataroot_lq (str): Data root path for lq.
        meta_info_file (str): Path for meta information file.
        io_backend (dict): IO backend type and other kwarg.
        filename_tmpl (str): Template for each filename. Note that the template excludes the file extension.
            Default: '{}'.
        gt_size (int): Cropped patched size for gt patches.
        use_hflip (bool): Use horizontal flips.
        use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).
        scale (bool): Scale, which will be added automatically.
        phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(IVIFImageDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.ir_folder=opt['dataroot_ir']
        self.vi_folder=opt['dataroot_vi']
        self.ve_folder = opt['dataroot_ve']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'
        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.ir_folder, self.vi_folder,self.ve_folder]
            self.io_backend_opt['client_keys'] = ['ir', 'vi', "ve"]
            self.paths = pairedIVIF_paths_from_lmdb([self.ir_folder, self.vi_folder,self.ve_folder], ['ir', 'vi', "ve"])
        else:
            self.paths = pairedIVIF_paths_from_folder([self.ir_folder, self.vi_folder,self.ve_folder], ['ir', 'vi', "ve"], self.filename_tmpl)
        # self.paths = paired_paths_from_folder([self.lq_folder, self.gt_folder], ['lq', 'gt'], self.filename_tmpl)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        ve_path = self.paths[index]['ve_path']
        img_bytes = self.file_client.get(ve_path, 've')
        img_ve = imfrombytes(img_bytes, float32=True)
        ir_path = self.paths[index]['ir_path']
        img_bytes = self.file_client.get(ir_path, 'ir')
        img_ir = imfrombytes(img_bytes, float32=True)
        vi_path = self.paths[index]['vi_path']
        img_bytes = self.file_client.get(vi_path, 'vi')
        img_vi = imfrombytes(img_bytes, float32=True)

        # augmentation for training
        # if self.opt['phase'] == 'train':
        #     gt_size = self.opt['gt_size']
        #     # random crop
        #     img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
        #     # flip, rotation
        #     img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_hflip'], self.opt['use_rot'])

        # color space transform
        # if 'color' in self.opt and self.opt['color'] == 'y':
        #     # img_ve = bgr2ycbcr(img_ve, y_only=True)[..., None]
        #     img_ir_255 = bgr2ycbcr(img_ir, y_only=True)[..., None]
            # img_vi = bgr2ycbcr(img_vi, y_only=True)[..., None]

        # crop the unmatched GT images during validation or testing, especially for SR benchmark datasets
        # TODO: It is better to update the datasets, rather than force to crop
        if self.opt['phase'] != 'train':
            img_ve = img_ve[0:img_ir.shape[0] * scale, 0:img_ir.shape[1] * scale, :]

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_ve= img2tensor(img_ve, bgr2rgb=True, float32=True,yonly=False)
        img_ir= img2tensor(img_ir, bgr2rgb=True, float32=True,yonly=True)
        img_vi = img2tensor(img_vi, bgr2rgb=True, float32=True,yonly=False)
        # img_ir_255=img2tensor(img_ir_255, bgr2rgb=False, float32=True,yonly=False)
        # torchvision.utils.save_image(img_ir[:,:,:],"./tt_ir.png")
        # torchvision.utils.save_image(img_ir_255[:,:,:],"./tt_ir_16.png")
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_ve, self.mean, self.std, inplace=True)
            normalize(img_ir, self.mean, self.std, inplace=True)
            normalize(img_vi, self.mean, self.std, inplace=True)

        return {'ir': img_ir, 'vi': img_vi, 've': img_ve, 'ir_path': ir_path, 'vi_path': vi_path, 've_path': ve_path}

    def __len__(self):
        return len(self.paths)
