import torch
from torch.nn import functional as F
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm
from basicsr.archs import build_network_IVIF,build_network
# from basicsr.archs import build_network_VCIF
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from .sr_model import SRModel
from .base_model import BaseModel
import torchvision
import torchvision.transforms.functional as TF
@MODEL_REGISTRY.register()
class VCIFModel(BaseModel):
    def __init__(self, opt):
        super(VCIFModel, self).__init__(opt)
        # define network
        self.net_vi=build_network(opt=opt['network_vi'])
        self.net_ir=build_network(opt=opt['network_ir'])
        self.net_g = build_network_IVIF(self.net_ir,self.net_vi,opt['network_g']['fusion_type'])
        self.net_ir = self.model_to_device(self.net_ir)
        self.net_vi = self.model_to_device(self.net_vi)
        self.net_g = self.model_to_device(self.net_g)
        # a = torch.normal(mean=0, std=0.03, size=[6, 3, 256, 256]).cuda()
        # self.print_network(self.net_g)

        # load pretrained models
        load_path_fuse = self.opt['path'].get('pretrain_network_g', None)
        load_path_ir=self.opt['path'].get('pretrain_network_ir', None)
        load_path_vi=self.opt['path'].get('pretrain_network_vi', None)
        if load_path_ir is not None and load_path_vi is not None:
            param_key_vi = self.opt['path'].get('param_key_vi', 'params')
            param_key_ir = self.opt['path'].get('param_key_ir', 'params')
            self.load_network(self.net_ir,load_path_ir, self.opt['path'].get('strict_load_ir', True), param_key_ir)
            self.load_network(self.net_vi,load_path_vi, self.opt['path'].get('strict_load_vi', True), param_key_vi)
        if load_path_fuse is not None:
            param_key_fuse = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g,load_path_fuse, self.opt['path'].get('strict_load_g', True), param_key_fuse)
        if self.is_train:
            self.init_training_settings()
    def init_training_settings(self):
        self.net_ir.eval()
        self.net_vi.eval()
        for param in self.net_ir.parameters():
            param.requires_grad = False
        for param in self.net_vi.parameters():
            param.requires_grad = False
        self.net_g.train()
        train_opt = self.opt['train']
        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if train_opt.get('color_opt'):
            self.cri_color = build_loss(train_opt['color_opt']).to(self.device)
        else:
            self.cri_color = None

        if self.cri_pix is None and self.cri_perceptual is None and self.cri_color is None:
            raise ValueError('All losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()
    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)
    def feed_data(self, data):
        self.ir = data['ir'].to(self.device)
        self.vi = data['vi'].to(self.device)
        self.ve = data['ve'].to(self.device)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        noise = torch.normal(mean=0,std=0.03,size=self.vi.size()).to(self.device)
        noise_input=noise+self.vi
        noise_input=torch.clip(noise_input,0,1)
        # torchvision.utils.save_image(self.ir[0:1,:,:,:],"./tt_ir.png")
        # torchvision.utils.save_image(self.vi[0:1,:,:,:],"./tt_vi.png")
        # ir_t=torch.ones([6,1,256,256])*0.5
        # ir_t=ir_t.cuda()

        self.output = self.net_g(torch.concat([self.ir,noise_input],dim=1))
        # torchvision.utils.save_image(self.output[0:1,:,:,:],"./fuse.png")
        l_total = 0
        loss_dict = OrderedDict()
        ir_en=TF.adjust_contrast(self.ir,1.5)
        self.ve=TF.adjust_contrast(self.ve,1.1)
        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.ve,ir_en)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        # perceptual loss
        if self.cri_perceptual:
            l_percep = self.cri_perceptual(self.output, self.ve,ir_en)
            l_total += l_percep
            loss_dict['l_percep'] = l_percep
        if self.cri_color:
            l_color = self.cri_color(self.output, self.ve)
            l_total += l_color
            loss_dict['l_color'] = l_color
        loss_dict['l_total'] = l_total
        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)
    def test_selfensemble(self):
        # TODO: to be tested
        # 8 augmentations
        # modified from https://github.com/thstkdgus35/EDSR-PyTorch

        def _transform(v, op):
            # if self.precision != 'single': v = v.float()
            v2np = v.data.cpu().numpy()
            if op == 'v':
                tfnp = v2np[:, :, :, ::-1].copy()
            elif op == 'h':
                tfnp = v2np[:, :, ::-1, :].copy()
            elif op == 't':
                tfnp = v2np.transpose((0, 1, 3, 2)).copy()

            ret = torch.Tensor(tfnp).to(self.device)
            # if self.precision == 'half': ret = ret.half()

            return ret

        # prepare augmented data
        lq_list = [self.lq]
        for tf in 'v', 'h', 't':
            lq_list.extend([_transform(t, tf) for t in lq_list])

        # inference
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                out_list = [self.net_g_ema(aug) for aug in lq_list]
        else:
            self.net_g.eval()
            with torch.no_grad():
                out_list = [self.net_g_ema(aug) for aug in lq_list]
            self.net_g.train()

        # merge results
        for i in range(len(out_list)):
            if i > 3:
                out_list[i] = _transform(out_list[i], 't')
            if i % 4 > 1:
                out_list[i] = _transform(out_list[i], 'h')
            if (i % 4) % 2 == 1:
                out_list[i] = _transform(out_list[i], 'v')
        output = torch.cat(out_list, dim=0)

        self.output = output.mean(dim=0, keepdim=True)

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)
    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['vi_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            metric_data['fuse_img'] = sr_img
            img_vi = tensor2img([visuals['vi']])
            img_ir = tensor2img([visuals['ir']])
            metric_data['img'] = img_vi
            metric_data['img2'] = img_ir
            # tentative for out of GPU memory
            del self.ir
            del self.vi
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["name"]}.png')
                imwrite(sr_img, save_img_path)

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'
        log_str+="optimal SD: 0.218, optimal MI: 3.287, optimal VIF: 1.044\n"
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)
    def test(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(self.lq)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(torch.concat([self.ir,self.vi],dim=1))
            self.net_g.train()
    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['vi'] = self.vi.detach().cpu()
        out_dict['ir'] = self.ir.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        return out_dict
    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)
