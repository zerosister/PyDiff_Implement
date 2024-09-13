'''
ArcFace MS1MV3 r50
https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch
'''
import math
import os.path as osp
from basicsr.utils.img_util import img2tensor
import torch
from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.losses.losses import r1_penalty
from basicsr.metrics import calculate_metric
from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from collections import OrderedDict
from torch.nn import functional as F
from torchvision.ops import roi_align
from tqdm import tqdm
import sys
sys.path.append('.')
import numpy as np
import cv2
import torch.nn as nn
cv2.setNumThreads(1)
import torchvision
from torch.nn.parallel import DataParallel, DistributedDataParallel
from scripts.utils import generate_position_encoding, hiseq_color_cv2_img, pad_tensor, pad_tensor_back

@MODEL_REGISTRY.register()
class PyDiffModel(BaseModel):

    def __init__(self, opt):
        super(PyDiffModel, self).__init__(opt)

        # define u-net network
        self.unet = build_network(opt['network_unet'])
        self.unet = self.model_to_device(self.unet)     # 将构建好的 U-Net 网络放到 GPU 上
        opt['network_ddpm']['denoise_fn'] = self.unet   # U-Net 网络作为 denoise_fn

        self.global_corrector = build_network(opt['network_global_corrector'])    # 构建全局矫正网络
        self.global_corrector = self.model_to_device(self.global_corrector)     # 将构建好的全局矫正网络放到 GPU 上
        opt['network_ddpm']['color_fn'] = self.global_corrector                 # 全局矫正网络作为 color_fn （颜色矫正）

        self.ddpm = build_network(opt['network_ddpm'])          # 根据配置构建一个 DDPM（去噪扩散概率模型）网络
        self.ddpm = self.model_to_device(self.ddpm)
        if isinstance(self.ddpm, (DataParallel, DistributedDataParallel)):
            self.bare_model = self.ddpm.module
        else:
            self.bare_model = self.ddpm

        self.bare_model.set_new_noise_schedule(schedule_opt=opt['ddpm_schedule'],
                                            device=self.device)                   # 设置 DDPM 网络的噪声调度
        self.bare_model.set_loss(device=self.device)                           # 设置 DDPM 网络的损失函数 (L2)
        self.print_network(self.ddpm)

        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.ddpm, load_path, self.opt['path'].get('strict_load_g', True), param_key)
        if 'metrics' in self.opt['val'] and 'lpips' in self.opt['val']['metrics']:
            import lpips
            self.lpips = lpips.LPIPS(net='alex')
            self.lpips = self.model_to_device(self.lpips)
            if isinstance(self.lpips, (DataParallel, DistributedDataParallel)):
                self.lpips_bare_model = self.lpips.module
            else:
                self.lpips_bare_model = self.lpips



        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.ddpm.train()

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']

        # ----------- optimizer g ----------- #
        net_g_reg_ratio = 1
        normal_params = []
        logger = get_root_logger()
        for _, param in self.ddpm.named_parameters():
            if self.opt['train'].get('frozen_denoise', False):
                if 'denoise' in _:
                    logger.info(f'frozen {_}')
                    continue
            normal_params.append(param)
        optim_params_g = [{  # add normal params first
            'params': normal_params,
            'lr': train_opt['optim_g']['lr']
        }]
        optim_type = train_opt['optim_g'].pop('type')
        lr = train_opt['optim_g']['lr'] * net_g_reg_ratio
        betas = (0**net_g_reg_ratio, 0.99**net_g_reg_ratio)
        self.optimizer_g = self.get_optimizer(optim_type, optim_params_g, lr, betas=betas)
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        self.LR = data['LR'].to(self.device)  # 低分辨率图像移动到指定设备
        self.HR = data['HR'].to(self.device)  # 高分辨率图像移动到指定设备
        if 'pad_left' in data:
            self.pad_left = data['pad_left'].to(self.device)
            self.pad_right = data['pad_right'].to(self.device)
            self.pad_top = data['pad_top'].to(self.device)
            self.pad_bottom = data['pad_bottom'].to(self.device)

    def optimize_parameters(self, current_iter):
        # if self.opt['train'].get('mask_loss', False):
        #     assert self.opt['train'].get('cal_noise_only', False), "mask_loss can only used with cal_noise_only, now"
        # optimize net_g
        assert 'ddpm_cs' in self.opt['train'].get('train_type', None), "train_type must be ddpm_cs"
        self.optimizer_g.zero_grad()
        pred_noise, noise, x_recon_cs, x_start, t, color_scale = self.ddpm(self.HR, self.LR, 
                  train_type=self.opt['train'].get('train_type', None),
                  different_t_in_one_batch=self.opt['train'].get('different_t_in_one_batch', None),
                  t_sample_type=self.opt['train'].get('t_sample_type', None),
                  pred_type=self.opt['train'].get('pred_type', None),
                  clip_noise=self.opt['train'].get('clip_noise', None),
                  color_shift=self.opt['train'].get('color_shift', None),
                  color_shift_with_schedule= self.opt['train'].get('color_shift_with_schedule', None),
                  t_range=self.opt['train'].get('t_range', None),
                  cs_on_shift=self.opt['train'].get('cs_on_shift', None),
                  cs_shift_range=self.opt['train'].get('cs_shift_range', None),
                  t_border=self.opt['train'].get('t_border', None),
                  down_uniform=self.opt['train'].get('down_uniform', False),
                  down_hw_split=self.opt['train'].get('down_hw_split', False),
                  pad_after_crop=self.opt['train'].get('pad_after_crop', False),
                  input_mode=self.opt['train'].get('input_mode', None),
                  crop_size=self.opt['train'].get('crop_size', None),
                  divide=self.opt['train'].get('divide', None),
                  frozen_denoise=self.opt['train'].get('frozen_denoise', None),
                  cs_independent=self.opt['train'].get('cs_independent', None),
                  shift_x_recon_detach=self.opt['train'].get('shift_x_recon_detach', None))
        if self.opt['train'].get('vis_train', False) and current_iter <= self.opt['train'].get('vis_num', 100) and \
            self.opt['rank'] == 0:
            '''
            When the parameter 'vis_train' is set to True, the training process will be visualized. 
            The value of 'vis_num' corresponds to the number of visualizations to be generated.
            '''
            save_img_path = osp.join(self.opt['path']['visualization'], 'train',
                                            f'{current_iter}_noise_level_{self.bare_model.t}.png')
            x_recon_print = tensor2img(self.bare_model.x_recon, min_max=(-1, 1))
            noise_print = tensor2img(self.bare_model.noise, min_max=(-1, 1))
            pred_noise_print = tensor2img(self.bare_model.pred_noise, min_max=(-1, 1))
            x_start_print = tensor2img(self.bare_model.x_start, min_max=(-1, 1))
            x_noisy_print = tensor2img(self.bare_model.x_noisy, min_max=(-1, 1))

            img_print  = np.concatenate([x_start_print, noise_print, x_noisy_print, pred_noise_print, x_recon_print], axis=0)
            imwrite(img_print, save_img_path)
        l_g_total = 0
        loss_dict = OrderedDict()

        l_g_x0 = F.l1_loss(x_recon_cs, x_start) * self.opt['train'].get('l_g_x0_w', 1.0)
        if self.opt['train'].get('gamma_limit_train', None) and color_scale <= self.opt['train'].get('gamma_limit_train', None):
            l_g_x0 = l_g_x0 * 1e-12
        loss_dict['l_g_x0'] = l_g_x0
        l_g_total += l_g_x0

        if not self.opt['train'].get('frozen_denoise', False):
            l_g_noise = F.l1_loss(pred_noise, noise)
            loss_dict['l_g_noise'] = l_g_noise
            l_g_total += l_g_noise

        l_g_total.backward()
        self.optimizer_g.step()
        self.log_dict = self.reduce_loss_dict(loss_dict)

    def test(self):
        if self.opt['val'].get('test_speed', False):
            assert self.opt['val'].get('ddim_pyramid', False), "please use ddim_pyramid"
            with torch.no_grad():
                iterations = self.opt['val'].get('iterations', 100)
                input_size = self.opt['val'].get('input_size', [400, 600])

                LR = torch.randn(1, 10, input_size[0], input_size[1]).to(self.device)
                starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                self.bare_model.denoise_fn.eval()
                
                # GPU warm up
                print('GPU warm up')
                for _ in tqdm(range(50)):
                    self.output = self.bare_model.ddim_pyramid_sample(LR, 
                                                    pyramid_list=self.opt['val'].get('pyramid_list'),
                                                    continous=self.opt['val'].get('ret_process', False), 
                                                    ddim_timesteps = self.opt['val'].get('ddim_timesteps', 50),
                                                    return_pred_noise=self.opt['val'].get('return_pred_noise', False),
                                                    return_x_recon=self.opt['val'].get('ret_x_recon', False),
                                                    ddim_discr_method=self.opt['val'].get('ddim_discr_method', 'uniform'),
                                                    ddim_eta=self.opt['val'].get('ddim_eta', 0.0),
                                                    pred_type=self.opt['val'].get('pred_type', 'noise'),
                                                    clip_noise=self.opt['val'].get('clip_noise', False),
                                                    save_noise=self.opt['val'].get('save_noise', False),
                                                    color_gamma=self.opt['val'].get('color_gamma', None),
                                                    color_times=self.opt['val'].get('color_times', 1),
                                                    return_all=self.opt['val'].get('ret_all', False))
                
                # speed test
                times = torch.zeros(iterations)     # Store the time of each iteration
                for iter in tqdm(range(iterations)):
                    starter.record()
                    self.output = self.bare_model.ddim_pyramid_sample(LR, 
                                                    pyramid_list=self.opt['val'].get('pyramid_list'),
                                                    continous=self.opt['val'].get('ret_process', False), 
                                                    ddim_timesteps = self.opt['val'].get('ddim_timesteps', 50),
                                                    return_pred_noise=self.opt['val'].get('return_pred_noise', False),
                                                    return_x_recon=self.opt['val'].get('ret_x_recon', False),
                                                    ddim_discr_method=self.opt['val'].get('ddim_discr_method', 'uniform'),
                                                    ddim_eta=self.opt['val'].get('ddim_eta', 0.0),
                                                    pred_type=self.opt['val'].get('pred_type', 'noise'),
                                                    clip_noise=self.opt['val'].get('clip_noise', False),
                                                    save_noise=self.opt['val'].get('save_noise', False),
                                                    color_gamma=self.opt['val'].get('color_gamma', None),
                                                    color_times=self.opt['val'].get('color_times', 1),
                                                    return_all=self.opt['val'].get('ret_all', False))
                    ender.record()
                    # Synchronize GPU
                    torch.cuda.synchronize()
                    curr_time = starter.elapsed_time(ender)
                    times[iter] = curr_time
                    # print(curr_time)

                mean_time = times.mean().item()
                logger = get_root_logger()
                logger.info("Inference time: {:.6f}, FPS: {} ".format(mean_time, 1000/mean_time))
                import sys
                sys.exit()
        with torch.no_grad():
            self.bare_model.denoise_fn.eval()
            self.output = self.bare_model.ddim_pyramid_sample(self.LR, 
                                                pyramid_list=self.opt['val'].get('pyramid_list'),
                                                continous=self.opt['val'].get('ret_process', False), 
                                                ddim_timesteps = self.opt['val'].get('ddim_timesteps', 50),
                                                return_pred_noise=self.opt['val'].get('return_pred_noise', False),
                                                return_x_recon=self.opt['val'].get('ret_x_recon', False),
                                                ddim_discr_method=self.opt['val'].get('ddim_discr_method', 'uniform'),
                                                ddim_eta=self.opt['val'].get('ddim_eta', 0.0),
                                                pred_type=self.opt['val'].get('pred_type', 'noise'),
                                                clip_noise=self.opt['val'].get('clip_noise', False),
                                                save_noise=self.opt['val'].get('save_noise', False),
                                                color_gamma=self.opt['val'].get('color_gamma', None),
                                                color_times=self.opt['val'].get('color_times', 1),
                                                return_all=self.opt['val'].get('ret_all', False),
                                                fine_diffV2=self.opt['val'].get('fine_diffV2', False),
                                                fine_diffV2_st=self.opt['val'].get('fine_diffV2_st', 200),
                                                fine_diffV2_num_timesteps=self.opt['val'].get('fine_diffV2_num_timesteps', 20),
                                                do_some_global_deg=self.opt['val'].get('do_some_global_deg', False),
                                                use_up_v2=self.opt['val'].get('use_up_v2', False))
            self.bare_model.denoise_fn.train()
            
            if hasattr(self, 'pad_left') and not self.opt['val'].get('ret_process', False):
                self.output = pad_tensor_back(self.output, self.pad_left, self.pad_right, self.pad_top, self.pad_bottom)
                self.LR = pad_tensor_back(self.LR, self.pad_left, self.pad_right, self.pad_top, self.pad_bottom)
                self.HR = pad_tensor_back(self.HR, self.pad_left, self.pad_right, self.pad_top, self.pad_bottom)

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def find_lol_dataset(self, name):
        if name[0] == 'r':
            return 'SYNC'
        elif name[0] == 'n' or name[0] == 'l':
            return 'REAL'
        else:
            return 'LOL'

    # 非分布式环境下进行模型验证
    # dataloader: 数据集加载器
    # current_iter: 当前迭代次数
    # tb_logger: tensorboard日志记录器
    # save_img: 是否保存验证结果图片
    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None # 检查配置中是否包含验证指标
        # 如果配置中包含验证指标，则初始化指标字典
        if self.opt['val'].get('fix_seed', False):
            next_seed = np.random.randint(10000000)
            logger = get_root_logger()
            logger.info(f'next_seed={next_seed}')
        # 如果配置中设置了返回处理结果，则不计算指标
        if self.opt['val'].get('ret_process', False):
            with_metrics = False
        # 初始化指标字典，用于存储每个指标的累积结果
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
        metric_data = dict()          # 初始化指标数据字典，用于存储每个图像的指标数据
        metric_data_pytorch = dict()  # 初始化指标数据字典，用于存储计算 PyTorch 指标的结果
        pbar = tqdm(total=len(dataloader), unit='image')  # 初始化进度条
        # 如果设置了分割日志
        if self.opt['val'].get('split_log', False):
            self.split_results = {}
            self.split_results['SYNC'] = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            self.split_results['REAL'] = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            self.split_results['LOL'] = {metric: 0 for metric in self.opt['val']['metrics'].keys()}

        # 开始验证 
        for idx, val_data in enumerate(dataloader):
            if self.opt['val'].get('fix_seed', False):
                from basicsr.utils import set_random_seed
                set_random_seed(0)
            # 如果配置中没有设置计算所有数据或计算分数，并且时间步数大于等于 4 且索引大于等于 3，则跳出循环
            if not self.opt['val'].get('cal_all', False) and \
               not self.opt['val'].get('cal_score', False) and \
               int(self.opt['ddpm_schedule']['n_timestep']) >= 4 and idx >= 3:
                break
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)  # 将数据输入模型
            self.test()               # 进行模型推理

            visuals = self.get_current_visuals()  # 获取当前可视化结果
            # 将可视化结果转换为图像
            sr_img = tensor2img([visuals['sr']], min_max=(-1, 1))
            gt_img = tensor2img([visuals['gt']], min_max=(-1, 1))
            lq_img = tensor2img([visuals['lq']], min_max=(-1, 1))
            # 若配置中设置了使用 Kind 进行对齐，则进行对齐
            if self.opt['val'].get('use_kind_align', False):
                '''
                References:
                https://github.com/zhangyhuaee/KinD/blob/master/evaluate_LOLdataset.py
                https://github.com/wyf0912/LLFlow/blob/main/code/test.py
                '''
                gt_mean = np.mean(gt_img)
                sr_mean = np.mean(sr_img)
                sr_img = sr_img * gt_mean / sr_mean
                sr_img = np.clip(sr_img, 0, 255)
                sr_img = sr_img.astype('uint8')
            # 将图像数据添加到指标数据字典中，用于计算指标
            metric_data['img'] = sr_img
            metric_data['img2'] = gt_img
            metric_data_pytorch['img'] = self.output
            metric_data_pytorch['img2'] = self.HR
            path = val_data['lq_path'][0]
            
            # 存储图片
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
                # print(save_img_path)
                if idx < self.opt['val'].get('show_num', 3) or self.opt['val'].get('show_all', False):
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                            f'{img_name}_{current_iter}.png')
                    if not self.opt['val'].get('ret_process', False):
                        if self.opt['val'].get('only_save_sr', False):
                            save_img_path = osp.join(self.opt['path']['visualization'],
                                            f'{img_name}.png')
                            imwrite(sr_img, save_img_path)
                        else:
                            imwrite(np.concatenate([lq_img, sr_img, gt_img], axis=1), save_img_path)
                    else:
                        imwrite(sr_img, save_img_path)

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    if 'lpips' in opt_['type']:
                        opt_['device'] = self.device
                        opt_['model'] = self.lpips_bare_model
                    if 'pytorch' in opt_['type']:
                        if self.opt['val'].get('split_log', False):
                            self.split_results[self.find_lol_dataset(img_name)][name] += calculate_metric(metric_data_pytorch, opt_).item()
                        self.metric_results[name] += calculate_metric(metric_data_pytorch, opt_).item()
                    else:
                        if self.opt['val'].get('split_log', False):
                            self.split_results[self.find_lol_dataset(img_name)][name] += calculate_metric(metric_data, opt_)
                        self.metric_results[name] += calculate_metric(metric_data, opt_)
            
            # tentative for out of GPU memory
            del self.LR
            del self.output
            torch.cuda.empty_cache()
            pbar.update(1)
            pbar.set_description(f'Test {img_name}')
            if self.opt['val'].get('cal_score_num', None):
                if idx >= self.opt['val'].get('cal_score_num', None):
                    break
        pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
        if self.opt['val'].get('cal_score', False):
            import sys
            sys.exit()
        if self.opt['val'].get('fix_seed', False):
            from basicsr.utils import set_random_seed
            set_random_seed(next_seed)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        logger = get_root_logger()
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}\n'
        logger.info(log_str)
        if self.opt['val'].get('split_log', False):
            for dataset_name, num in zip(['LOL', 'REAL', 'SYNC'], [15, 100, 100]):
                log_str = f'Validation {dataset_name}\n'
                for metric, value in self.split_results[dataset_name].items():
                    log_str += f'\t # {metric}: {value/num:.4f}\n'
                logger.info(log_str)
        
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        if self.LR.shape != self.output.shape:
            self.LR = F.interpolate(self.LR, self.output.shape[2:])
            self.HR = F.interpolate(self.HR, self.output.shape[2:])
        out_dict['gt'] = self.HR.detach().cpu()
        out_dict['sr'] = self.output.detach().cpu()
        out_dict['lq'] = self.LR[:, :3, :, :].detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        self.save_network([self.ddpm], 'net_g', current_iter, param_key=['params'])

    def single_png_output(self, gt_img_path, lr_img_path, save_path):
        # Step 1: 加载输入图像
        data =self.getitem(input_path=lr_img_path, gt_path=gt_img_path)
        self.feed_data2(data=data)  # 注意这里是 lq 图像的输入

        # Step 3: 模型推理
        self.test()  # 进行推理（调用 test 函数）

        # Step 4: 获取推理后的图像并保存
        visuals = self.get_current_visuals()  # 获取当前的视觉输出结果
        sr_img = tensor2img([visuals['sr']], min_max=(-1, 1))  # 将推理结果转换为图像

        # Step 5: 保存增强后的图像
        imwrite(sr_img, save_path)  # 保存图像到目标路径
        print(f"图像已增强并保存至 {save_path}")
        
    def feed_data2(self, data):
      self.LR = data['LR'].to(self.device)  # 低分辨率图像移动到指定设备
      if data['HR'] is None:
        self.HR = None
      else:
        self.HR = data['HR'].to(self.device)  # 高分辨率图像移动到指定设备
      if 'pad_left' in data:
          self.pad_left = data['pad_left'].to(self.device)
          self.pad_right = data['pad_right'].to(self.device)
          self.pad_top = data['pad_top'].to(self.device)
          self.pad_bottom = data['pad_bottom'].to(self.device)

    def getitem(self, input_path, gt_path = None):
        if gt_path is not None:
          gt_img = cv2.cvtColor(cv2.imread(gt_path), cv2.COLOR_BGR2RGB) / 255.
        input_img = cv2.cvtColor(cv2.imread(input_path), cv2.COLOR_BGR2RGB) / 255.
        newOpt = self.opt['train']

        if newOpt.get('bright_aug', False):
            bright_aug_range = newOpt.get('bright_aug_range', [0.5, 1.5])
            input_img = input_img * np.random.uniform(*bright_aug_range)
        
        if newOpt.get('concat_with_hiseq', False):
            hiseql = cv2.cvtColor(hiseq_color_cv2_img(cv2.imread(input_path)), cv2.COLOR_BGR2RGB) / 255.
            if newOpt.get('hiseq_random_cat', False) and np.random.uniform(0, 1) < newOpt.get('hiseq_random_cat_p', 0.5):
                input_img = np.concatenate([hiseql, input_img], axis=2)
            else:
                input_img = np.concatenate([input_img, hiseql], axis=2)
            if newOpt.get('random_drop', False):
                if np.random.uniform() <= newOpt.get('random_drop_p', 1.0):
                    random_drop_val = newOpt.get('random_drop_val', 0)
                    if np.random.uniform() < 0.5:
                        input_img[:, :, :3] = random_drop_val
                    else:
                        input_img[:, :, 3:] = random_drop_val
            if newOpt.get('random_drop_hiseq', False):
                if np.random.uniform() < 0.5:
                    input_img[:, :, 3:] = 0

        if newOpt.get('use_flip', False) and np.random.uniform() < 0.5:
            if gt_path is not None:
              gt_img = cv2.flip(gt_img, 1, gt_img)
            input_img = cv2.flip(input_img, 1, input_img)
        
        if newOpt.get('input_with_low_resolution_hq', False):
            low_resolution_hq_size = newOpt.get('low_resolution_hq_size', 256)
            if gt_path is not None:
              self.low_resolution_hq = cv2.resize(
                gt_img,
                (low_resolution_hq_size, low_resolution_hq_size)
            )
        
        
        if newOpt.get('concat_with_position_encoding', False):
            H, W, _ = input_img.shape
            L = newOpt.get('position_encoding_L', 1)
            position_encoding = generate_position_encoding(H, W, L)
            input_img = np.concatenate([input_img, position_encoding], axis=2)
        
        if newOpt.get('resize', False):
            resize_size = newOpt['resize_size']
            if newOpt.get('resize_nearest', False):
                if gt_path is not None:
                  gt_img = cv2.resize(gt_img, dsize=(resize_size[1], resize_size[0]), interpolation=cv2.INTER_NEAREST)
                input_img = cv2.resize(input_img, dsize=(resize_size[1], resize_size[0]), interpolation=cv2.INTER_NEAREST)
            else:
                if gt_path is not None:
                  gt_img = cv2.resize(gt_img, dsize=(resize_size[1], resize_size[0]))
                input_img = cv2.resize(input_img, dsize=(resize_size[1], resize_size[0]))

        if newOpt['input_mode'] == 'crop':
          crop_size = newOpt['crop_size']
          print(f"crop_size: {crop_size}")
          H, W, _ = input_img.shape
          if gt_path is not None:
            assert input_img.shape[:2] == gt_img.shape[:2], f"{input_img.shape}, {gt_img.shape}, {gt_path}"
          h = np.random.randint(0, H - crop_size + 1)
          w = np.random.randint(0, W - crop_size + 1)
        if gt_path is not None:
          gt_img = gt_img[h: h + crop_size, w: w + crop_size, :]
        input_img = input_img[h: h + crop_size, w: w + crop_size, :]
        if newOpt['input_mode'] == 'pad':
            divide = newOpt['divide']
            if gt_path is not None:
              gt_img_pt = torch.from_numpy(gt_img.transpose((2, 0, 1)))
              gt_img_pt = torch.unsqueeze(gt_img_pt, 0)
              gt_img_pt, pad_left, pad_right, pad_top, pad_bottom = pad_tensor(gt_img_pt, divide)
              gt_img_pt = gt_img_pt[0, ...]
              gt_img = gt_img_pt.numpy().transpose((1, 2, 0))
            
            input_img_pt = torch.from_numpy(input_img.transpose((2, 0, 1)))
            input_img_pt = torch.unsqueeze(input_img_pt, 0)
            input_img_pt, pad_left, pad_right, pad_top, pad_bottom = pad_tensor(input_img_pt, divide)
            input_img_pt = input_img_pt[0, ...]
            input_img = input_img_pt.numpy().transpose((1, 2, 0))
        if gt_path is not None:
          gt_img_pt = torch.from_numpy(gt_img.transpose((2, 0, 1)))
        input_img_pt = torch.from_numpy(input_img.transpose((2, 0, 1)))
        if hasattr(self, 'low_resolution_hq'):
            self.low_resolution_hq = torch.from_numpy(
                self.low_resolution_hq.transpose((2, 0, 1))
            ).float()

        input_img_pt = input_img_pt.float()
        if gt_path is not None:
          gt_img_pt = gt_img_pt.float()
          cv2.normalize(gt_img_pt, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], inplace=True)
        cv2.normalize(input_img_pt, [0.5] * input_img_pt.shape[0], [0.5] * input_img_pt.shape[0], inplace=True)
        
        if hasattr(self, 'low_resolution_hq'):
            cv2.normalize(
                self.low_resolution_hq, 
                [0.5, 0.5, 0.5], 
                [0.5, 0.5, 0.5], 
                inplace=True
            )
        if gt_path is not None:
          return_dict = {"LR": input_img_pt, "HR": gt_img_pt, "lq_path": gt_path}
        else:
          return_dict = {"LR": input_img_pt, "HR": None, "lq_path": input_path}
          if newOpt['input_mode'] == 'pad':
            return_dict["pad_left"] = pad_left
            return_dict["pad_right"] = pad_right
            return_dict["pad_top"] = pad_top
            return_dict["pad_bottom"] = pad_bottom
        if newOpt.get('input_with_low_resolution_hq', False):
            return_dict["low_resolution_hq"] = self.low_resolution_hq
        return return_dict