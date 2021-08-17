from collections import OrderedDict
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.utils as torch_utils
from torch.optim import lr_scheduler
from torch.optim import Adam

from models.select_network import define_G, define_D
from models.model_base import ModelBase
from models.loss_ssim import SSIMLoss
from models.loss import GANLoss, PerceptualLoss

from utils.utils_model import test_mode
from utils.utils_regularizers import regularizer_orth, regularizer_clip
from data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from utils.img_process_util import USMSharp, filter2D
from utils.utils_image import paired_random_crop
from utils.diffjpeg import DiffJPEG
from torch.nn import functional as F
import utils.utils_image as utils


class ModelDegradationGAN(ModelBase):
    """Train with pixel loss"""

    def __init__(self, opt, scaler):
        super(ModelDegradationGAN, self).__init__(opt, scaler)
        # ------------------------------------
        # define network
        # ------------------------------------
        self.netG = define_G(opt)
        self.netG = self.model_to_device(self.netG)
        self.jpeger = DiffJPEG(differentiable=False).cuda()
        self.usm_sharpener = USMSharp().cuda()
        self.queue_size = opt['queue_size']

        self.ema_decay = opt['ema_decay'] if opt['ema_decay'] is not None else 0

        if self.ema_decay > 0:
            self.netG_ema = define_G(opt)
            self.netG_ema = self.model_to_device(self.netG_ema)
            self.model_ema(0)
            self.netG_ema.eval()

        if self.is_train:
            self.netD = define_D(opt)
            self.netD = self.model_to_device(self.netD)



    """
    # ----------------------------------------
    # Preparation before training with data
    # Save model during training
    # ----------------------------------------
    """

    # ----------------------------------------
    # initialize training
    # ----------------------------------------
    def init_train(self):
        self.opt_train = self.opt['train']  # training option
        self.load()  # load model
        self.netG.train()  # set training mode,for BN
        self.netD.train()
        self.define_loss()  # define loss
        self.define_optimizer()  # define optimizer
        self.load_optimizers()  # load optimizer
        self.define_scheduler()  # define scheduler
        self.log_dict = OrderedDict()  # log

    # ----------------------------------------
    # load pre-trained G model
    # ----------------------------------------
    def load(self):
        load_path_G = self.opt['path']['pretrained_netG']
        if load_path_G is not None:
            if hasattr(self, 'netG_ema'):
                print('Loading EMA_model for G [{:s}] ...'.format(load_path_G))
                self.load_network(load_path_G, self.netG_ema, strict=self.opt['path']['strict_netG'], param_key='params_ema')
            print('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, strict=self.opt['path']['strict_netG'])
        load_path_D = self.opt['path']['pretrained_netD']
        if self.opt['is_train'] and load_path_D is not None:
            print('Loading model for D [{:s}] ...'.format(load_path_D))
            self.load_network(load_path_D, self.netD, strict=self.opt['path']['strict_netD'])


    # ----------------------------------------
    # load optimizer
    # ----------------------------------------
    def load_optimizers(self):
        load_path_optimizerG = self.opt['path']['pretrained_optimizerG']
        if load_path_optimizerG is not None and self.opt_train['G_optimizer_reuse']:
            print('Loading optimizerG [{:s}] ...'.format(load_path_optimizerG))
            self.load_optimizer(load_path_optimizerG, self.G_optimizer)
        load_path_optimizerD = self.opt['path']['pretrained_optimizerD']
        if load_path_optimizerD is not None and self.opt_train['D_optimizer_reuse']:
            print('Loading optimizerD [{:s}] ...'.format(load_path_optimizerD))
            self.load_optimizer(load_path_optimizerD, self.D_optimizer)

    # ----------------------------------------
    # save model / optimizer(optional)
    # ----------------------------------------
    def save(self, iter_label):
        if hasattr(self, 'netG_ema'):
            self.save_network(self.save_dir, [self.netG, self.netG_ema], 'G', iter_label, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.save_dir, self.netG, 'G', iter_label)
        self.save_network(self.save_dir, self.netD, 'D', iter_label)
        if self.opt_train['G_optimizer_reuse']:
            self.save_optimizer(self.save_dir, self.G_optimizer, 'optimizerG', iter_label)
        if self.opt_train['D_optimizer_reuse']:
            self.save_optimizer(self.save_dir, self.D_optimizer, 'optimizerD', iter_label)

    # ----------------------------------------
    # define loss
    # ----------------------------------------
    def define_loss(self):
        # ------------------------------------
        # G_loss
        # ------------------------------------
        if self.opt_train['G_lossfn_weight'] > 0:
            G_lossfn_type = self.opt_train['G_lossfn_type']
            if G_lossfn_type == 'l1':
                self.G_lossfn = nn.L1Loss().to(self.device)
            elif G_lossfn_type == 'l2':
                self.G_lossfn = nn.MSELoss().to(self.device)
            elif G_lossfn_type == 'l2sum':
                self.G_lossfn = nn.MSELoss(reduction='sum').to(self.device)
            elif G_lossfn_type == 'ssim':
                self.G_lossfn = SSIMLoss().to(self.device)
            else:
                raise NotImplementedError('Loss type [{:s}] is not found.'.format(G_lossfn_type))
            self.G_lossfn_weight = self.opt_train['G_lossfn_weight']
        else:
            print('Do not use pixel loss.')
            self.G_lossfn = None

        # ------------------------------------
        # F_loss
        # ------------------------------------
        if self.opt_train['F_lossfn_weight'] > 0:
            F_lossfn_type = self.opt_train['F_lossfn_type']
            F_use_input_norm = self.opt_train['F_use_input_norm']
            F_feature_layer = self.opt_train['F_feature_layer']
            F_layer_weights = self.opt_train['F_layer_weights']
            if self.opt['dist']:
                self.F_lossfn = PerceptualLoss(feature_layer=F_feature_layer, use_input_norm=F_use_input_norm,
                                               lossfn_type=F_lossfn_type).to(self.device)
            else:
                self.F_lossfn = PerceptualLoss(layer_weights= F_layer_weights, feature_layer=F_feature_layer, use_input_norm=F_use_input_norm,
                                               lossfn_type=F_lossfn_type)
                self.F_lossfn.vgg = self.model_to_device(self.F_lossfn.vgg)
                self.F_lossfn.lossfn = self.F_lossfn.lossfn.to(self.device)
            self.F_lossfn_weight = self.opt_train['F_lossfn_weight']
        else:
            print('Do not use feature loss.')
            self.F_lossfn = None

        # ------------------------------------
        # D_loss
        # ------------------------------------
        self.D_lossfn = GANLoss(self.opt_train['gan_type'], 1.0, 0.0).to(self.device)
        self.D_lossfn_weight = self.opt_train['D_lossfn_weight']

        self.D_update_ratio = self.opt_train['D_update_ratio'] if self.opt_train['D_update_ratio'] else 1
        self.D_init_iters = self.opt_train['D_init_iters'] if self.opt_train['D_init_iters'] else 0
    # ----------------------------------------
    # define optimizer, G and D
    # ----------------------------------------
    def define_optimizer(self):
        G_optim_params = []  # optimizer parameter groups
        for k, v in self.netG.named_parameters():
            if v.requires_grad:
                G_optim_params.append(v)
        self.G_optimizer = Adam(G_optim_params, lr=self.opt_train['G_optimizer_lr'], weight_decay=0)
        self.D_optimizer = Adam(self.netD.parameters(), lr=self.opt_train['D_optimizer_lr'], weight_decay=0)
        del G_optim_params

    # ----------------------------------------
    # define scheduler, only "MultiStepLR"
    # ----------------------------------------
    def define_scheduler(self):
        G_scheduler_type = self.opt_train['G_scheduler_type']
        if G_scheduler_type == "MultiStepLR":
            self.schedulers.append(lr_scheduler.MultiStepLR(self.G_optimizer,
                                                            self.opt_train['G_scheduler_milestones'],
                                                            self.opt_train['G_scheduler_gamma']
                                                            ))
        elif G_scheduler_type == "CyclicLR":
            self.schedulers.append(lr_scheduler.CyclicLR(self.G_optimizer,
                                                         base_lr=1e-5,
                                                         max_lr=2e-4,
                                                         step_size_up=5,
                                                         step_size_down=100,
                                                         mode='exp_range',
                                                         gamma=0.9995))
        D_scheduler_type = self.opt_train['D_scheduler_type']
        if D_scheduler_type == "MultiStepLR":
            self.schedulers.append(lr_scheduler.MultiStepLR(self.D_optimizer,
                                                            self.opt_train['D_scheduler_milestones'],
                                                            self.opt_train['D_scheduler_gamma']
                                                            ))
        elif D_scheduler_type == "CyclicLR":
            self.schedulers.append(lr_scheduler.CyclicLR(self.D_optimizer,
                                                         base_lr=1e-5,
                                                         max_lr=2e-4,
                                                         step_size_up=5,
                                                         step_size_down=100,
                                                         mode='exp_range',
                                                         gamma=0.9995))
        else:
            print('error : set G_scheduler_type')

    """
    # ----------------------------------------
    # Optimization during training with data
    # Testing/evaluation
    # ----------------------------------------
    """

    @torch.no_grad()
    def _dequeue_and_enqueue(self):
        # training pair pool
        # initialize
        b, c, h, w = self.L.size()
        if not hasattr(self, 'queue_L'):
            assert self.queue_size % b == 0, 'queue size should be divisible by batch size'
            self.queue_L = torch.zeros(self.queue_size, c, h, w).cuda()
            _, c, h, w = self.H.size()
            self.queue_H = torch.zeros(self.queue_size, c, h, w).cuda()
            self.queue_ptr = 0
        if self.queue_ptr == self.queue_size:  # full
            # do dequeue and enqueue
            # shuffle
            idx = torch.randperm(self.queue_size)
            self.queue_L = self.queue_L[idx]
            self.queue_H = self.queue_H[idx]
            # get
            L_dequeue = self.queue_L[0:b, :, :, :].clone()
            H_dequeue = self.queue_H[0:b, :, :, :].clone()
            # update
            self.queue_L[0:b, :, :, :] = self.L.clone()
            self.queue_H[0:b, :, :, :] = self.H.clone()

            self.L = L_dequeue
            self.H = H_dequeue
        else:
            # only do enqueue
            self.queue_L[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.L.clone()
            self.queue_H[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.H.clone()
            self.queue_ptr = self.queue_ptr + b

    # ----------------------------------------
    # feed L/H data
    # ----------------------------------------
    @torch.no_grad()
    def feed_data(self, data):
        if self.is_train:
            self.H = data['H'].to(self.device)
            # USM(UnSharp Masking) H_images
            if self.opt['USM'] is True:
                self.H = self.usm_sharpener(self.H)

            self.kernel1 = data['kernel1'].to(self.device)
            self.kernel2 = data['kernel2'].to(self.device)
            self.sinc_kernel = data['sinc_kernel'].to(self.device)

            h, w = self.H.size()[2:4]
            # ----------------------- The first degradation process ----------------------- #
            # blur
            out = filter2D(self.H, self.kernel1)
            # random resize
            updown_type = random.choices(['up', 'down', 'keep'], self.opt['resize_prob'])[0] # [0.2, 0.7, 0.1]
            if updown_type == 'up':
                scale = np.random.uniform(1, self.opt['resize_range'][1]) # [0.15, 1.5]
            elif updown_type == 'down':
                scale = np.random.uniform(self.opt['resize_range'][0], 1) # [0.15, 1.5]
            else:
                scale = 1
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, scale_factor=scale, mode=mode)
            # noise
            gray_noise_prob = self.opt['gray_noise_prob']
            if np.random.uniform() < self.opt['gaussian_noise_prob']:
                out = random_add_gaussian_noise_pt(out, sigma_range=self.opt['noise_range'], gray_prob=gray_noise_prob, clip=True, rounds=False)
            else:
                out = random_add_poisson_noise_pt(out, scale_range=self.opt['poisson_scale_range'], gray_prob=gray_noise_prob, clip=True, rounds=False)
            #JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range'])
            out = torch.clamp(out, 0, 1)
            out = self.jpeger(out, quality=jpeg_p)

            # ----------------------- The second degradation process ----------------------- #
            # blur
            if np.random.uniform() < self.opt['second_blur_prob']:
                out = filter2D(out, self.kernel2)
            # random resize
            updown_type = random.choices(['up', 'down', 'keep'], self.opt['resize_prob2'])[0]
            if updown_type == 'up':
                scale = np.random.uniform(1, self.opt['resize_range2'][1])
            elif updown_type == 'down':
                scale = np.random.uniform(self.opt['resize_range2'][0], 1)
            else:
                scale = 1
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(
                out, size=(int(h / self.opt['scale'] * scale), int(w / self.opt['scale'] * scale)), mode=mode)
            # noise
            gray_noise_prob = self.opt['gray_noise_prob2']
            if np.random.uniform() < self.opt['gaussian_noise_prob2']:
                out = random_add_gaussian_noise_pt(
                    out, sigma_range=self.opt['noise_range2'], clip=True, rounds=False, gray_prob=gray_noise_prob)
            else:
                out = random_add_poisson_noise_pt(out, scale_range=self.opt['poisson_scale_range2'], gray_prob=gray_noise_prob, clip=True, rounds=False)

            # JPEG compression + the final sinc filter
            # We also need to resize images to desired sizes. We group [resize back + sinc filter] together
            # as one operation.
            # We consider two orders:
            #   1. [resize back + sinc filter] + JPEG compression
            #   2. JPEG compression + [resize back + sinc filter]
            # Empirically, we find other combinations (sinc + JPEG + Resize) will introduce twisted lines.
            if np.random.uniform() < 0.5:
                # resize back + the final sinc filter
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                out = F.interpolate(out, size=(h // self.opt['scale'], w // self.opt['scale']), mode=mode)
                out = filter2D(out, self.sinc_kernel)
                # JPEG compression
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range2'])
                out = torch.clamp(out, 0, 1)
                out = self.jpeger(out, quality=jpeg_p)
            else:
                # JPEG compression
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range2'])
                out = torch.clamp(out, 0, 1)
                out = self.jpeger(out, quality=jpeg_p)
                # resize back + the final sinc filter
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                out = F.interpolate(out, size=(h // self.opt['scale'], w // self.opt['scale']), mode=mode)
                out = filter2D(out, self.sinc_kernel)
            # clamp and round
            self.L = torch.clamp((out * 255.0).round(), 0, 255) / 255.

            # random crop
            H_size = self.opt['datasets']['train']['H_size']
            self.H, self.L = paired_random_crop(self.H, self.L, H_size, self.opt['scale'])

            # training pair pool
            self._dequeue_and_enqueue()
        else:
            self.L = data['L'].to(self.device)
            if 'H' in data:
                self.H = data['H'].to(self.device)
    # ----------------------------------------
    # feed L to netG
    # ----------------------------------------
    def netG_forward(self, test_ema = False):
        if test_ema == True:
            self.E = self.netG_ema(self.L)
        else:
            self.E = self.netG(self.L)

    # ----------------------------------------
    # update parameters and get loss
    # ----------------------------------------
    def optimize_parameters(self, current_step):
        # ------------------------------------
        # optimize G
        # ------------------------------------
        for p in self.netD.parameters():
            p.requires_grad = False

        self.G_optimizer.zero_grad()
        if self.amp:
            with torch.cuda.amp.autocast():
                self.netG_forward()

                loss_G_total = 0
                if current_step % self.D_update_ratio == 0 and current_step > self.D_init_iters:  # updata D first
                    if self.opt_train['G_lossfn_weight'] > 0:
                        G_loss = self.G_lossfn_weight * self.G_lossfn(self.E, self.H)
                        loss_G_total += G_loss  # 1) pixel loss

                    if self.opt_train['F_lossfn_weight'] > 0:
                        F_loss = self.F_lossfn_weight * self.F_lossfn(self.E, self.H)
                        loss_G_total += F_loss  # 2) VGG feature loss

                    if self.opt['train']['gan_type'] in ['gan', 'lsgan', 'wgan', 'softplusgan']:
                        pred_g_fake = self.netD(self.E)
                        D_loss = self.D_lossfn_weight * self.D_lossfn(pred_g_fake, True)
                    elif self.opt['train']['gan_type'] == 'ragan':
                        pred_d_real = self.netD(self.H).detach()
                        pred_g_fake = self.netD(self.E)
                        D_loss = self.D_lossfn_weight * (
                            self.D_lossfn(pred_d_real - torch.mean(pred_g_fake, 0, True), False) +
                            self.D_lossfn(pred_g_fake - torch.mean(pred_d_real, 0, True), True)) / 2
                    loss_G_total += D_loss                     # 3) GAN loss
            self.scaler.scale(loss_G_total).backward()
            self.scaler.step(self.G_optimizer)
            self.scaler.update()
        else:
            self.netG_forward()

            loss_G_total = 0
            if current_step % self.D_update_ratio == 0 and current_step > self.D_init_iters:  # updata D first
                if self.opt_train['G_lossfn_weight'] > 0:
                    G_loss = self.G_lossfn_weight * self.G_lossfn(self.E, self.H)
                    loss_G_total += G_loss  # 1) pixel loss

                if self.opt_train['F_lossfn_weight'] > 0:
                    F_loss = self.F_lossfn_weight * self.F_lossfn(self.E, self.H)
                    loss_G_total += F_loss  # 2) VGG feature loss

                if self.opt['train']['gan_type'] in ['gan', 'lsgan', 'wgan', 'softplusgan']:
                    pred_g_fake = self.netD(self.E)
                    D_loss = self.D_lossfn_weight * self.D_lossfn(pred_g_fake, True)
                elif self.opt['train']['gan_type'] == 'ragan':
                    pred_d_real = self.netD(self.H).detach()
                    pred_g_fake = self.netD(self.E)
                    D_loss = self.D_lossfn_weight * (
                            self.D_lossfn(pred_d_real - torch.mean(pred_g_fake, 0, True), False) +
                            self.D_lossfn(pred_g_fake - torch.mean(pred_d_real, 0, True), True)) / 2
                loss_G_total += D_loss  # 3) GAN loss

            loss_G_total.backward()
            self.G_optimizer.step()

        # ------------------------------------
        # optimize D
        # ------------------------------------
        for p in self.netD.parameters():
            p.requires_grad = True

        self.D_optimizer.zero_grad()

        if self.opt['dist']:
            # In order to avoid the error in distributed training:
            # "Error detected in CudnnBatchNormBackward: RuntimeError: one of
            # the variables needed for gradient computation has been modified by
            # an inplace operation",
            # we separate the backwards for real and fake, and also detach the
            # tensor for calculating mean.
            if self.opt_train['gan_type'] in ['gan', 'lsgan', 'wgan', 'softplusgan']:
                # real
                pred_d_real = self.netD(self.H)  # 1) real data
                l_d_real = self.D_lossfn(pred_d_real, True)
                l_d_real.backward()
                # fake
                pred_d_fake = self.netD(self.E.detach())  # 2) fake data, detach to avoid BP to G
                l_d_fake = self.D_lossfn(pred_d_fake, False)
                l_d_fake.backward()
            elif self.opt_train['gan_type'] == 'ragan':
                # real
                pred_d_fake = self.netD(self.E).detach()  # 1) fake data, detach to avoid BP to G
                pred_d_real = self.netD(self.H)  # 2) real data
                l_d_real = 0.5 * self.D_lossfn(pred_d_real - torch.mean(pred_d_fake, 0, True), True)
                l_d_real.backward()
                # fake
                pred_d_fake = self.netD(self.E.detach())
                l_d_fake = 0.5 * self.D_lossfn(pred_d_fake - torch.mean(pred_d_real.detach(), 0, True), False)
                l_d_fake.backward()
            self.D_optimizer.step()
        elif self.amp:
            with torch.cuda.amp.autocast():
                loss_D_total = 0
                pred_d_fake = self.netD(self.E.detach())  # 1) fake data, detach to avoid BP to G
                pred_d_real = self.netD(self.H)  # 2) real data
                if self.opt_train['gan_type'] in ['gan', 'lsgan', 'wgan', 'softplusgan']:
                    l_d_real = self.D_lossfn(pred_d_real, True)
                    l_d_fake = self.D_lossfn(pred_d_fake, False)
                    loss_D_total = l_d_real + l_d_fake
                elif self.opt_train['gan_type'] == 'ragan':
                    l_d_real = self.D_lossfn(pred_d_real - torch.mean(pred_d_fake, 0, True), True)
                    l_d_fake = self.D_lossfn(pred_d_fake - torch.mean(pred_d_real, 0, True), False)
                    loss_D_total = (l_d_real + l_d_fake) / 2
            self.scaler.scale(loss_D_total).backward()
            self.scaler.step(self.D_optimizer)
            self.scaler.update()
        else:
            loss_D_total = 0
            pred_d_fake = self.netD(self.E.detach())  # 1) fake data, detach to avoid BP to G
            pred_d_real = self.netD(self.H)  # 2) real data
            if self.opt_train['gan_type'] in ['gan', 'lsgan', 'wgan', 'softplusgan']:
                l_d_real = self.D_lossfn(pred_d_real, True)
                l_d_fake = self.D_lossfn(pred_d_fake, False)
                loss_D_total = l_d_real + l_d_fake
            elif self.opt_train['gan_type'] == 'ragan':
                l_d_real = self.D_lossfn(pred_d_real - torch.mean(pred_d_fake, 0, True), True)
                l_d_fake = self.D_lossfn(pred_d_fake - torch.mean(pred_d_real, 0, True), False)
                loss_D_total = (l_d_real + l_d_fake) / 2
            loss_D_total.backward()
            self.D_optimizer.step()

        # ------------------------------------
        # record log
        # ------------------------------------
        if current_step % self.D_update_ratio == 0 and current_step > self.D_init_iters:
        # self.log_dict['G_loss'] = G_loss.item()/self.E.size()[0]  # if `reduction='sum'`
            if self.opt_train['G_lossfn_weight'] > 0:
                self.log_dict['G_loss'] = G_loss.item()
            if self.opt_train['F_lossfn_weight'] > 0:
                self.log_dict['F_loss'] = F_loss.item()
            self.log_dict['D_loss'] = D_loss.item()

        self.log_dict['D_real'] = torch.mean(pred_d_real.detach())
        self.log_dict['D_fake'] = torch.mean(pred_d_fake.detach())

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)
    # ----------------------------------------
    # test / inference
    # ----------------------------------------
    def test(self):
        if hasattr(self, 'netG_ema'):
            self.netG_ema.eval()
            with torch.no_grad():
                if self.amp:
                    with torch.cuda.amp.autocast():
                        self.netG_forward(test_ema=True)
                else:
                    self.netG_forward(test_ema=True)
        else:
            self.netG.eval()
            with torch.no_grad():
                if self.amp:
                    with torch.cuda.amp.autocast():
                        self.netG_forward()
                else:
                    self.netG_forward()
            self.netG.train()

    # ----------------------------------------
    # test / inference x8
    # ----------------------------------------
    def testx8(self):
        self.netG.eval()
        with torch.no_grad():
            self.E = test_mode(self.netG, self.L, mode=3, sf=self.opt['scale'], modulo=1)
        self.netG.train()

    # ----------------------------------------
    # get log_dict
    # ----------------------------------------
    def current_log(self):
        return self.log_dict

    # ----------------------------------------
    # get L, E, H image
    # ----------------------------------------
    def current_visuals(self, need_H=True):
        out_dict = OrderedDict()
        out_dict['L'] = self.L.detach()[0].float().cpu()
        out_dict['E'] = self.E.detach()[0].float().cpu()
        if need_H:
            out_dict['H'] = self.H.detach()[0].float().cpu()
        return out_dict

    # ----------------------------------------
    # get L, E, H batch images
    # ----------------------------------------
    def current_results(self, need_H=True):
        out_dict = OrderedDict()
        out_dict['L'] = self.L.detach().float().cpu()
        out_dict['E'] = self.E.detach().float().cpu()
        if need_H:
            out_dict['H'] = self.H.detach().float().cpu()
        return out_dict

    """
    # ----------------------------------------
    # Information of netG
    # ----------------------------------------
    """

    # ----------------------------------------
    # print network
    # ----------------------------------------
    def print_network(self):
        msg = self.describe_network(self.netG)
        print(msg)
        if self.is_train:
            msg = self.describe_network(self.netD)
            print(msg)

    # ----------------------------------------
    # print params
    # ----------------------------------------
    def print_params(self):
        msg = self.describe_params(self.netG)
        print(msg)

    # ----------------------------------------
    # network information
    # ----------------------------------------
    def info_network(self):
        msg = self.describe_network(self.netG)
        if self.is_train:
            msg += self.describe_network(self.netD)
        return msg

    # ----------------------------------------
    # params information
    # ----------------------------------------
    def info_params(self):
        msg = self.describe_params(self.netG)
        if self.is_train:
            msg += self.describe_params(self.netD)
        return msg
