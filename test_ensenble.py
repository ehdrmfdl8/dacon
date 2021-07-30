import os.path
import logging
import re

import numpy as np
from collections import OrderedDict

import torch

from utils import utils_logger
from utils import utils_image as util
from utils import utils_model

def main():

    # ----------------------------------------
    # Preparation
    # ----------------------------------------
    model_type = 'urrdbx2_grid_shuffle_2'
    model_name_1 = '249000_G'        # 'rrdb_x4_esrgan' | 'rrdb_x4_psnr'
    model_name_2 = '251000_G'        # 'rrdb_x4_esrgan' | 'rrdb_x4_psnr'

    testset_name = 'test_input_img'       # test set,  'val' | 'test_input_img'
    need_degradation = False              # default: True
    x8 = False                           # default: False, x8 to boost performance
    sf = 1 # [int(s) for s in re.findall(r'\d+', model_name)][0]  # scale factor
    show_img = False                     # default: False




    task_current = 'sr'       # 'dn' for denoising | 'sr' for super-resolution
    n_channels = 3            # fixed

    model_pool = 'urrdb_psnrx2/urrdbx2_grid_shuffle_2/models/lr=5.0xe-5_RandomGridShuffle'
    input_sets = 'testsets'                                  # 'testsets' | 'trainsets/train_input_img/case3'
    label_sets = 'trainsets/train_label_img/case3'           # None | 'trainsets/train_label_img/case3'
    results = 'results'         # 'val' | 'results'
    result_name = model_type + '_' + model_name_1 + '+' + model_name_2
    border = sf if task_current == 'sr' else 0     # shave boader to calculate PSNR and SSIM
    model_1_path = os.path.join(model_pool, model_name_1+'.pth')
    model_2_path = os.path.join(model_pool, model_name_2+'.pth')

    # ----------------------------------------
    # L_path, E_path, H_path
    # ----------------------------------------

    L_path = os.path.join(input_sets, testset_name) # L_path, for Low-quality images
    H_path = None if testset_name == 'test_input_img' else os.path.join(label_sets, testset_name) # H_path, for High-quality images
    E_path = os.path.join(results, result_name)   # E_path, for Estimated images
    util.mkdir(E_path)

    if H_path == L_path:
        need_degradation = True
    logger_name = result_name
    utils_logger.logger_info(logger_name, log_path=os.path.join(E_path, logger_name+'.log'))
    logger = logging.getLogger(logger_name)

    need_H = True if H_path is not None else False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # FP32 -> FP16
    scaler = torch.cuda.amp.GradScaler()

    # ----------------------------------------
    # load model
    # ----------------------------------------

    from models.network_u_rrdbx2 import URRDBNetx2 as net
    model_1 = net(in_nc=n_channels, out_nc=n_channels, nc=32, nb=20, gc=32, act_mode='L', upsample_mode='convtranspose', downsample_mode='strideconv')
    model_2 = net(in_nc=n_channels, out_nc=n_channels, nc=32, nb=20, gc=32, act_mode='L', upsample_mode='convtranspose', downsample_mode='strideconv')

    model_1.load_state_dict(torch.load(model_1_path), strict=True)  # strict=False
    model_2.load_state_dict(torch.load(model_2_path), strict=True)  # strict=False

    model_1.eval()
    model_2.eval()

    for (k1, v1), (k2, v2) in zip(model_1.named_parameters(), model_2.named_parameters()):
        v1.requires_grad = False
        v2.requires_grad = False

    model_1 = model_1.to(device)
    model_2 = model_2.to(device)

    logger.info('Model path: {:s}'.format(model_1_path))
    logger.info('Model path: {:s}'.format(model_2_path))

    number_parameters_1 = sum(map(lambda x: x.numel(), model_1.parameters()))
    logger.info('Params number: {}'.format(number_parameters_1))

    number_parameters_2 = sum(map(lambda x: x.numel(), model_2.parameters()))
    logger.info('Params number: {}'.format(number_parameters_2))

    test_results = OrderedDict()
    test_results['psnr'] = []

    logger.info('model_name:{}'.format(model_name_1))
    logger.info('model_name:{}'.format(model_name_2))
    logger.info(L_path)
    L_paths = util.get_image_paths(L_path)
    H_paths = util.get_image_paths(H_path) if need_H else None

    result_img = []
    for idx, img in enumerate(L_paths):

        # ------------------------------------
        # (1) img_L
        # ------------------------------------

        img_name, ext = os.path.splitext(os.path.basename(img))
        # logger.info('{:->4d}--> {:>10s}'.format(idx+1, img_name+ext))
        img_L = util.imread_uint(img, n_channels=n_channels)
        img_L = util.uint2tensor4(img_L)
        img_L = img_L.to(device)

        # ------------------------------------
        # (2) img_E
        # ------------------------------------
        if scaler is not None:
            with torch.cuda.amp.autocast():
                img_E1 = model_1(img_L)
                img_E2 = model_2(img_L)
        else:
            img_E1 = model_1(img_L)
            img_E2 = model_2(img_L)

        img_E1 = util.tensor2uint(img_E1)
        img_E2 = util.tensor2uint(img_E2)

        # ensemble
        img_E = 0.5 * img_E1 + 0.5 * img_E2

        if need_H:

            # --------------------------------
            # (3) img_H
            # --------------------------------

            img_H = util.imread_uint(H_paths[idx], n_channels=n_channels)

            # --------------------------------
            # PSNR and SSIM
            # --------------------------------

            psnr = util.calculate_psnr(img_E, img_H, border=border)
            test_results['psnr'].append(psnr)

            logger.info('{:s} - PSNR: {:.2f} dB;'.format(img_name+ext, psnr))

        # ------------------------------------
        # save results
        # ------------------------------------

        util.imsave(img_E, os.path.join(E_path, img_name+'.png'))
        result_img.append(img_E)
    if need_H:
        ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
        logger.info('Average PSNR(RGB) - {} - --PSNR: {:.2f} dB; '.format(result_name, ave_psnr))

if __name__ == '__main__':

    main()