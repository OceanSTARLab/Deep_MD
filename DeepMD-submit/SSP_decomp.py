#!/usr/bin/env python

'''
    Deep decomposition for SSP vs. SVD for SSP
'''

# import os
# import glob
# from collections import defaultdict
# from scipy import linalg
# import sio as sio
# import importlib
# import cv2
# from scipy import io
# from skimage.metrics import structural_similarity as ssim_func

import sys
import tqdm
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from numpy import matmul
import scipy.io as sio
from sklearn.metrics import mean_squared_error
from math import sqrt
import time
from scipy.io import savemat
from scipy.linalg import pinv
from modules.utils import mixgausnoise
from modules import models, utils, losses, deep_prior

sys.path.append('E:/Deep tensor/DeepTensor-main/modules')



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':

    # Step 1a: Set simulation constants

    datatemp = sio.loadmat('SSPdata/ssp_mat_60.mat')
    ssp_mat_ori = datatemp['ssp_mat']
    train_begin = 0
    train_size = 256
    test_size = 256
    ssp_train = ssp_mat_ori[:, train_begin:train_begin + train_size]
    ssp_test = ssp_mat_ori[:, train_begin + train_size:train_begin + train_size + test_size]

    train_mean = np.mean(ssp_train, axis=1, keepdims=True)
    test_mean = np.mean(ssp_test, axis=1, keepdims=True)
    ssp_train_demean = ssp_train - train_mean
    ssp_test_demean = ssp_test - test_mean
    test_gt = ssp_test_demean
    mat_gt = ssp_train_demean
    # train_mean = np.mean(ssp_train, axis=1, keepdims=True)
    # ssp_train_demean = ssp_train - train_mean
    nrows = ssp_train_demean.shape[0]  # Size of the matrix
    ncols = ssp_train_demean.shape[1]
    rank = 15  # Rank

    noise_type = 'gaussian'  # Type of noise
    noise_snr = 1.0  # Std. dev for gaussian noise
    singnal_snr = 10 * np.log10(np.var(ssp_train) / (np.square(noise_snr) + 1e-6))
    tau = 1000  # Max. lambda for photon noise

    # Step 1b:Set network parameters
    n_inputs = rank
    init_nconv = 32
    num_channels_up = 3 # The number of upsampling/downsampling times

    sched_args = argparse.Namespace()  # 一个包含字符串表示的object子类

    # Step 1c: Learning constants

    scheduler_type = 'none'
    learning_rate = 1e-4
    epochs = 1500
    sched_args.step_size = 2000
    sched_args.gamma = pow(10, -1 / epochs)
    sched_args.max_lr = learning_rate
    sched_args.min_lr = 1e-6
    sched_args.epochs = epochs

    # Step 2a: Generate data

    mat_gt = ssp_train_demean
    mat_gt_ten = torch.tensor(mat_gt)[None, ...].to(device)
    # Ground Truth
    mat = ssp_train_demean + noise_snr * np.random.randn(ssp_train_demean.shape[0], ssp_train_demean.shape[1])
    # mat = ssp_train_demean + mixgausnoise(ssp_train_demean, 0.4, 4, 0.01)
    # Move them to device
    mat_ten = torch.tensor(mat)[None, ...].to(device)

    test_noise = ssp_test_demean + noise_snr * np.random.randn(ssp_test_demean.shape[0], ssp_test_demean.shape[1])
    # test_noise = ssp_test_demean + mixgausnoise(ssp_test_demean, 0.4, 4, 0.01)

    # Step 2b: Initialization of decomposition matrix

    u_inp = utils.get_inp([1, n_inputs, nrows])
    v_inp = utils.get_inp([1, n_inputs, ncols])

    # Step 2c: Create networks


    u_net = deep_prior.get_net(n_inputs, 'skip1d', 'reflection',
                               # Number of input layer channels; Neural network type; whether to add zeros or reflect parameters
                                   upsample_mode='linear',  # Type of upsampling
                                   skip_n33d=init_nconv,  # d-down, the number of channels during the downsampling process
                                   skip_n33u=init_nconv,  # u-up, the number of channels during upsampling
                                   num_scales=num_channels_up,  # The number of convolutional layers
                                   n_channels=rank).to(device)  # The number of columns in a factor matrix
    v_net = deep_prior.get_net(n_inputs, 'skip1d', 'reflection',
                                   upsample_mode='linear',
                                   skip_n33d=init_nconv,
                                   skip_n33u=init_nconv,
                                   num_scales=num_channels_up,
                                   n_channels=rank).to(device)


    # Step 2d: Optimize settings
    # 1. Extract training parameters
    net_params = list(u_net.parameters()) + list(v_net.parameters())
    inp_params = [u_inp] + [v_inp]

    # 2. You can either optimize both net and inputs, or just net

    # params = net_params + inp_params
    params = net_params
    optimizer = torch.optim.Adam(lr=learning_rate, params=params)

    # 3. Create a learning scheduler
    scheduler = utils.get_scheduler(scheduler_type, optimizer, sched_args)

    # 4. Create loss functions -- loses.L1Norm() or losses.L2Norm()
    criterion = torch.nn.MSELoss()
    best_mat = np.zeros(mat_gt.shape)
    rmse_list = []  #
    total_params = sum(p.numel() for p in u_net.parameters())

    # total_params = sum(p.numel() for p in v_net.parameters())


    # Step 3: Iterations
    best_mse = float('inf')
    tbar = tqdm.tqdm(range(epochs))  #
    start = time.time()
    rank_all = []
    for idx in tbar:
        u_output = u_net(u_inp).permute(0, 2, 1)
        v_output = v_net(v_inp)

        mat_estim = torch.bmm(u_output, v_output)


        loss = criterion(mat_estim.float(), mat_ten.float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        mat_cpu = mat_estim.squeeze().detach().cpu().numpy()
        rmse_list.append(np.sqrt(((mat_estim - mat_gt_ten) ** 2).mean().item()))
        tbar.set_description('%.4e' % rmse_list[idx])
        tbar.refresh()
        Urank, Srank, Vrank = np.linalg.svd(mat_estim.detach().numpy().reshape(60, 256))
        total = np.sum(Srank)
        cumsum = np.cumsum(Srank)
        rankn = np.searchsorted(cumsum, 0.95 * total) + 1
        # if idx % 10 == 0:
        rank_all.append(rankn)
        # tbar.set_description('%.1e' % rank_all[idx])

        # if mean_squared_error(mat_gt, mat_estim.detach().numpy().reshape(60, 256)) < best_mse:
        # best_epoch = idx
        best_mat = mat_cpu
        best_mse = mean_squared_error(mat_gt, mat_estim.detach().numpy().reshape(60, 256))
        best_Udic = u_output.reshape(60, rank)
        best_V = v_output.reshape(rank, train_size)


    # Step 5: Compute accuracy

    end = time.time()
    time_curr = end - start
    U_deep = best_Udic.detach().numpy()
    V_deep = best_V.detach().numpy()
    print("running time:" + " " + str(time_curr) + "\n")

    psnr1 = utils.psnr(mat_gt, best_mat)
    psnr2 = utils.psnr(mat_gt, utils.lr_decompose(mat, rank))

    rmse1 = sqrt(mean_squared_error(mat_gt, best_mat))
    rmse2 = sqrt(mean_squared_error(mat_gt, utils.lr_decompose(mat, rank)))
    # del(rank_all[0])
    minrank = min(rank_all)
    best_epoc = len(rank_all) - 1 - rank_all[::-1].index(minrank)

    bestrmse = rmse_list[best_epoc]


    print('training rmse of DeepTensor: %.4f' % rmse1)
    print('best training rmse is: %.4f' % bestrmse)
    print('best epoc is : %.4f' % best_epoc)

    # test
    V_estim = matmul(pinv(U_deep), test_noise)
    test_estim = matmul(U_deep, V_estim)
    rmse_test = sqrt(mean_squared_error(test_estim, test_gt))
    print('test rmse of DeepTensor: %.4f' % rmse_test)

    # plot Rank estimate vs epoch and and compare it with the training rmse to verify that
    # the best_epoc selected according to Rank estimate is a good value.
    # In actual use, the model can obtain the best_epoc value without knowing the training rmse.
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)

    ax1.plot(rank_all)
    ax1.scatter(best_epoc, rank_all[best_epoc], c='r', marker='*')
    ax1.set_title('Rank estimate vs Epoch')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Rank')

    ax2.plot(rmse_list)
    ax2.scatter(best_epoc, rmse_list[best_epoc], c='r', marker='*')
    ax2.set_title('RMSE vs Epoch')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('RMSE')

    plt.show()

    # outputs: factor matrices U and V
    savemat('U.mat', {'U': U_deep})
    savemat('V.mat', {'V': V_deep})


