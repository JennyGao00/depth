# -*- coding: utf-8 -*-
"""
 @Time    : 2021/1/28
 @Author  : Jinyan Gao
 @Email   : gaojinyan@nuaa.edu.cn
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import time
import socket
from datetime import datetime
import os
from tensorboardX import SummaryWriter
from torch.optim import lr_scheduler
import shutil

from dataloader import nyudv2_dataloader as nyudata
# from model import DORNnet
from metrics import Result, AverageMeter
# from utils.utils import *
import utils.utils
import criteria
from model.tiny_unet import UNet

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser(description='pytorch depth training')
parser.add_argument('--data_root', default='/media/gao/Gao106/NYUV2/data/nyu2_test/', help='path to load nuydv2')
# parser.add_argument('--list_root', default=ROOT_DIR + '/data', help='path of trainlist or testlist')
parser.add_argument('--train_phase', type=str, default='train', help='train phase')
parser.add_argument('--val_phase', type=str, default='val', help='test phase')
parser.add_argument('--epoch', default=20, type=int, help='training epoch')
parser.add_argument('--batch_size', type=int, default=2, help='batch size')
parser.add_argument('--output_dir', type=str, default=ROOT_DIR + '/result', help='output dir')
parser.add_argument('--lr', default=0.000001, help='learning rate')
parser.add_argument('--lr_patience', default=3, type=int, help='Patience of LR scheduler')
parser.add_argument('--epsilon', default=0.001, type=float, help='epsilon')
parser.add_argument('--optimizer_name', default="adam", type=str, help="Optimizer selection")
parser.add_argument('--weight_decay', default=1e-5, type=float, help='weight decay')
parser.add_argument('--momentum', default=0.9, help='momentum')
parser.add_argument('--do_summary', action='store_true', default=True, help='whether do summary or not')
opt = parser.parse_args()
print(opt)

best_result = Result()
best_result.set_to_worst()

def train(epoch, trainData, model, crite, optimizer, logger):
    average_meter = AverageMeter()
    model.train()  # switch to train mode
    end = time.time()

    for i, (image, depth) in enumerate(trainData):
        image = image.cuda()
        depth = depth.cuda()
        # normal = normal.cuda()
        # image = torch.autograd.Variable(image)
        # depth = torch.autograd.Variable(depth)
        torch.cuda.synchronize()
        data_time = time.time() - end

        end = time.time()
        optimizer.zero_grad()
        pred = model(image)
        loss = crite(pred, depth)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        gpu_time = time.time() - end

        result = Result()
        result.evaluate(pred.data, depth.data)
        average_meter.update(result, gpu_time, data_time, image.size(0))
        end = time.time()

        if (i + 1) % 10 == 0:
            print('=> output: {}'.format(opt.output_dir))
            print('Train Epoch: {0} [{1}/{2}]\t'
                  't_Data={data_time:.3f}({average.data_time:.3f}) '
                  't_GPU={gpu_time:.3f}({average.gpu_time:.3f})\n\t'
                  'Loss={Loss:.5f} '
                  'RMSE={result.rmse:.2f}({average.rmse:.2f}) '
                  'RML={result.absrel:.2f}({average.absrel:.2f}) '
                  'Log10={result.lg10:.3f}({average.lg10:.3f}) '
                  'Delta1={result.delta1:.3f}({average.delta1:.3f}) '
                  'Delta2={result.delta2:.3f}({average.delta2:.3f}) '
                  'Delta3={result.delta3:.3f}({average.delta3:.3f})'.format(
                epoch, i + 1, len(trainData), data_time=data_time,
                gpu_time=gpu_time, Loss=loss.item(), result=result, average=average_meter.average()))
            current_step = epoch * len(trainData) + i
            logger.add_scalar('Train/loss', loss, current_step)
            logger.add_scalar('Train/RMSE', result.rmse, current_step)
            logger.add_scalar('Train/rml', result.absrel, current_step)
            logger.add_scalar('Train/Log10', result.lg10, current_step)
            logger.add_scalar('Train/Delta1', result.delta1, current_step)
            logger.add_scalar('Train/Delta2', result.delta2, current_step)
            logger.add_scalar('Train/Delta3', result.delta3, current_step)



def validate(epoch, valData, model, logger):
    average_meter = AverageMeter()
    model.eval()  # switch to evaluate mode
    end = time.time()
    # skip = len(valData) // 9  # save images every skip iters

    for i, (image, depth) in enumerate(valData):
        image = image.cuda()
        depth = depth.cuda()
        # normal = normal.cuda()
        torch.cuda.synchronize()
        data_time = time.time() - end

        end = time.time()
        with torch.no_grad():
            pred = model(image)

        torch.cuda.synchronize()
        gpu_time = time.time() - end

        result = Result()
        result.evaluate(pred.data, depth.data)
        average_meter.update(result, gpu_time, data_time, image.size(0))
        end = time.time()

        if (i + 1) % 10 == 0:
            print('Test Epoch: [{0}/{1}]\t'
                  't_Data={data_time:.3f}({average.data_time:.3f}) '
                  't_GPU={gpu_time:.3f}({average.gpu_time:.3f})\n\t'
                  'RMSE={result.rmse:.2f}({average.rmse:.2f}) '
                  'RML={result.absrel:.2f}({average.absrel:.2f}) '
                  'Log10={result.lg10:.3f}({average.lg10:.3f}) '
                  'Delta1={result.delta1:.3f}({average.delta1:.3f}) '
                  'Delta2={result.delta2:.3f}({average.delta2:.3f}) '
                  'Delta3={result.delta3:.3f}({average.delta3:.3f})'.format(
                i + 1, len(valData), data_time=data_time,
                gpu_time=gpu_time, result=result, average=average_meter.average()))


        avg = average_meter.average()

        print('\n*\n'
              'RMSE={average.rmse:.3f}\n'
              'Rel={average.absrel:.3f}\n'
              'Log10={average.lg10:.3f}\n'
              'Delta1={average.delta1:.3f}\n'
              'Delta2={average.delta2:.3f}\n'
              'Delta3={average.delta3:.3f}\n'
              't_GPU={time:.3f}\n'.format(
            average=avg, time=avg.gpu_time))

        logger.add_scalar('Test/rmse', avg.rmse, epoch)
        logger.add_scalar('Test/Rel', avg.absrel, epoch)
        logger.add_scalar('Test/log10', avg.lg10, epoch)
        logger.add_scalar('Test/Delta1', avg.delta1, epoch)
        logger.add_scalar('Test/Delta2', avg.delta2, epoch)
        logger.add_scalar('Test/Delta3', avg.delta3, epoch)
        return avg

def main():
    global opt, best_result
    trainData = nyudata.getTrainingData_NYUDV2(opt.batch_size, opt.train_phase, opt.data_root)
    valData = nyudata.getTestingData_NYUDV2(opt.batch_size,  opt.val_phase, opt.data_root)
    print("load data finished!")

    print('create the model')
    # 定义损失模型
    model = UNet()
    optimizer = utils.utils.build_optimizer(model=model,
                                learning_rate=opt.lr,
                                optimizer_name=opt.optimizer_name,
                                weight_decay=opt.weight_decay,
                                epsilon=opt.epsilon,
                                momentum=opt.momentum
                                )
    model = model.cuda()
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=opt.lr_patience)
    # 定义损失函数
    crite = criteria.MaskedL1Loss()

    # create directory path
    output_directory = utils.utils.get_output_dir(opt.output_dir)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    best_txt = os.path.join(output_directory, 'best.txt')
    config_txt = os.path.join(output_directory, 'config.txt')

    # write training parameters to config file
    if not os.path.exists(config_txt):
        with open(config_txt, 'w') as txtfile:
            args_ = vars(opt)
            args_str = ''
            for k, v in args_.items():
                args_str = args_str + str(k) + ':' + str(v) + ',\t\n'
            txtfile.write(args_str)

    # create log
    log_path = os.path.join(output_directory, 'logs',
                            datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
    if os.path.isdir(log_path):
        shutil.rmtree(log_path)
    os.makedirs(log_path)
    logger = SummaryWriter(log_path)

    for epoch in range(0, opt.epoch):
        train(epoch, trainData, model, crite, optimizer, logger)
        result = validate(epoch, valData, model, logger)

        # remember best rmse and save checkpoint
        is_best = result.rmse < best_result.rmse
        if is_best:
            best_result = result
            with open(best_txt, 'w') as txtfile:
                txtfile.write(
                    "epoch={}, rmse={:.3f}, rml={:.3f}, log10={:.3f}, d1={:.3f}, d2={:.3f}, dd31={:.3f}, "
                    "t_gpu={:.4f}".
                        format(epoch, result.rmse, result.absrel, result.lg10, result.delta1, result.delta2,
                               result.delta3,
                               result.gpu_time))


        # save checkpoint for each epoch
        utils.utils.save_checkpoint({
            'args': opt,
            'epoch': epoch,
            'model': model,
            'best_result': best_result,
            'optimizer': optimizer,
        }, is_best, epoch, output_directory)

        # when rml doesn't fall, reduce learning rate
        scheduler.step(result.absrel)

    logger.close()


if __name__ == '__main__':
    main()