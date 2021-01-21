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
import nyudv2_dataloader as nyudata
import model.DORNnet
from loss import *
import metrics
from metric_depth_model import *

parser = argparse.ArgumentParser(description='pytorch depth training')
parser.add_argument('--dataroot', default='/media/gao/Gao106', help='path to load nuydv2')
parser.add_argument('--trainlist_path', default='/media/gao/Gao106/data/nyu2_train.csv', help='the path of train.csv file ')
parser.add_argument('--testlist_path', default='/media/gao/Gao106/data/nyu2_test.csv', help='the path of test.csv file ')
parser.add_argument('--phase', type=str, default='train', help='train or test phase')
parser.add_argument('--epoch', default=5, type=int, help='training epoch')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--output_dir', type=str, default='/result', help='output dir')
parser.add_argument('--lr', default=0.001, help='learning rate')
parser.add_argument('--momentum', default=0.9, help='momentum')
opt = parser.parse_args()


def train(epochs, trainData, model, model_loss, optimizer, logger):
    for epoch in range(epochs):
        for i, sample in enumerate(len(trainData)):
            image, depth = sample['image'], sample['depth']
            image = image.cuda()
            depth = depth.cuda()
            image = torch.autograd.Variable(image)
            depth = torch.autograd.Variable(depth)
            optimizer.zero_grad()
            pred = model(image)
            loss = total_loss(pred, depth)
            model_loss.update(loss.item)



    print(1)


if __name__ == '__main__':
    trainData = nyudata.getTrainingData_NYUDV2(opt.batch_size, opt.trainlist_path, opt.dataroot)
    # trainLoader = torch.utils.data.DataLoader(dataset=trainData, batch_size=opt.batch_size, shuffle=True)
    print("loaddata finished!")

    print('create the model')
    # 定义损失模型
    model = model.DORNnet.ResNet()
    optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum)
    model = model.cuda()
    # 定义损失函数
    model_loss = ModelLoss()
    # 初始化輸出文件
    log_path = os.path.join(opt.output_dir, 'logs', datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
    os.makedirs(log_path)
    logger = SummaryWriter(log_path)

    train(opt.epoch, trainData, model, model_loss, optimizer, logger)






