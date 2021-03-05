import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from PIL import Image
import torch
import os

def png_reader_32bit(depth_path, img_size=(0,0)):
    # 16-bit png will be read in as a 32-bit img
    depth = Image.open(depth_path)

    pixel = np.array(depth)
    pixel = pixel.astype(float)
    pixel = pixel / 1000
    pixel = np.log(pixel)

    if img_size[0]: #nearest interpolation
        step = pixel.shape[0]//img_size[0]
        pixel = pixel[0::step, :]
        pixel = pixel[:, 0::step]

    depth = pixel[np.newaxis, :, :]
    depth = torch.from_numpy(depth)

    return depth

def png_reader_uint8(path, img_size=(0,0)):
    image = Image.open(path)
    pixel = np.array(image, dtype=np.uint8)
    # if img_size[0]:
        # pixel = m.imresize(pixel, (img_size[0], img_size[1]))#only works for 8 bit image
        # pixel = transform.resize(pixel, (img_size[0], img_size[1]))
    if img_size[0]: #nearest interpolation
        step = pixel.shape[0]//img_size[0]
        pixel = pixel[0::step, :]
        pixel = pixel[:, 0::step]


    return pixel


class NYUDV2Dataset(Dataset):
    """NYUV2D dataset."""

    def __init__(self, phase, dataroot, transform):
        # self.listroot = listroot
        # self.phase = phase
        # self.dataroot = dataroot
        # self.transform = transform
        # self.data_size = (240, 320)
        # self.data = [line.rstrip() for line in open(os.path.join(listroot, (phase + '.txt')))]
        # print(self.data)

        self.dataroot = dataroot
        self.transform = transform
        self.data_size = (240, 320)
        self.img_path = self.dataroot + 'colors'
        self.image_files = [d for d in os.listdir(self.img_path)]
        self.depth_path = self.dataroot + 'depth'
        self.depth_files = [d for d in os.listdir(self.depth_path)]
        if type == "train":
            self.image_files = self.image_files[0:458]
            self.depth_files = self.depth_files[0:458]
        elif type == "val":
            self.image_files = self.image_files[458:]
            self.depth_files = self.depth_files[458:]

    def __getitem__(self, idx):
        # base_path = str(self.dataroot) + str(self.phase) + '/' + self.data[idx]
        # depth_np = png_reader_32bit(base_path, self.data_size)
        # depth_np = depth_np.astype(float)
        # depth_np = depth_np / 10000
        # depth = depth_np[np.newaxis, :, :]
        # depth_torch = torch.from_numpy(depth).float()
        #
        # # depth_mask = (depth_np > 0.0001).astype(float)
        # # depth_mask = torch.from_numpy(depth_mask).float()
        #
        # rgb_path = base_path.replace('depth', 'colors')
        # img = png_reader_uint8(rgb_path, self.data_size)
        # img = img.astype(float)
        # img = img / 255
        # img = img.transpose(2, 0, 1)
        # img_torch = torch.from_numpy(img).float()

        # normal_path = base_path.replace('depth', 'normal')
        # normal = png_reader_uint8(normal_path, self.data_size)
        # normal = normal.astype(float)
        # normal = normal / 255
        # normal = normal.transpose(2, 0, 1)
        # normal_torch = torch.from_numpy(normal).float()

        depth_path = self.depth_files[idx]
        image_path = depth_path.replace('depth', 'colors')
        depth_path = self.dataroot + 'depth/' + depth_path
        image_path = self.dataroot + 'colors/' + image_path

        # img = png_reader_uint8(image_path, self.data_size)
        # img = img.astype(float)
        # img = img / 255
        # img = img.transpose(2, 0, 1)
        # img_torch = torch.from_numpy(img).float()
        #
        # depth_np = png_reader_32bit(depth_path, self.data_size)
        # depth_np = depth_np.astype(float)
        # depth_np = depth_np / 10000
        # depth = depth_np[np.newaxis, :, :]
        # depth_torch = torch.from_numpy(depth).float()

        image = Image.open(image_path)
        depth = png_reader_32bit(depth_path, (240, 320))

        if self.transform:
            image = self.transform(image)

        return image, depth

    def __len__(self):
        return len(self.image_files)


def getTrainingData_NYUDV2(batch_size, phase, dataroot):
    __imagenet_pca = {
        'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
        'eigvec': torch.Tensor([
            [-0.5675, 0.7192, 0.4009],
            [-0.5808, -0.0045, -0.8140],
            [-0.5836, -0.6948, 0.4203],
        ])
    }
    __imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]}

    transformed_training = NYUDV2Dataset(phase=phase, dataroot=dataroot,
                                         transform=transforms.Compose([
                                             transforms.Resize((240, 320)),
                                             # transforms.RandomHorizontalFlip(),

                                             transforms.ToTensor(),
                                             # transforms.Lighting(0.1, __imagenet_pca[
                                             #     'eigval'], __imagenet_pca['eigvec']),
                                             # transforms.ColorJitter(
                                             #     brightness=0.4,
                                             #     contrast=0.4,
                                             #     saturation=0.4,
                                             # ),
                                             transforms.Normalize(__imagenet_stats['mean'],
                                                       __imagenet_stats['std'])
                                         ]))

    dataloader_training = DataLoader(transformed_training, batch_size,
                                     shuffle=True, num_workers=1, pin_memory=False)

    return dataloader_training


def getTestingData_NYUDV2(batch_size, phase, dataroot):
    __imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]}

    transformed_testing = NYUDV2Dataset(phase=phase, dataroot=dataroot,
                                        transform=transforms.Compose([
                                            transforms.Resize((240, 320)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(__imagenet_stats['mean'],
                                                      __imagenet_stats['std'])
                                        ]))

    dataloader_testing = DataLoader(transformed_testing, batch_size,
                                    shuffle=False, num_workers=4, pin_memory=False)

    return dataloader_testing

