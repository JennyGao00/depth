import torch
import torch.nn as nn
import numpy as np
from plyfile import PlyData,PlyElement
import os
from PIL import Image
import matplotlib.pyplot as plt
import dataloaders.data_transform as transforms


class Visualization_XYZ(nn.Module):
    """
    depth -> point cloud. depth visualization Function.
    """
    def __init__(self, focal_x, focal_y, input_size):
        super(Visualization_XYZ, self).__init__()
        self.fx = torch.tensor([focal_x], dtype=torch.float32).cuda()
        self.fy = torch.tensor([focal_y], dtype=torch.float32).cuda()
        self.input_size = input_size
        self.u0 = torch.tensor(input_size[1] // 2, dtype=torch.float32).cuda()
        self.v0 = torch.tensor(input_size[0] // 2, dtype=torch.float32).cuda()
        self.init_image_coor()
        self.pred_path = 'ply/pred.ply'
        self.gt_path = 'ply/gt.ply'

    def init_image_coor(self):
        x_row = np.arange(0, self.input_size[1])
        x = np.tile(x_row, (self.input_size[0], 1))
        x = x[np.newaxis, :, :]
        x = x.astype(np.float32)
        x = torch.from_numpy(x.copy()).cuda()
        self.u_u0 = x - self.u0

        y_col = np.arange(0, self.input_size[0])  # y_col = np.arange(0, height)
        y = np.tile(y_col, (self.input_size[1], 1)).T
        y = y[np.newaxis, :, :]
        y = y.astype(np.float32)
        y = torch.from_numpy(y.copy()).cuda()
        self.v_v0 = y - self.v0

    def transfer_xyz(self, depth):
        depth = torch.squeeze(depth, 1)
        x = self.u_u0 * torch.abs(depth) / self.fx
        y = self.v_v0 * torch.abs(depth) / self.fy
        z = depth
        pw = torch.cat([x, y, z], 0).permute(1, 2, 0) # [b, h, w, c]
        pw = torch.reshape(pw, (-1, 3))
        print(pw.shape)
        pw_np = pw.cpu().numpy()
        return pw_np

    def write_pred_ply(self, points, text=True):
        """
        save_path : path to save: '/yy/XX.ply'
        pt: point_cloud: size (N,3)
        """
        points = [(points[i, 0], points[i, 1], points[i, 2]) for i in range(points.shape[0])]
        vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
        el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
        PlyData([el], text=text).write(self.pred_path)

    def write_gt_ply(self, points, text=True):
        """
        save_path : path to save: '/yy/XX.ply'
        pt: point_cloud: size (N,3)
        """
        points = [(points[i, 0], points[i, 1], points[i, 2]) for i in range(points.shape[0])]
        vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
        el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
        PlyData([el], text=text).write(self.gt_path)


    def forward(self, pred_depth, gt_depth):
        # depth -> xyz -> ply
        # pred_xyz = self.transfer_xyz(pred_depth)
        # self.write_pred_ply(pred_xyz)
        # print('pred.ply finished!')
        # gt_xyz = self.transfer_xyz(gt_depth)
        # self.write_gt_ply(gt_xyz)
        # print('gt.ply finished!')
        # imshow gt_depth
        imshow(pred_depth, 'pred')
        imshow(gt_depth, 'gt')



def imshow(img, type):
    # img = img.squeeze()
    # img = img.cpu().numpy()
    # # img = np.transpose(img, (1, 2, 0))
    # plt.imshow(img)
    # plt.show()
    img = img.cpu().clone()
    img = img.squeeze(0)
    img = transforms.ToPILImage()(img).convert('L')
    img.save(type+'.jpg')
    plt.imshow(img, plt.cm.gray)
    plt.show()


if __name__ == '__main__':
    # import cv2
    # print(os.getcwd())
    xyz = Visualization_XYZ(1.0, 1.0, (480, 640))
    pred_depth = np.random.randn(1, 1, 480, 640)
    pred_depth = torch.tensor(np.asarray(pred_depth, np.float32)).cuda()
    gt_depth = np.random.randn(1, 1, 480, 640)
    gt_depth = torch.tensor(np.asarray(gt_depth, np.float32)).cuda()

    xyz2ply = xyz(pred_depth, gt_depth)



