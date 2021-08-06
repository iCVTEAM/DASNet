#!/usr/bin/python3
# coding=utf-8

import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
# from torchvision import transforms as tfs
# import albumentations as A


########################### Data Augmentation ###########################
class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, mask, depth):
        image = (image - self.mean) / self.std
        mask /= 255
        depth /= 255
        return image, mask, depth


class RandomCrop(object):
    def __call__(self, image, mask, depth):
        H, W, _ = image.shape
        randw = np.random.randint(W / 8)
        randh = np.random.randint(H / 8)
        offseth = 0 if randh == 0 else np.random.randint(randh)
        offsetw = 0 if randw == 0 else np.random.randint(randw)
        p0, p1, p2, p3 = offseth, H + offseth - randh, offsetw, W + offsetw - randw
        return image[p0:p1, p2:p3, :], mask[p0:p1, p2:p3], depth[p0:p1, p2:p3]


class RandomFlip(object):
    def __call__(self, image, mask, depth):
        if np.random.randint(2) == 0:
            return image[:, ::-1, :], mask[:, ::-1], depth[:, ::-1]
        else:
            return image, mask, depth


class Resize(object):
    def __init__(self, H, W):
        self.H = H
        self.W = W

    def __call__(self, image, mask, depth):
        image = cv2.resize(image, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        depth = cv2.resize(depth, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        return image, mask, depth


class ToTensor(object):
    def __call__(self, image, mask, depth):
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1)
        mask = torch.from_numpy(mask)
        depth = torch.from_numpy(depth)
        return image, mask, depth


########################### Config File ###########################
class Config(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        # for rgb DUT
        # self.mean   = np.array([[[124.55, 118.90, 102.94]]])
        # self.std    = np.array([[[ 56.77,  55.97,  57.50]]])
        #
        # for rgbd sod
        # self.mean = np.array([[[112.03, 108.56, 100.15]]])
        # self.std = np.array([[[55.86, 53.71, 53.99]]])
        # for rgbd sod 0810
        self.mean = np.array([[[112.77, 109.66, 102.11]]])
        self.std = np.array([[[55.45, 53.60, 53.86]]])
        # for imagenet
        #self.mean = np.array([[[123.67, 116.28, 103.53]]])
        #self.std = np.array([[[58.39, 57.12, 57.37]]])
        print('\nParameters...')
        for k, v in self.kwargs.items():
            print('%-10s: %s' % (k, v))

    def __getattr__(self, name):
        if name in self.kwargs:
            return self.kwargs[name]
        else:
            return None


########################### Dataset Class ###########################
class Data(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.normalize = Normalize(mean=cfg.mean, std=cfg.std)
        self.randomcrop = RandomCrop()
        self.randomflip = RandomFlip()
        self.resize = Resize(352, 352)
        self.totensor = ToTensor()
        with open(cfg.datapath + '/' + cfg.mode + '.txt', 'r') as lines:
            self.samples = []
            for line in lines:
                self.samples.append(line.strip())

    def __getitem__(self, idx):
        name = self.samples[idx]
        image = cv2.imread(self.cfg.datapath + '/image/' + name + '.jpg')[:, :, ::-1].astype(np.float32)
        mask = cv2.imread(self.cfg.datapath + '/mask/' + name + '.png', 0).astype(np.float32)
        depth = cv2.imread(self.cfg.datapath + '/depth/' + name + '.png', 0).astype(np.float32)
        shape = mask.shape

        if self.cfg.mode == 'train':
            image, mask, depth = self.normalize(image, mask, depth)
            image, mask, depth = self.randomcrop(image, mask, depth)
            image, mask, depth = self.randomflip(image, mask, depth)
            return image, mask, depth
        else:
            image, mask, depth = self.normalize(image, mask, depth)
            image, mask, depth = self.resize(image, mask, depth)
            image, mask, depth = self.totensor(image, mask, depth)
            return image, mask, shape, name

    def collate(self, batch):
        size = [224, 256, 288, 320, 352][np.random.randint(0, 5)]
        image, mask, depth = [list(item) for item in zip(*batch)]
        for i in range(len(batch)):
            image[i] = cv2.resize(image[i], dsize=(size, size), interpolation=cv2.INTER_LINEAR)
            mask[i] = cv2.resize(mask[i], dsize=(size, size), interpolation=cv2.INTER_LINEAR)
            depth[i] = cv2.resize(depth[i], dsize=(size, size), interpolation=cv2.INTER_LINEAR)
        image = torch.from_numpy(np.stack(image, axis=0)).permute(0, 3, 1, 2)
        mask = torch.from_numpy(np.stack(mask, axis=0)).unsqueeze(1)
        depth = torch.from_numpy(np.stack(depth, axis=0)).unsqueeze(1)
        return image, mask, depth

    def __len__(self):
        return len(self.samples)


########################### Testing Script ###########################
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    plt.ion()

    cfg = Config(mode='train', datapath='../data/NJU2K')
    data = Data(cfg)
    for i in range(1000):
        image, mask = data[i]
        image = image * cfg.std + cfg.mean
        plt.subplot(121)
        plt.imshow(np.uint8(image))
        plt.subplot(122)
        plt.imshow(mask)
        input()
