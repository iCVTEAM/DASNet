#!/usr/bin/python3
# coding=utf-8

import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.insert(0, '../')
sys.dont_write_bytecode = True

import cv2
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import dataset as dataset
from net_MM import DASNet

import time


class Test(object):
    def __init__(self, Dataset, Network, path):
        ## dataset
        self.cfg = Dataset.Config(datapath=path, snapshot='.{MODEL_PATH}', mode='test')
        self.data = Dataset.Data(self.cfg)
        self.loader = DataLoader(self.data, batch_size=1, shuffle=False, num_workers=8)
        ## network
        device = torch.device('cuda')
        self.net = Network(self.cfg)
        self.net.train(False)
        self.net = nn.DataParallel(self.net)
        self.net.to(device)


if __name__ == '__main__':
    for path in ['../data/NJU2K', '../data/NLPR', '../data/STERE', '../data/SIP', '../data/LFSD']:
    # for path in ['../data/SSD', '../data/DES']:
        print(path)
        test = Test(dataset, DASNet, path)

        times = []
        with torch.no_grad():
            for image, mask, shape, name in test.loader:
                image = image.cuda().float()

                torch.cuda.synchronize()
                start = time.time()
                pred, out2h, out3h, out4h, out5h, dpred= test.net(image, shape)
                torch.cuda.synchronize()
                end = time.time()
                times.append(end-start)
                pred = (torch.sigmoid(pred[0, 0]) * 255).cpu().numpy()
                head = '../eval/maps/DASNet/' + test.cfg.datapath.split('/')[-1]
                if not os.path.exists(head):
                    os.makedirs(head)
                cv2.imwrite(head + '/' + name[0] + '.png', np.round(pred))

        # FPS
        time_sum = 0
        for i in times:
            time_sum += i
        print("FPS: %f" % (1.0 / (time_sum / len(times))))
