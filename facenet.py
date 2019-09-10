# --------------------------------------------------------
# SMNet FaceNet
# Licensed under The MIT License [see LICENSE for details]
# Copyright 2019 smarsu. All Rights Reserved.
# --------------------------------------------------------

import os.path as osp
from tqdm import tqdm
import glog
import time
import cv2
import numpy as np
import smnet as sm
from model import vgg16, facenet_head, facenet_loss
from imageshop import blur, flip, noise


class FaceNet(object):
    def __init__(self,
                 batch_size=32,
                 face_size=(96, 96),
                 alpha=0.5,
                 model_root='data/model'):
        self.batch_size = batch_size
        self.face_size = face_size
        self.alpha = alpha
        self.model_root = model_root
        
        sm.reset_default_graph()
        self.sess = sm.Session()
        self._setup()


    def _setup(self):
        h, w = self.face_size

        self.x = sm.Tensor()
        self.mask = sm.Tensor()
        self.feat, ho, wo, co = vgg16(self.x, h, w)
        self.embed = facenet_head(self.feat, ho, wo, co)
        self.loss, self.dis = facenet_loss(self.embed, self.alpha, self.mask)

    
    def augment(self, image):
        """Image augment."""
        image = blur(image)
        image = noise(image)
        image, flip_target = flip(image)

        #assert 0 <= image <= 255
        return image, flip_target


    def _preprocess(self, data, augment=False):
        """
        Args:
            data: str or ndarray. If ndarray, the shape shoule be [h, w, 3]

        Returns:
            image: ndarray. Shape [face_size, face_size, 3], and it is 
                normalized to [-1, 1].
        """
        image = cv2.imread(data) if isinstance(data, str) else data
        if image.shape[:2] != self.face_size:
        #     glog.warning('Resize {} with shape {} to {}'.format(data, 
        #         image.shape[:2], self.face_size))
            image = cv2.resize(image, self.face_size)

        if augment:
            image, flip_target = self.augment(image)
        
        image = image / 127.5 - 1
        assert image.shape[:2] == self.face_size
        return image


    def train(self, epoch, train_datas, lr, image2id=None, testpaths=None):
        """
        Args:
            epoch: int.
            datas: Func. The function which will return shuffled datas. 
                Usage: datas = train_datas(batch_size=32)
            lr: float. Learning rate.
        """
        for step in range(epoch):
            pbar = tqdm(train_datas(self.batch_size))
            # datas shape: [n * 3, image_names]
            # The order of data is [anchor, pos, neg, anchor, pos, neg, ...]
            losses = []
            augment = 1
            for datas in pbar:
                augment = 1 - augment
                t0 = time.time()
                ids = [image2id[osp.split(data)[-1]] for data in datas] \
                    if image2id is not None else None
                datas = [self._preprocess(data, augment) for data in datas]
                t1 = time.time()
                cur_loss, cur_dis = self.sess.forward([self.loss, self.dis], 
                    feed_dict={self.x: datas, 
                               self.mask: np.full([self.batch_size, 1], -self.alpha)})
                self.sess.optimize([self.loss], lr=lr)
                t2 = time.time()
                losses.append(np.sum(np.mean(cur_dis[cur_dis > -self.alpha], 0)) + self.alpha)
                avg_loss = np.mean(losses)
                t3 = time.time()

                times = (t1 - t0, t2 - t1, t3 - t2)
                times = [round(time * 1000000) for time in times]

                if testpaths:
                    datas = [self._preprocess(data) for data in testpaths]
                    cur_loss, cur_dis = self.sess.forward([self.loss, self.dis], 
                        feed_dict={self.x: datas, 
                                   self.mask: np.full([1, 1], -self.alpha)})

                pbar.set_description('step: {}, loss: {}, ids: {}, time: {}, test loss: {}'.format(
                    step, avg_loss, ids[:3], times, cur_dis + self.alpha))
            
            self.sess.save('/'.join([self.model_root, '{}_{}_{}'.format(str(avg_loss), step, lr)]))


    def test(self, data):
        """
        Args:
            data: str or ndarray.
        """
        data = [self._preprocess(data)] * 3
        feat = self.sess.forward([self.embed], 
                                 feed_dict={self.x: data,
                                 self.mask: np.zeros([1, 1])})
        return feat[0][0]
