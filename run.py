# --------------------------------------------------------
# SMNet FaceNet
# Licensed under The MIT License [see LICENSE for details]
# Copyright 2019 smarsu. All Rights Reserved.
# --------------------------------------------------------

import sys
import os.path as osp
import numpy as np
from datasets import CelebA
from facenet import FaceNet
from euclidean import euclidean_distance

#np.random.seed(196)
dataset_root = '/datasets/CelebA'

if __name__ == '__main__':
    parse = sys.argv[1]  # 'train'
    print(parse)
    face_size = (96, 96)

    if parse == 'train':
        batch_size = 8
        celebA = CelebA(dataset_root, face_size=face_size)
        facenet = FaceNet(batch_size=batch_size, face_size=face_size, alpha=0.5)
        # Restore from:
        #   1. 0.2943756_7_8.npz test loss: 0
        #   2. 0.2817538_2_0.8.npz test loss: 0
        #   3. 0.28175184_22_0.08.npz test loss: 0
        #   4. 0.2740853_5_0.008.npz  test loss: 0

        #   1. 0.52165955_1_0.8.npz
        facenet.sess.restore(osp.join(facenet.model_root, '0.33327728476725504_94_1.npz'))
        # Train stage:
        #   1. epoch: 100, lr: 1
        #   2. epoch: 100, lr: 0.1
        #   3. epoch: 100, lr: 0.1
        #   4. epoch: 100, lr: 0.1
        #   5. epoch: 100, lr: 0.01
        #   6. epoch: 100, lr: 0.01
        #   6. epoch: 100, lr: 0.001
        #facenet.train(100, celebA.train_datas, 1 * batch_size, celebA.image2identity, testpaths=['data/demo/zjl0.jpg', 'data/demo/zjl1.jpg', 'data/demo/ffy.jpg'])
        #facenet.train(100, celebA.train_datas, 0.1 * batch_size, celebA.image2identity, testpaths=['data/demo/zjl0.jpg', 'data/demo/zjl1.jpg', 'data/demo/ffy.jpg'])
        facenet.train(100, celebA.train_datas, 1, celebA.image2identity, testpaths=['data/demo/ffy.jpg', 'data/demo/ffy1.jpg', 'data/demo/zjl.jpg'])
        #facenet.train(100, celebA.train_datas, 0.01 * batch_size, celebA.image2identity, testpaths=['data/demo/zjl0.jpg', 'data/demo/zjl1.jpg', 'data/demo/ffy.jpg'])
        #facenet.train(100, celebA.train_datas, 0.001 * batch_size, celebA.image2identity, testpaths=['data/demo/zjl0.jpg', 'data/demo/zjl1.jpg', 'data/demo/ffy.jpg'])
    elif parse == 'test':
        celebA = CelebA(dataset_root, face_size=face_size)
        facenet = FaceNet(face_size=face_size)
        # well-done 0.3703060848329446_30_1.npz
        facenet.sess.restore(osp.join(facenet.model_root, 'nan_99_1.npz'))

        images = ['data/demo/zjl.jpg', 'data/demo/zjl1.jpg', 'data/demo/ffy.jpg']
        feat0 = facenet.test(images[0])
        feat1 = facenet.test(images[1])
        feat2 = facenet.test(images[2])
        print(euclidean_distance(feat0, feat1))
        print(euclidean_distance(feat0, feat2))
        print(euclidean_distance(feat1, feat2))

        images = celebA.identity2images['1']
        feat0 = facenet.test(images[0])
        feat1 = facenet.test(images[1])
        feat3 = facenet.test(images[2])
        feat5 = facenet.test(images[3])
        images = celebA.identity2images['2']
        feat2 = facenet.test(images[0])
        feat4 = facenet.test(images[1])
        feat6 = facenet.test(images[2])
        # print(feat0)
        # print(feat1)
        # print(feat2)
        print(euclidean_distance(feat0, feat1))
        print(euclidean_distance(feat0, feat2))
        print(euclidean_distance(feat0, feat3))
        print(euclidean_distance(feat0, feat4))
        print(euclidean_distance(feat0, feat5))
        print(euclidean_distance(feat0, feat6))
    else:
        raise NotImplementedError
