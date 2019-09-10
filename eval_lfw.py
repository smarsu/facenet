# --------------------------------------------------------
# SMNet FaceNet
# Licensed under The MIT License [see LICENSE for details]
# Copyright 2019 smarsu. All Rights Reserved.
# --------------------------------------------------------

import os.path as osp
import cv2
from facenet import FaceNet

face_size = (96, 96)
facenet = FaceNet(face_size=face_size)
facenet.sess.restore(osp.join(facenet.model_root, 'nan_99_1.npz'))


def extract_feature(img):
    """Extract features of image.

    Args:
        img: ndarray, [h, w, c]
    """
    return facenet.test(img)


def main():
    lfw_detected_root = '/datasets/lfw_detected'
    with open('lfw_person.txt') as fb:
        lines = fb.readlines()
        lfw_persons = [osp.join(lfw_detected_root, line.strip()) for line in lines]

    with open('features.txt', 'w') as fb:
        for person in lfw_persons:
            print('INFO: extract feature of ', person)
            img = cv2.imread(person)
            if img is None:
                print('WARNING: no detected face ', person)
                continue

            feature = extract_feature(img)
            _person = osp.join(*person.split()[-2:])

            fb.write(person)
            for v in feature:
                fb.write(' ' + str(v))
            fb.write('\n')


if __name__ == '__main__':
    main()

