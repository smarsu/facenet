# --------------------------------------------------------
# FaceNet Datasets
# Licensed under The MIT License [see LICENSE for details]
# Copyright 2019 smarsu. All Rights Reserved.
# --------------------------------------------------------

import os
import os.path as osp
import cv2
import numpy as np


class Dataset(object):
    def __init__(self):
        pass


    def train_datas(self, batch_size):
        """"""
        n = len(self.identitys) // (batch_size * 2)
        # [n, batch_size]
        identitys = np.random.choice(self.identitys, n * batch_size * 2, replace=False)
        datas = []
        for idx, identity in enumerate(identitys):
            images = self.identity2images[identity]
            if idx % 2 == 0:
                choosed_images = np.random.choice(images, 2, replace=False).reshape(2)
            else:
                choosed_images = np.random.choice(images, 1, replace=False).reshape(1)
            datas.append(choosed_images)
        datas = np.concatenate(datas, 0).reshape(n, batch_size * 3)
        return datas


class CelebA(Dataset):
    def __init__(self, root, face_size=(128, 128)):
        """

        TODO: Seperate train, value and test data.

        Args:
            root: The root path of CelebA dataset. The root path should be be 
                like `xxx/CelebA`.
        """
        if osp.split(root)[-1] != 'CelebA':
            raise ValueError('The root of CelebA dataset should be like '
                             '`xxx/CelebA` not {}'.format(self._root))

        self._root = root
        self._eval_path = osp.join(self._root, 
                                   'Eval', 
                                   'list_eval_partition.txt')
        self.images, self.train_set, self.val_set, self.test_set = \
            self._load_eval(self._eval_path)
        self._anno_root = osp.join(self._root, 'Anno')

        self._croped_image_path = osp.join(self._root, 'Img', 'image-crop')
        if not osp.exists(self._croped_image_path):
            self._inTheWild_image_path = osp.join(self._root, 
                                                  'Img', 
                                                  'img_align_celeba')
            os.mkdir(self._croped_image_path)
            self._crop_write_image(self._inTheWild_image_path, 
                                   self.images,
                                   self._croped_image_path)

        # The attr and landmark are not been used.
        self._identity_path = osp.join(self._anno_root, 'identity_CelebA.txt')
        self.image2identity, self.identity2images = \
            self._load_identity(self._identity_path)
        
        self.train_size = len(self.images)
        self.identitys = list(self.identity2images.keys())
    

    def real_image_path(self, image):
        """Get the real image path.
        
        Args:
            image: The name of image. e.g. `000001.jpg`
        """
        return osp.join(self._croped_image_path, image)


    def _crop_write_image(self, inroot, images, outroot):
        """Crop and write in-the-wild image to a new folder.
        
        Args:
            inroot: The root path of in-the-wild images.
            images: The list of image names.
            outroot: The destination path of croped images.
        """
        for image in images:
            inimage_path = osp.join(inroot, image)
            cvimg = cv2.imread(inimage_path)
            cvimg = cvimg[60:-30, 25:-25]
            h, w, _ = cvimg.shape
            assert h == w == 128
            outimage_path = osp.join(outroot, image)
            cv2.imwrite(outimage_path, cvimg)
            print(outimage_path)


    def _load_eval(self, eval_path):
        """Load three set of train, value and test.
        
        "0" represents training image, "1" represents validation image, 
        "2" represents testing image;

        Args:
            eval_path: The path look like 
                `xxx/CelebA/Eval/list_eval_partition.txt`

        Returns:
            train set, value set, test set.
        """
        with open(eval_path, 'r') as fb:
            images = list()
            setmap = {'0': set(), '1': set(), '2': set()}
            for line in fb.readlines():
                image, tag = line.split()
                setmap[tag].add(image)
                images.append(image)
            return images, setmap['0'], setmap['1'], setmap['2']


    def _load_identity(self, identity_path):
        """Load the map between the identity and image.
        
        Args:
            identity_path: The path look like 
                `xxx/CelebA/Anno/identity_CelebA.txt`
        """
        with open(identity_path, 'r') as fb:
            image2identity = {}
            identity2images = {}
            for line in fb.readlines():
                image, identity = line.split()
                image2identity[image] = identity
                # Here use real path as it will be train.
                identity2images[identity] = identity2images.get(identity, []) \
                    + [self.real_image_path(image)]

            invalid = []
            for identity, images in identity2images.items():
                if len(images) < 2:
                    invalid.append(identity)
            for identity in invalid:
                identity2images.pop(identity)
            return image2identity, identity2images


if __name__ == '__main__':
    root = '/share/datasets/CelebA'
    celebA = CelebA(root)
