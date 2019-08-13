# --------------------------------------------------------
# FaceNet Datasets
# Licensed under The MIT License [see LICENSE for details]
# Copyright 2019 smarsu. All Rights Reserved.
# --------------------------------------------------------

import numpy as np


def euclidean_distance(a, b):
    """"""
    return np.sqrt(np.sum(np.square(a - b)))
