# --------------------------------------------------------
# ImageShop
# Licensed under The MIT License [see LICENSE for details]
# Copyright 2019 smarsu. All Rights Reserved.
# --------------------------------------------------------

import cv2
import numpy as np


def blur(image, factor=(0.25, 1)):
    """
    Args:
        image: ndarray.
        factor: (float, float) (low, high)
    """
    low, high = factor
    factor = np.random.uniform(low, high)
    h, w = image.shape[:2]
    tmph, tmpw = round(h * factor), round(w * factor)
    tmpimage = cv2.resize(image, (tmpw, tmph))
    image = cv2.resize(tmpimage, (w, h))
    return image


def flip(image):
    """0.5 perceptor for horizontal flip.

    Args:
        image: ndarray.

    Returns:
        image:
        target: If true, the ret image are flipped.
    """
    target = np.random.choice(2)
    if target:
        image = cv2.flip(image, 1)
    return image, target


def noise(image, alpha=(0.8, 1.25), beta=(-10, 10)):
    """Apply y = alpha * x + beta to image.
    
    Args:
        image: ndarray.
        alpha:
        beta:
    """
    image = image.astype(np.float32)
    alpha = np.random.uniform(alpha[0], alpha[1])
    beta = np.random.uniform(beta[0], beta[1])
    image = alpha * image + beta
    image = np.maximum(image, 0)
    image = np.minimum(image, 255)
    return image
