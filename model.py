# --------------------------------------------------------
# SMNet FaceNet backbone
# Licensed under The MIT License [see LICENSE for details]
# Copyright 2019 smarsu. All Rights Reserved.
# --------------------------------------------------------

"""
1. VGG16 meta-architecture definition.

Reference:
https://arxiv.org/pdf/1409.1556.pdf

2. Triplet loss of FaceNet
https://arxiv.org/pdf/1503.03832.pdf
"""
import math
import smnet as sm


def _shapeinfer(h, w):
    """The shape infer for vgg16.
    
    Args:
        h: The height of input image.
        w: The width of input image.

    Returns:
        h: The height of output feature map.
        w: The width of output feature map.
    """
    # 5 for the five times downsampling.
    for _ in range(5):
        h = math.ceil(h / 2)
        w = math.ceil(w / 2)
    return h, w


def block(x, times, ci, co):
    """The block of vgg16 network.

    The default filter size of vgg16 block is 3. The default stride is 1. 
    The default padding strategy is SAME. The default dilation is 1.
    
    e.g.
    [conv3-512]
    [conv3-512]
    [conv3-512]
    [ maxpool ]

    Args:
        x: The input tensor.
        times: The times of repeat conv2d op.
        ci: The in channel of input tensor.
        co: The out channel of output tensor.
    """
    x = sm.slim.conv2d(x, ci, co, 3, 1)
    for _ in range(times - 1):
        x = sm.slim.conv2d(x, co, co, 3, 1)
    x = sm.max_pool(x, (1, 2, 2, 1), (1, 2, 2, 1), 'SAME')
    return x


def vgg16(x, h, w):
    """The body of vgg16.
    
    224 -> 7

    Args:
        x: The input tensor. [n, 224, 224, 3]
        h: int, the height of input.
        w: int, the width of input.
    
    Returns:
        x: The output feature map. [n, 7, 7, 512]
        ho, wo, co: int, part shape of the fmap.
    """
    if sm.net.use_cuda:
        x = sm.transform(x, 'NHWC2NCHW')
    x = block(x, 2, 3, 64)  # 224 -> 112
    x = block(x, 2, 64, 128)  # 112 -> 56   
    x = block(x, 3, 128, 256)  # 56 -> 28
    x = block(x, 3, 256, 512)  # 28 -> 14
    x = block(x, 3, 512, 512)  # 14 -> 7
    if sm.net.use_cuda:
        x = sm.transform(x, 'NCHW2NHWC')

    # The head for classify are removed
    co = 512
    ho, wo = _shapeinfer(h, w)
    return x, ho, wo, co


def facenet_head(x, h, w, c):
    """The head to embed.
    
    The embedding size of face is 128.

    Args:
        x: Tensor. [n, h, w, c].
        h: int, the height of feature map.
        w: int, the width of feature map.
        c: int, the number of channel in.

    Returns:
        x: The normlized
    """
    if sm.net.use_cuda:
        x = sm.transform(x, 'NHWC2NCHW')
    #x = sm.avg_pool(x, (1, h, w, 1), (1, h, w, 1), 'VALID')
    x = sm.max_pool(x, (1, h, w, 1), (1, h, w, 1), 'VALID')
    if sm.net.use_cuda:
        x = sm.transform(x, 'NCHW2NHWC')
    x = sm.reshape(x, (-1, c))
    x = sm.slim.fc(x, c, 128, act=None)
    # norm
    x = x / sm.sqrt(sm.sum((sm.square(x)), -1, keepdims=True) + 1.084202172485504e-19)
    return x


def facenet_loss(logits, alpha, mask):
    """Variant of triplet loss

    We implemented the triplet loss following the Facenet. Howerver, it is too
    difficult to converge. So some improvement have be done to help the converge
    of it. For example, we scaled the distance between anchor and negative 
    samples to ensure that when the distance are large, the grad will be small
    otherwise the grad will be large.

    Args:
        logits: The output of facenet_head. [n * 3, 128], [anchor, pos, neg]
        alpha: A margin that is enforced between positive and negative pairs.
        mask: Tensor. To restrain the loss. It should be zeros_like [n, 1, 1].

    Returns:
        loss:
    """
    embeds = sm.reshape(logits, (-1, 3, 128))
    anchor, pos, neg = sm.split(embeds, 3, 1)

    disp = sm.sum(sm.square(anchor - pos), -1)
    disn = sm.sum(sm.square(anchor - neg), -1)
    dis = disp - disn
    mask = dis > mask

    # disn = sm.pow(sm.sum(sm.square(anchor - neg), -1), 0.25)

    return mask * (disp - disn), dis

    embeds = sm.reshape(logits, (-1, 3, 128))
    # [n, 1, 128]
    anchor, pos, neg = sm.split(embeds, 3, 1)
    # [n, 1]
    disp = sm.sum(sm.square(anchor - pos), -1)
    # Here we use sqrt to active the distance of anchor and neg to help for 
    # converge.
    #disn = sm.pow(sm.sum(sm.square(anchor - neg), -1), 0.25)
    disn = sm.sum(sm.square(anchor - neg), -1)
    #return sm.maximum(disp - disn + alpha, 0)

    # [n, 1]
    #loss = sm.relu(disp - disn + alpha)
    #return loss
    loss = sm.maximum(disp - disn + alpha, 0)
    scale = loss > mask
    scale = sm.sum(scale)
    scale = sm.maximum(scale, 1)
    loss /= scale
    return loss


def _check_validity():
    """Check the validity of vgg16 in smnet."""
    lr = 1
    epoch = 10000
    alpha = 0.5
    n = 4 * 3
    h, w = 128, 128
    x = sm.Tensor()
    mask = sm.Tensor()
    feat, ho, wo, co = vgg16(x, h, w)
    embed = facenet_head(feat, ho, wo, co)
    loss = facenet_loss(embed, alpha, mask)

    # data in host.
    dataHs, maskHs = [], []
    dataH = np.random.normal(0, 1, (n, h, w, 3))
    maskH = np.zeros([n // 3, 1])
    for _ in range(epoch):
        dataHs.append(dataH)
        maskHs.append(maskH)

    # smnet
    featHs, embedHs, lossHs = [], [], []
    for _ in range(epoch):
        featH, embedH, lossH = sm.forward([feat, embed, loss], 
                                          feed_dict={x: dataHs[_], 
                                                     mask: maskHs[_]})
        loss_sum = np.sum(lossH)
        sm.optimize([loss], lr=lr)
        featHs.append(featH)
        embedHs.append(embedH)
        lossHs.append(loss_sum)
        print(_, loss_sum)

    print(featHs)
    print()
    print(embedHs)
    print()
    print(lossHs)


if __name__ == '__main__':
    import numpy as np
    np.random.seed(196)
    _check_validity()
