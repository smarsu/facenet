# --------------------------------------------------------
# SMNet FaceNet
# Licensed under The MIT License [see LICENSE for details]
# Copyright 2019 smarsu. All Rights Reserved.
# --------------------------------------------------------

import os.path as osp
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from euclidean import euclidean_distance

EPS = 1e-12


def load_feature_map_from_txt(path_txt):
    """"""
    with open(path_txt, 'r') as fb:
        lines = fb.readlines()
        feature_map = {}
        for line in lines:
            line = line.strip().split()
            name = line[0]
            feature = [float(v) for v in line[1:]]
            feature_map[name] = np.array(feature, dtype=np.float64)
        return feature_map


def load_pairs(pair_path):
    with open(pair_path, 'r') as fb:
        lfw_root = '/datasets/lfw_detected'
        lines = fb.readlines()
        pairs = []
        for line in lines:
            fst, snd, match = line.strip().split()
            fst = osp.join(lfw_root, fst)
            snd = osp.join(lfw_root, snd)
            pairs.append([fst, snd, int(match)])
        return pairs


def l2_norm(x):
    """
    Args:
        x: ndarray, [n, feature_len]
    """
    x = np.array(x, dtype=np.float64)
    return x / (np.sqrt(np.sum(np.square(x), axis=-1, keepdims=True)) + EPS)


def cosine_similarity(a, b):
    """
    Args:
        a: ndarray, [feature_len]
        b: ndarray, [feature_len]
    """
    a = np.array(a, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    return np.sum(a * b)


def auc(scores, labels):
    """"""
    return metrics.roc_auc_score(labels, scores)


def roc(scores, labels):
    """"""
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores)
    plt.plot(fpr, tpr)
    plt.savefig('roc.png')


def main():
    feature_map = load_feature_map_from_txt('features.txt')
    feature_map = {k: l2_norm(v) for k, v in feature_map.items()}

    pairs = load_pairs('parsed_pair.txt')
    scores = []
    labels = []
    for fst, snd, match in pairs:
        labels.append(match)
        if fst not in feature_map:
            scores.append(1)
            print('WARNING: not found', fst)
            continue
        elif snd not in feature_map:
            scores.append(1)
            print('WARNING: not found', snd)
            continue
        score = 2 - euclidean_distance(feature_map[fst], feature_map[snd])
        scores.append(score)

    print(scores)
    print(labels)

    print(min(scores))
    print(max(scores))
    
    print(auc(scores, labels))
    roc(scores, labels)


if __name__ == '__main__':
    main()
