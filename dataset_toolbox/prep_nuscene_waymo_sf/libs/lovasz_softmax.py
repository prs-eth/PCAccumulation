"""

MIT License

Copyright (c) 2018 Maxim Berman
Copyright (c) 2020 Tiago Cortinhal, George Tzelepis and Eren Erdal Aksoy


Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

"""
import torch
import torch.nn as nn
from torch.autograd import Variable


try:
    from itertools import ifilterfalse
except ImportError:
    from itertools import filterfalse as ifilterfalse


def isnan(x):
    return x != x


def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_softmax_flat(probas, labels):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
    """
    if probas.numel() == 0:
        # only void pixels, the gradients should be 0
        return probas * 0.
    C = probas.size(1) # number of classes
    assert C > 1
    class_to_sum = list(range(C))
    losses = []
    for c in class_to_sum:
        fg = (labels == c).float()  # foreground for class c
        if (fg.sum() == 0):
            continue
        class_pred = probas[:, c]
        errors = (Variable(fg) - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted))))
    return mean(losses)

class Lovasz_softmax(nn.Module):

    def __init__(self):
        super(Lovasz_softmax, self).__init__()

    def forward(self, probas, labels):
        """
        Multi-class Lovasz-Softmax loss
            probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
            labels: [P] Tensor, ground truth labels (between 0 and C - 1)
        """
        return lovasz_softmax_flat(probas, labels)