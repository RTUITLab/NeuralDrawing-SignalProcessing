import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
# from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from collections import defaultdict

class _SepConv1d(nn.Module):
    """A simple separable convolution implementation.

    The separable convlution is a method to reduce number of the parameters
    in the deep learning network for slight decrease in predictions quality.
    """

    def __init__(self, ni, no, kernel, stride, pad):
        super().__init__()
        self.depthwise = nn.Conv1d(ni, ni, kernel, stride, padding=pad, groups=ni)
        self.pointwise = nn.Conv1d(ni, no, kernel_size=1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class Flatten(nn.Module):
    """Converts N-dimensional tensor into 'flat' one."""

    def __init__(self, keep_batch_dim=True):
        super().__init__()
        self.keep_batch_dim = keep_batch_dim

    def forward(self, x):
        if self.keep_batch_dim:
            return x.view(x.size(0), -1)
        return x.view(-1)


def cosine(epoch, t_max, ampl):
    """Shifted and scaled cosine function."""

    t = epoch % t_max
    return (1 + np.cos(np.pi * t / t_max)) * ampl / 2


def inv_cosine(epoch, t_max, ampl):
    """A cosine function reflected on X-axis."""

    return 1 - cosine(epoch, t_max, ampl)


def one_cycle(epoch, t_max, a1=0.6, a2=1.0, pivot=0.3):
    """A combined schedule with two cosine half-waves."""

    pct = epoch / t_max
    if pct < pivot:
        return inv_cosine(epoch, pivot * t_max, a1)
    return cosine(epoch - pivot * t_max, (1 - pivot) * t_max, a2)


class SepConv1d(nn.Module):
    """Implementes a 1-d convolution with 'batteries included'.

    The module adds (optionally) activation function and dropout
    layers right after a separable convolution layer.
    """

    def __init__(self, ni, no, kernel, stride, pad,
                 drop=None, bn=True,
                 activ=lambda: nn.PReLU()):

        super().__init__()
        assert drop is None or (0.0 < drop < 1.0)
        layers = [_SepConv1d(ni, no, kernel, stride, pad)]
        if activ:
            layers.append(activ())
        if bn:
            layers.append(nn.BatchNorm1d(no))
        if drop is not None:
            layers.append(nn.Dropout(drop))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Scheduler:
    """Updates optimizer's learning rates using provided scheduling function."""

    def __init__(self, opt, schedule):
        self.opt = opt
        self.schedule = schedule
        self.history = defaultdict(list)

    def step(self, t):
        for i, group in enumerate(self.opt.param_groups):
            lr = self.opt.defaults['lr'] * self.schedule(t)
            group['lr'] = lr
            self.history[i].append(lr)


class Classifier(nn.Module):
    def __init__(self, raw_ni, fft_ni, no, drop=.5):

        super().__init__()

        self.raw = nn.Sequential(
            SepConv1d(raw_ni, 32, 8, 2, 3, drop=drop),
            SepConv1d(32, 32, 3, 1, 1, drop=drop),
            SepConv1d(32, 64, 8, 4, 2, drop=drop),
            SepConv1d(64, 64, 3, 1, 1, drop=drop),
            SepConv1d(64, 128, 8, 4, 2, drop=drop),
            SepConv1d(128, 128, 3, 1, 1, drop=drop),
            SepConv1d(128, 256, 8, 4, 2),
            Flatten(),
            nn.Dropout(drop), nn.Linear(256, 64), nn.PReLU(), nn.BatchNorm1d(64),
            nn.Dropout(drop), nn.Linear(64, 64), nn.PReLU(), nn.BatchNorm1d(64))

        self.fft = nn.Sequential(
            SepConv1d(fft_ni, 32, 8, 2, 4, drop=drop),
            SepConv1d(32, 32, 3, 1, 1, drop=drop),
            SepConv1d(32, 64, 8, 2, 4, drop=drop),
            SepConv1d(64, 64, 3, 1, 1, drop=drop),
            SepConv1d(64, 128, 8, 4, 4, drop=drop),
            SepConv1d(128, 128, 8, 4, 4, drop=drop),
            SepConv1d(128, 256, 8, 2, 3),
            Flatten(),
            nn.Dropout(drop), nn.Linear(256, 64), nn.PReLU(), nn.BatchNorm1d(64),
            nn.Dropout(drop), nn.Linear(64, 64), nn.PReLU(), nn.BatchNorm1d(64))

        self.out = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(inplace=True), nn.Linear(64, no))

        self.init_weights(nn.init.kaiming_normal_)

    def init_weights(self, init_fn):
        def init(m):
            for child in m.children():
                if isinstance(child, nn.Conv1d):
                    init_fn(child.weights)

        init(self)

    def forward(self, t_raw, t_fft):
        raw_out = self.raw(t_raw)
        fft_out = self.fft(t_fft)
        t_in = torch.cat([raw_out, fft_out], dim=1)
        out = self.out(t_in)
        return out
