
import argparse
import pathlib
from argparse import ArgumentParser

# import h5py
import numpy as np
import torch
from runstats import Statistics
from skimage.metrics import structural_similarity, peak_signal_noise_ratio


def to_log(im, a=1000.):
    im_cur = np.copy(im)
    im_cur[im < 0] = 0
    return np.log10(a * im_cur + 1.) / np.log10(a)

def to_log_t(im, a=1000.):
    im_cur = torch.clone(im)
    im_cur[im < 0] = 0
    return torch.log10(a * im_cur + 1.) / torch.log10(a)

def to_lin(im, a=1000.):
    return (a ** im - 1.) / a


# second method
def rescale_lin(im):
    im = im-im.min()
    im = im/(im.max()+1e-7)
    return im

def rescale_log(im):
    im = im-im.min()
    im = np.log10(1000.*im+1.)/3.  # This is 3 = log10(1000.)
    return im

def rescale_log_t(im):
    im = im-im.min()
    im = torch.log10(1000.*im+1.)/3.  # This is 3 = log10(1000.)
    return im

def snr(true, y, e=1e-9):
    return 20 * np.log10(np.linalg.norm(true.flatten()) / (np.linalg.norm(true.flatten() - y.flatten()) + e))


def to_im(im):
    imc = np.copy(im)
    imc[imc > 1] = 1
    imc[imc < 0] = 0
    imc = np.uint8(imc * 255)
    return imc / 255.

def mse(gt, pred):
    """ Compute Mean Squared Error (MSE) """
    return np.mean((gt - pred) ** 2)


def nmse(gt, pred):
    """ Compute Normalized Mean Squared Error (NMSE) """
    return np.linalg.norm(gt - pred) ** 2 / np.linalg.norm(gt) ** 2


def psnr(gt, pred):
    """ Compute Peak Signal to Noise Ratio metric (PSNR) """
    return peak_signal_noise_ratio(gt, pred, data_range=gt.max())


def ssim(gt, pred):
    """ Compute Structural Similarity Index Metric (SSIM). """
    return structural_similarity(
        gt.transpose(1, 2, 0), pred.transpose(1, 2, 0), multichannel=True, data_range=gt.max()
    )


METRIC_FUNCS = dict(
    MSE=mse,
    NMSE=nmse,
    PSNR=psnr,
    SSIM=ssim,
)


class Metrics:
    """
    Maintains running statistics for a given collection of metrics.
    """

    def __init__(self, metric_funcs):
        self.metrics = {
            metric: Statistics() for metric in metric_funcs
        }

    def push(self, target, recons):
        for metric, func in METRIC_FUNCS.items():
            self.metrics[metric].push(func(target, recons))

    def means(self):
        return {
            metric: stat.mean() for metric, stat in self.metrics.items()
        }

    def stddevs(self):
        return {
            metric: stat.stddev() for metric, stat in self.metrics.items()
        }

    def __repr__(self):
        means = self.means()
        stddevs = self.stddevs()
        metric_names = sorted(list(means))
        return ' '.join(
            f'{name} = {means[name]:.4g} +/- {2 * stddevs[name]:.4g}' for name in metric_names
        )


