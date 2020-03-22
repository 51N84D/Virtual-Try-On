import os
import re
from pathlib import Path
import subprocess
from copy import copy
import yaml
from addict import Dict
from torch.nn import init
import torch
import numpy as np
import importlib
from models.base_model import BaseModel
import functools
import torch.nn as nn
import time
import torchvision.utils as vutils


def load_opts(path=None, default=None):
    """Loads a configuration Dict from 2 files:
    1. default files with shared values across runs and users
    2. an overriding file with run- and user-specific values
    Args:
        path (pathlib.Path): where to find the overriding configuration
            default (pathlib.Path, optional): Where to find the default opts.
            Defaults to None. In which case it is assumed to be a default config
            which needs processing such as setting default values for lambdas and gen
            fields
    Returns:
        addict.Dict: options dictionnary, with overwritten default values
    """
    if default is None:
        default_opts = Dict()
    else:
        with open(default, "r") as f:
            default_opts = Dict(yaml.safe_load(f))

    with open(path, "r") as f:
        overriding_opts = Dict(yaml.safe_load(f))

    default_opts.update(overriding_opts)

    return set_data_paths(default_opts)


def set_data_paths(opts):
    """Update the data files paths in data.files.train and data.files.val
    from data.files.base
    Args:
        opts (addict.Dict): options
    Returns:
        addict.Dict: updated options
    """

    for mode in ["train", "val"]:
        for domain in opts.data.files[mode]:
            opts.data.files[mode] = str(Path(opts.data.files.base) / opts.data.files[mode])

    return opts


class Timer:
    def __init__(self, msg):
        self.msg = msg
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_value, exc_tb):
        print(self.msg % (time.time() - self.start_time))


def set_mode(mode, opts):

    if mode == "train":
        opts.model.is_train = True
    elif mode == "test":
        opts.model.is_train = False
    else:
        raise "invalid mode"

    return opts


def write_images(image_outputs, curr_iter, im_per_row=6, comet_exp=None, store_im=False):
    """Save output image
    Arguments:
        image_outputs {Tensor list} -- list of output images
        im_per_row {int} -- number of images to be displayed (per row)
        file_name {str} -- name of the file where to save the images
    """

    image_outputs = torch.stack(image_outputs)
    image_grid = vutils.make_grid(image_outputs, nrow=im_per_row, normalize=True, scale_each=True)
    image_grid = image_grid.permute(1, 2, 0).cpu().detach().numpy()

    if comet_exp is not None:
        comet_exp.log_image(image_grid, name="test_iter_" + str(curr_iter))


def tv_loss(img, tv_weight):
    """
    Compute total variation loss.
    Inputs:
    - img: PyTorch Variable of shape (1, 3, H, W) holding an input image.
    - tv_weight: Scalar giving the weight w_t to use for the TV loss.
    Returns:
    - loss: PyTorch Variable holding a scalar giving the total variation loss
      for img weighted by tv_weight.
    """
    w_variance = torch.sum(torch.pow(img[:, :, :, :-1] - img[:, :, :, 1:], 2))
    h_variance = torch.sum(torch.pow(img[:, :, :-1, :] - img[:, :, 1:, :], 2))
    loss = tv_weight * (h_variance + w_variance)
    return loss


def prepare_sub_folder(output_directory):
    """Create images and checkpoints subfolders in output directory
    Arguments:
        output_directory {str} -- output directory
    Returns:
        checkpoint_directory, image_directory-- checkpoints and images directories
    """
    image_directory = os.path.join(output_directory, "images")
    if not os.path.exists(image_directory):
        print("Creating directory: {}".format(image_directory))
        os.makedirs(image_directory)
    checkpoint_directory = os.path.join(output_directory, "checkpoints")
    if not os.path.exists(checkpoint_directory):
        print("Creating directory: {}".format(checkpoint_directory))
        os.makedirs(checkpoint_directory)
    return checkpoint_directory, image_directory
