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


def write_images(image_outputs, curr_iter, im_per_row=5, comet_exp=None, store_im=False):
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
