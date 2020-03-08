from pathlib import Path
import sys
from addict import Dict
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))

from data.dataloader import *
from run import opts

from utils import set_mode

if __name__ == "__main__":
    print("Check the dataset for geometric matching module!")

    opts = set_mode("test", opts)
    dataset = CPDataset(opts)
    data_loader = get_loader(opts)

    for i in data_loader:
        i = Dict(i)
        print("cloth: ", i.cloth.shape)
        print("cloth mask: ", i.cloth_mask.shape)
        print("image: ", i.image.shape)
        print("parse cloth: ", i.parse_cloth.shape)
        print("c name: ", i.c_name)
        print("im name: ", i.im_name)
        print("**************************************")

        break
