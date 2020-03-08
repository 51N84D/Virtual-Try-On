from pathlib import Path
import sys
from addict import Dict

sys.path.append(str(Path(__file__).resolve().parent.parent))

from data.dataloader import *
from run import opts


if __name__ == "__main__":
    print("Check the dataset for geometric matching module!")

    dataset = CPDataset(opts)
    data_loader = get_loader(opts)

    for i in data_loader:
        i = Dict(i)
        print("cloth: ", i.cloth.shape)
        print("cloth mask: ", i.cloth_mask.shape)
        print("image: ", i.image.shape)
        print("parse cloth: ", i.parse_cloth.shape)

        break
