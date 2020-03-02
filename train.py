import time
from pathlib import Path
from comet_ml import Experiment
import sys
import copy
from models.SonderVITON import SonderFlowEstimator
from utils import load_opts, Timer
from data.dataloader import get_loader
from addict import Dict


# from data import CreateDataLoader
# from models import create_model
# from util.visualizer import Visualizer

if __name__ == "__main__":
    root = Path(__file__).parent.resolve()
    opt_file = "shared/defaults.yml"

    opt = load_opts(path=root / opt_file)

    loader = get_loader(opt)

    model = SonderFlowEstimator()
    model.initialize(opt)
    model.setup()

    total_steps = 0

    for epoch in range(opt.train.epochs):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(loader):
            with Timer("Elapsed time in update " + str(i) + ": %f"):

                total_steps += opt.data.loaders.batch_size
                epoch_iter += opt.data.loaders.batch_size

                model.set_input(Dict(data))
                # model.optimize_parameters()

                break
        break

