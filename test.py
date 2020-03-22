import time
from pathlib import Path
from comet_ml import Experiment
import sys
from models.SonderVITON import SonderFlowEstimator
from utils import *
from data.dataloader import get_loader
import copy


# from data import CreateDataLoader
# from models import create_model
# from util.visualizer import Visualizer

if __name__ == "__main__":
    root = Path(__file__).parent.resolve()
    opt_file = "shared/defaults.yml"

    opt = load_opts(path=root / opt_file, default=root / "shared/defaults.yml")

    opt = set_mode("test", opt)
    opt.data.loaders.batch_size = 1
    val_loader = get_loader(opt)
    dataset_size = len(val_loader)

    print("#testing images = %d" % dataset_size)

    comet_exp = Experiment(workspace=opt.comet.workspace, project_name=opt.comet.project_name)
    if comet_exp is not None:
        comet_exp.log_asset(file_data=str(root / opt_file), file_name=root / opt_file)
        comet_exp.log_parameters(opt)

    checkpoint_directory, image_directory = prepare_sub_folder(opt.train.output_dir)

    opt.comet.exp = comet_exp

    model = SonderFlowEstimator()
    model.initialize(opt)
    model.setup()

    total_steps = 0

    for i, data in enumerate(val_loader):
        with Timer("Elapsed time in update " + str(i) + ": %f"):
            total_steps += opt.data.loaders.batch_size
            model.set_input(Dict(data))
            print(Dict(data).data.keys())
            model.save_test_images([Dict(data)], total_steps)
