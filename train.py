import time
from pathlib import Path
from comet_ml import Experiment
import sys
import copy
from models.SonderVITON import SonderFlowEstimator
from utils import load_opts

# from data import CreateDataLoader
# from models import create_model
# from util.visualizer import Visualizer

if __name__ == "__main__":
    root = Path(__file__).parent.resolve()
    opt_file = "shared/defaults.yml"

    opt = load_opts(path=root / opt_file)

    model = SonderFlowEstimator()
    model.initialize(opt)
    model.setup()

