import os
from pathlib import Path
import argparse
import torch
import sys

sys.path.append(str(Path(__file__).parent.parent.resolve()))

from utils import load_opts

root = Path(__file__).parent.parent.resolve()


opts = load_opts(path=root / "shared/defaults.yml")
