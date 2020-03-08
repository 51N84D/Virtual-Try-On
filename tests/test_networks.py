import torch
from pathlib import Path
from torch.autograd import Variable
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.networks import FlowEstimator, FPN

if __name__ == "__main__":

    # Test FPN
    print("Testing FPN....")
    test_input = Variable(torch.randn(1, 3, 256, 192))
    net = FPN([2, 2, 2, 2])
    fms = net(test_input)
    for i in fms:
        print(i.shape)
    print("---------------")
    print("Testing Flow Estimator....")
    # Test Flow Estimator
    test_source_cloth = Variable(torch.randn(1, 3, 256, 192))
    test_source_seg = Variable(torch.randn(1, 3, 256, 192))
    test_target_seg = Variable(torch.randn(1, 3, 256, 192))
    net = FlowEstimator([2, 2, 2, 2])
    broh = net(test_source_cloth, test_source_seg, test_target_seg)
    # print(net)
