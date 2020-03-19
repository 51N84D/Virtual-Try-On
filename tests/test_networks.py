import torch
from pathlib import Path
from torch.autograd import Variable
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.networks import FlowEstimator, FPN, define_D, ResBlockFPN
from run import opts


if __name__ == "__main__":

    # Test FPN
    print("Testing FPN....")
    test_input = Variable(torch.randn(1, 4, 256, 192))
    net = ResBlockFPN(input_nc=4)

    fms = net(test_input)
    for i in fms:
        print(i.shape)

    print("---------------")
    print("Testing Flow Estimator....")
    # Test Flow Estimator
    test_source_cloth = Variable(torch.randn(2, 3, 256, 192))
    test_source_seg = Variable(torch.randn(2, 1, 256, 192))
    test_target_seg = Variable(torch.randn(2, 1, 256, 192))
    net = FlowEstimator()
    warped_cloth, warped_mask = net(test_source_cloth, test_source_seg, test_target_seg)
    print("warped cloth: ", warped_cloth.shape)
    print("warped mask: ", warped_mask.shape)
    # print(net)
    """
    print("---------------")
    print("Testing Discriminator....")
    dis = define_D(opts)
    dis_input = torch.cat([broh, test_target_seg], dim=1)
    dis_output = dis(dis_input)

    print("Dis output shape: ", dis_output.shape)
    """

