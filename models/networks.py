"""FPN in PyTorch.
See the paper "Feature Pyramid Networks for Object Detection" for more details.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class FPN(nn.Module):
    def __init__(self, num_blocks, input_nc=3, block=Bottleneck):
        super(FPN, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(input_nc, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Bottom-up layers
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        # Top layer
        self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels

        # Smooth layers
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        """Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        """
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode="bilinear") + y

    def forward(self, x):
        # Bottom-up
        c1 = F.relu(self.bn1(self.conv1(x)))
        # c1_1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        # Top-down
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p2 = self._upsample_add(p3, self.latlayer3(c2))

        # Smooth
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)

        return p2, p3, p4, p5


class FlowEstimator(nn.Module):
    def __init__(self, num_blocks, input_nc=256, block=Bottleneck):
        super(FlowEstimator, self).__init__()
        self.SourceFPN = FPN(num_blocks=num_blocks, input_nc=4, block=block)
        self.TargetFPN = FPN(num_blocks=num_blocks, input_nc=1, block=block)
        self.e5 = self.get_flow(input_nc)
        self.e4 = self.get_flow(input_nc)
        self.e3 = self.get_flow(input_nc)
        self.e2 = self.get_flow(input_nc)

    def forward(self, c_s, s_s, s_t):
        """[Forward pass of flow estimation network]

        Arguments:
            c_s {[torch Tensor]} -- [Source clothing item]
            s_s {[torch Tensor]} -- [Source segmentation]
            s_t {[torch Tensor]} -- [Target segmentation]

        Returns:
            [type] -- [description]
        """

        source_input = torch.cat([c_s, s_s], dim=1)
        s2, s3, s4, s5 = self.SourceFPN(source_input)
        t2, t3, t4, t5 = self.TargetFPN(s_t)

        f5 = self.e5(torch.cat([s5, t5], dim=1))
        f4 = self.upsample(f5) + self.e4(torch.cat([self.warp(s4, self.upsample(f5)), t4], dim=1))
        f3 = self.upsample(f4) + self.e3(torch.cat([self.warp(s3, self.upsample(f4)), t3], dim=1))
        f2 = self.upsample(f3) + self.e2(torch.cat([self.warp(s2, self.upsample(f3)), t2], dim=1))

        # Warped clothing item
        c_s_prime = self.warp(c_s, self.upsample(f2))

        return c_s_prime

    # Define convolutional encoder
    def get_flow(self, input_nc):
        """Given source and target feature maps, get flow

        Arguments:
            s {[torch Tensor]} -- [source feature map]
            t {[torch Tensor]} -- [target feature map]
        """
        # Concatenate source and target features

        flow_conv = nn.Conv2d(input_nc * 2, 2, kernel_size=1, stride=1)

        return flow_conv

    def upsample(self, F):
        """[2x nearest neighbor upsampling]

        Arguments:
            F {[torch tensor]} -- [tensor to be upsampled, (B, C, H, W)]
        """
        upsample = torch.nn.Upsample(scale_factor=2, mode="nearest")
        return upsample(F)

    def warp(self, S, F):
        """Warps the image S using flow F using bilinear interpolation

        Arguments:
            S {[torch Tensor]} -- [Image to interpolate from]
            F {[torch Tensor]} -- [Flow for interpolation]
        """
        # grid_samples expects (B, H, W, 2)
        F = F.permute(0, 2, 3, 1)
        return torch.nn.functional.grid_sample(S, F, mode="bilinear")


def FPN101():
    # return FPN(Bottleneck, [2,4,23,3])
    return FPN([2, 2, 2, 2])


import torch
import torchvision


class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=False):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.resize = resize

    def forward(self, input, target):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode="bilinear", size=(224, 224), align_corners=False)
            target = self.transform(target, mode="bilinear", size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss += torch.nn.functional.l1_loss(x, y)
        return loss


if __name__ == "__main__":
    test_input = Variable(torch.randn(1, 3, 128, 128))

    net = FPN101()
    fms = net(test_input)

