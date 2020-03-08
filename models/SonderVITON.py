from .base_model import BaseModel
from . import networks
import numpy as np
import torch
import itertools
from utils import write_images


class SonderFlowEstimator(BaseModel):
    def name(self):
        return "SonderFlowEstimatorModel"

    @staticmethod
    def modify_commandline_options(opts, is_train=True):
        print("modifying opts")
        return opts

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = []

        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ["Flow"]

        else:  # during test time, only load Gs
            self.model_names = []

        # load/define networks
        self.netFlow = networks.FlowEstimator([2, 2, 2, 2])

        self.comet_exp = opt.comet.exp
        self.store_image = opt.val.store_image
        self.opt = opt

        if self.isTrain:
            # define loss functions
            self.perceptual_loss = networks.VGGPerceptualLoss()

            self.optimizer_flow = torch.optim.Adam(
                itertools.chain(self.netFlow.parameters()),
                lr=opt.flow.opt.lr,
                betas=(opt.flow.opt.beta1, 0.999),
            )

            self.optimizers = []
            self.optimizers.append(self.optimizer_flow)

    def set_input(self, input):

        self.cloth = input.cloth.to(self.device)
        self.mask = input.cloth_mask.to(self.device)
        self.image = input.image.to(self.device)
        self.parse_cloth = input.parse_cloth.to(self.device)
        self.cloth_name = input.c_name
        self.im_name = input.im_name

    def forward(self):
        # Input args: c_s, s_s, s_t
        self.warped_cloth = self.netFlow(self.cloth, self.mask, self.parse_cloth)

    def backward_flow(self):

        # Try reconstructing the input (identity mapping):
        loss_func = torch.nn.L1Loss()

        self.loss = loss_func(self.warped_cloth, self.cloth)

        """
        self.loss_G = self.criterionGAN(
            self.netD(torch.cat([self.image, self.fake_mask], dim=1)), True
        )
        """
        # Log G loss to comet:
        if self.comet_exp is not None:
            self.comet_exp.log_metric("loss", self.loss.cpu().detach())
        self.loss.backward()

    def optimize_parameters(self):

        # forward
        self.forward()

        # Flow
        # self.set_requires_grad(self.netD, False)
        self.optimizer_flow.zero_grad()
        self.backward_flow()
        self.optimizer_flow.step()

        """
        # D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()
        """

    def save_test_images(self, test_display_data, curr_iter):
        save_images = []
        for i in range(len(test_display_data)):

            self.set_input(test_display_data[i])

            self.test()

            save_images.append(self.cloth[0])
            save_images.append(self.mask[0].repeat(3, 1, 1))
            save_images.append(self.image[0])
            save_images.append(self.parse_cloth[0].repeat(3, 1, 1))
            save_images.append(self.warped_cloth[0])

        write_images(save_images, curr_iter, comet_exp=self.comet_exp, store_im=self.store_image)

