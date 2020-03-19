from .base_model import BaseModel
from . import networks
from .networks import init_weights
import numpy as np
import torch
import itertools
from utils import write_images
from addict import Dict


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
        self.loss_name = opt.model.loss_name

        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ["Flow"]

        else:  # during test time, only load Gs
            self.model_names = []

        # load/define networks
        self.netFlow = networks.FlowEstimator().to(self.device)
        init_type = opt.dis.params.init_type
        init_gain = opt.dis.params.init_gain
        init_weights(self.netFlow, init_type, init_gain)

        self.netD = networks.define_D(opt).to(self.device)

        self.comet_exp = opt.comet.exp
        self.store_image = opt.val.store_image
        self.opt = opt

        if self.isTrain:
            # define loss functions
            self.criterionVGG = networks.VGGPerceptualLoss().to(self.device)
            self.criterionGAN = networks.GANLoss(self.loss_name).to(self.device)

            self.optimizer_flow = torch.optim.Adam(
                itertools.chain(self.netFlow.parameters()),
                lr=opt.flow.opt.lr,
                betas=(opt.flow.opt.beta1, 0.999),
            )

            self.optimizer_D = torch.optim.Adam(
                itertools.chain(self.netD.parameters()),
                lr=opt.dis.opt.lr,
                betas=(opt.dis.opt.beta1, 0.999),
            )

            self.optimizers = []
            self.optimizers.append(self.optimizer_flow)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):

        self.cloth = input.cloth.to(self.device)
        self.mask = input.cloth_mask.to(self.device)
        self.image = input.image.to(self.device)
        self.parse_cloth = input.parse_cloth.to(self.device)
        self.cloth_name = input.c_name
        self.im_name = input.im_name

    def forward(self):
        # Input args: c_s, s_s, s_t
        self.warped_cloth, self.warped_mask = self.netFlow(self.cloth, self.mask, self.parse_cloth)

    def backward_flow(self):

        l1_loss = torch.nn.L1Loss().to(self.device)

        # self.loss_G_GAN = self.criterionGAN(
        #    self.netD(torch.cat([self.cloth, self.parse_cloth, self.warped_cloth], dim=1)), True
        # )

        self.loss_G_struct = l1_loss(self.warped_mask, self.parse_cloth)
        self.loss_G_struct2 = l1_loss(
            self.warped_cloth * self.warped_mask, self.image * self.parse_cloth
        )

        # self.loss_G_perc = self.criterionVGG(self.warped_cloth, self.image)

        self.loss_G = (
            self.loss_G_struct + self.loss_G_struct2
        )  # self.loss_G_GAN + self.loss_G_struct + self.loss_G_perc * 0.1
        self.loss_G.backward()

        # Log G loss to comet:
        if self.comet_exp is not None:
            # self.comet_exp.log_metric("loss G GAN", self.loss_G_GAN.cpu().detach())
            self.comet_exp.log_metric("loss G struct", self.loss_G_struct.cpu().detach())
            self.comet_exp.log_metric("loss G struct 2", self.loss_G_struct2.cpu().detach())
            # self.comet_exp.log_metric("loss G perceptual", self.loss_G_perc.cpu().detach())

        # Log G loss to comet:
        """
        if self.comet_exp is not None:
            self.comet_exp.log_metric("loss inside mask", self.loss1.cpu().detach())
            self.comet_exp.log_metric("loss outside mask", self.loss2.cpu().detach())
            self.comet_exp.log_metric("loss", self.loss.cpu().detach())
        
        self.loss.backward()
        """

    def backward_D(self):
        # Real

        real_mask_d = torch.cat([self.cloth, self.parse_cloth, self.image], dim=1)
        fake_mask_d = torch.cat([self.cloth, self.parse_cloth, self.warped_cloth], dim=1)

        pred_real = self.netD(real_mask_d)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        # Fake
        pred_fake = self.netD(fake_mask_d.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        if self.loss_name == "wgan":  # Get gradient penalty loss
            grad_penalty = networks.calc_gradient_penalty(
                self.opt, self.netD, real_mask_d, fake_mask_d
            )
            self.loss_D = (self.loss_D_real + self.loss_D_fake) * 0.5 + grad_penalty
            if self.comet_exp is not None:
                self.comet_exp.log_metric("grad penalty", grad_penalty.cpu().detach())

        else:
            # Combined loss
            self.loss_D = (self.loss_D_real + self.loss_D_fake) * 0.5

        # Log D loss to comet:
        if self.comet_exp is not None:
            self.comet_exp.log_metric("loss D", self.loss_D.cpu().detach())
            self.comet_exp.log_metric("loss D real", self.loss_D_real.cpu().detach())
            self.comet_exp.log_metric("loss D fake", self.loss_D_fake.cpu().detach())

        # backward
        self.loss_D.backward()

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

            self.set_input(Dict(test_display_data[i]))
            self.test()

            save_images.append(self.cloth[0])
            save_images.append(self.mask[0].repeat(3, 1, 1))
            save_images.append(self.image[0])
            save_images.append(self.parse_cloth[0].repeat(3, 1, 1))
            save_images.append(self.warped_cloth[0] * self.warped_mask[0])
            save_images.append(self.warped_mask[0].repeat(3, 1, 1))

        write_images(
            save_images,
            curr_iter,
            im_per_row=6,
            comet_exp=self.comet_exp,
            store_im=self.store_image,
        )

