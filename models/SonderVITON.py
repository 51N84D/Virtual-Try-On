from .base_model import BaseModel
from . import networks


lass SonderFlowEstimator(BaseModel):
    def name(self):
        return "SonderFlowEstimatorModel"
c
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

            """
            self.optimizer_G = torch.optim.Adam(
                itertools.chain(self.netG.parameters()),
                lr=opt.gen.opt.lr,
                betas=(opt.gen.opt.beta1, 0.999),
            )
            self.optimizer_D = torch.optim.Adam(
                itertools.chain(self.netD.parameters()),
                lr=opt.dis.opt.lr,
                betas=(opt.dis.opt.beta1, 0.999),
            )
            self.optimizers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            """

    def set_input(self, input):
        print(input.keys())
        print("cloth: ", input.cloth)
        """
        self.image = input.data.x.to(self.device)
        mask = input.data.m.to(self.device)
        self.mask = mask[:, 0, :, :].unsqueeze(1)
        self.paths = input.paths
        """

    def forward(self):
        print("Forward pass after setting input")
        # self.fake_mask = self.netG(self.image)

    def backward_G(self):
        print("Backwards of generator")
        """
        self.loss_G = self.criterionGAN(
            self.netD(torch.cat([self.image, self.fake_mask], dim=1)), True
        )
        # Log G loss to comet:
        if self.comet_exp is not None:
            self.comet_exp.log_metric("loss G", self.loss_G.cpu().detach())
        self.loss_G.backward()
        """

    def optimize_parameters(self):
        print("optimize parameters")
        """
        # forward
        self.forward()

        # G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

        # D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()
        """

    def save_test_images(self, test_display_data, curr_iter):
        print("save images")
        """
        overlay = self.overlay
        save_images = []
        for i in range(len(test_display_data)):
            self.set_input(test_display_data[i])
            self.test()
            save_images.append(self.image[0])
            # Overlay mask:
            save_mask = (
                self.image[0]
                - (self.image[0] * self.mask[0].repeat(3, 1, 1))
                + self.mask[0].repeat(3, 1, 1)
            )

            save_fake_mask = (
                self.image[0]
                - (self.image[0] * self.fake_mask[0].repeat(3, 1, 1))
                + self.fake_mask[0].repeat(3, 1, 1)
            )

            if overlay:
                save_images.append(save_mask)
                save_images.append(save_fake_mask)
            else:
                save_images.append(self.mask[0].repeat(3, 1, 1))
                save_images.append(self.fake_mask[0].repeat(3, 1, 1))
        write_images(save_images, curr_iter, comet_exp=self.comet_exp, store_im=self.store_image)
        """
