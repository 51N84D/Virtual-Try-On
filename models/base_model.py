import os
import torch
from collections import OrderedDict
from . import networks


class BaseModel:

    # modify parser to add command line options,
    # and also change the default values if needed
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def name(self):
        return "BaseModel"

    def initialize(self, opt):
        self.opt = opt
        self.use_gpu = opt.model.use_gpu
        self.device = torch.device(
            "cuda" if (torch.cuda.is_available() and self.use_gpu) else "cpu"
        )
        self.isTrain = opt.model.is_train
        self.loss_names = []
        self.model_names = []
        self.comet_exp = opt.comet_exp
        self.save_dir = opt.train.output_dir + "/checkpoints"

    def set_input(self, input):
        pass

    def forward(self):
        pass

    def validate(self):
        pass

    # load and print networks; create schedulers
    def setup(self):
        if not self.isTrain or self.opt.train.resume_checkpoint:
            load_suffix = "iter_%d" % self.opt.train.load_iter
            self.load_networks(load_suffix)
        if not self.isTrain:
            self.eval()
        self.print_networks(self.opt.model.verbose)

    # make models eval mode during test time
    def eval(self):
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, "net" + name)
                net.eval()

    # used in test time, wrapping `forward` in no_grad() so we don't save
    # intermediate steps for backprop
    def test(self):
        with torch.no_grad():
            self.forward()

    def optimize_parameters(self):
        pass

    # save models to the disk
    def save_networks(self, epoch):
        print("save models")  # TODO: save checkpoints
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = "%s_net_%s.pth" % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, "net" + name)

                if self.use_gpu and torch.cuda.is_available():
                    torch.save(net.state_dict(), save_path)
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    # load models from the disk
    def load_networks(self, epoch):

        for name in self.model_names:
            if isinstance(name, str):
                load_filename = "%s_net_%s.pth" % (epoch, name)
                load_path = os.path.join(self.save_dir, load_filename)
                net = getattr(self, "net" + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print("loading the model from %s" % load_path)
                # if you are using PyTorch newer than 0.4 (e.g., built from
                # GitHub source), you can remove str() on self.device
                state_dict = torch.load(load_path, map_location=str(self.device))
                if hasattr(state_dict, "_metadata"):
                    del state_dict._metadata

                # patch InstanceNorm checkpoints prior to 0.4
                for key in list(
                    state_dict.keys()
                ):  # need to copy keys here because we mutate in loop
                    self.__patch_instance_norm_state_dict(state_dict, net, key.split("."))
                net.load_state_dict(state_dict)

    # print network information
    def print_networks(self, verbose):
        print("---------- Networks initialized -------------")
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, "net" + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print("[Network %s] Total number of parameters : %.3f M" % (name, num_params / 1e6))
        print("-----------------------------------------------")

    # set requies_grad=Fasle to avoid computation
    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith("InstanceNorm") and (
                key == "running_mean" or key == "running_var"
            ):
                if getattr(module, key) is None:
                    state_dict.pop(".".join(keys))
            if module.__class__.__name__.startswith("InstanceNorm") and (
                key == "num_batches_tracked"
            ):
                state_dict.pop(".".join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)
