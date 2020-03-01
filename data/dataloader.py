# coding=utf-8
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image
from PIL import ImageDraw

import os.path as osp
import numpy as np
import argparse

import matplotlib.pyplot as plt

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="GMM")
    parser.add_argument("--gpu_ids", default="")
    parser.add_argument('-j', '--workers', type=int, default=1)
    parser.add_argument('-b', '--batch-size', type=int, default=4)

    parser.add_argument("--dataroot", default= osp.abspath(__file__ + "/../../") + "/data")
    parser.add_argument("--datamode", default="train")
    parser.add_argument("--data_list", default="train_pairs.txt")
    parser.add_argument("--fine_width", type=int, default=192)
    parser.add_argument("--fine_height", type=int, default=256)

    parser.add_argument('--result_dir', type=str,
                        default= osp.abspath(__file__ + "/../../") + '/result', help='save result infos')
    parser.add_argument('--checkpoint', type=str, default='',
                        help='model checkpoint for test')
    
    parser.add_argument("--display_count", type=int, default=1)
    parser.add_argument("--shuffle", action='store_true',
                        help='shuffle input data')

    opt = parser.parse_args()
    return opt


class CPDataset(data.Dataset):

    def __init__(self, opt):
        super(CPDataset, self).__init__()
        # base setting
        self.opt = opt
        self.root = opt.dataroot
        self.datamode = opt.datamode  # train or test or self-defined
        self.data_list = opt.data_list
        self.fine_height = opt.fine_height
        self.fine_width = opt.fine_width
        self.data_path = osp.join(opt.dataroot, opt.datamode)
        self.transform = transforms.Compose([
                transforms.ToTensor(), 
                transforms.Normalize((0.5,), (0.5,))])

        # load data list
        im_names = []
        c_names = []
        print(opt.dataroot)
        with open(osp.join(opt.dataroot, opt.data_list), 'r') as f:
            print(f)
            for line in f.readlines():
                im_name, c_name = line.strip().split()
                im_names.append(im_name)
                c_names.append(c_name)

        self.im_names = im_names
        self.c_names = c_names

    def name(self):
        return "CPDataset"

    def __getitem__(self, index):
        c_name = self.c_names[index]
        im_name = self.im_names[index]

        # cloth image & cloth mask
        c = Image.open(osp.join(self.data_path, 'cloth', c_name))
        cm = Image.open(osp.join(self.data_path, 'cloth-mask', c_name))
        
        c = self.transform(c)  # [-1,1]
        cm_array = np.array(cm)
        cm_array = (cm_array >= 128).astype(np.float32)
        cm = torch.from_numpy(cm_array)  # [0,1]
        cm.unsqueeze_(0)

        # person image
        im = Image.open(osp.join(self.data_path, 'image', im_name))
        im = self.transform(im)  # [-1,1]

        # load parsing image
        parse_name = im_name.replace('.jpg', '.png')
        im_parse = Image.open(
            osp.join(self.data_path, 'image-parse', parse_name))
        parse_array = np.array(im_parse)

#-------Find segmentation class labels manually
#        Image1 = Image.open(osp.join(self.data_path, 'image-parse', parse_name))

#        plt.imshow(im_parse, cmap='jet')
#        plt.imshow(parse_array, alpha=0.5) 

#        plt.colorbar()
#        plt.show()        
        #shirt = 5, pants = 9
        #hair = 2, face = 13 
# ------End 

        parse_shape = (parse_array > 0).astype(np.float32)
        
        parse_cloth = (parse_array == 5).astype(np.float32) + \
                (parse_array == 9).astype(np.float32) 

        # shape downsample
        parse_shape = Image.fromarray((parse_shape*255).astype(np.uint8))
        parse_shape = parse_shape.resize(
            (self.fine_width//16, self.fine_height//16), Image.BILINEAR)
        parse_shape = parse_shape.resize(
            (self.fine_width, self.fine_height), Image.BILINEAR)
        shape = self.transform(parse_shape)  # [-1,1]
        pcm = torch.from_numpy(parse_cloth)  # [0,1]

        # clean up
        im_c = im * pcm + (1 - pcm)  # [-1,1], fill 1 for other parts

        result = {
            'c_name':   c_name,     # for visualization
            'im_name':  im_name,    # for visualization or ground truth
            'cloth':    c,          # for input
            'cloth_mask':     cm,   # for input
            'image':    im,         # for visualization
            'parse_cloth': im_c,    # for ground truth
            'shape': shape,         # for visualization
            }

        return result

    def __len__(self):
        return len(self.im_names)

class CPDataLoader(object):
    def __init__(self, opt, dataset):
        super(CPDataLoader, self).__init__()

        if opt.shuffle :
            train_sampler = torch.utils.data.sampler.RandomSampler(dataset)
        else:
            train_sampler = None

        self.data_loader = torch.utils.data.DataLoader(
                dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
                num_workers=opt.workers, pin_memory=True, sampler=train_sampler)
        self.dataset = dataset
        self.data_iter = self.data_loader.__iter__()
       
    def next_batch(self):
        try:
            batch = self.data_iter.__next__()
        except StopIteration:
            self.data_iter = self.data_loader.__iter__()
            batch = self.data_iter.__next__()

        return batch

