# ============================================================================
#
#    Copyright 2020-2023 Irina Grigorescu
#    Copyright 2020-2023 King's College London
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
#
# ============================================================================

##############################################################################
#
# models.py
#
##############################################################################
import os
import time
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import init
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from torch.nn import BCELoss, L1Loss, MSELoss, SmoothL1Loss, CrossEntropyLoss

import src.schedulers as schedulers
import src.utils as utils
from src.networks import UNetModel, NLayerDiscriminator
from src.transformers import ToTensor, RandomCrop, RandomCropCortex, SplitLab, SplitLabEPRIME, SplitLabCortex
from src.dataloaders import SegmentationDataLoader, SegmentationDataLoaderCortex
from src.losses import gd_loss as GeneralisedDiceLoss
from src.losses import ncc as NCCLoss


# ==================================================================================================================== #
#
#  Helper functions for networks and models
#
# ==================================================================================================================== #
def init_weights(net, is_gen=False):
    """
    Initialises the weights of a network
    Link:  https://github.com/arnab39/cycleGAN-PyTorch/blob/ecdc9735e426992056c70e78f3143bb6d538e48c/arch/ops.py

    :param net:
    :return:
    """
    def init_func(m):
        classname = m.__class__.__name__

        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if is_gen:
                init.xavier_normal_(m.weight, gain=0.1)
            else:
                init.kaiming_normal_(m.weight, mode='fan_out')

            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias, 0.0)

        elif hasattr(m, 'weight') and classname.find('Linear') != -1:
            init.kaiming_normal_(m.weight, mode='fan_out')
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias, 0.0)

        elif (classname.find('BatchNorm') != -1) or (classname.find('GroupNorm') != -1):
            init.constant_(m.weight, 1.0)
            init.constant_(m.bias, 0.0)

    if is_gen:
        print('    >> Network initialized with xavier_normal_.')
    else:
        print('    >> Network initialized with kaiming_normal_.')

    net.apply(init_func)


def init_network(net, gpu_ids=[], is_gen=False):
    """
    Initialise network
    Link:  https://github.com/arnab39/cycleGAN-PyTorch/blob/ecdc9735e426992056c70e78f3143bb6d538e48c/arch/ops.py

    :param net:
    :param gpu_ids:
    :return:
    """
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.cuda(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, is_gen)
    return net


def set_grad(nets, requires_grad=False):
    """
    Set gradients
    Link:  https://github.com/arnab39/cycleGAN-PyTorch/blob/ecdc9735e426992056c70e78f3143bb6d538e48c/arch/ops.py

    :param nets:
    :param requires_grad:
    :return:
    """
    for net in nets:
        for param in net.parameters():
            param.requires_grad = requires_grad


def define_Net(input_nc, output_nc, n_features, net_name, gpu_ids=[0]):
    """
    Define the network (call the apropriate class based on the string provided)

    :param input_nc:
    :param output_nc:
    :param n_features:
    :param net_name:
    :param gpu_ids:
    :return:
    """
    is_gen = False

    # SEGMENTATION NETWORK
    if net_name == 'unet3D':
        net = UNetModel(num_channels=input_nc, num_classes=output_nc,
                        n_features=n_features, block='cli', name='UNet3D')

    # GENERATOR NETWORK
    elif net_name == 'unetgen3D':
        net = UNetModel(num_channels=input_nc, num_classes=output_nc,
                        n_features=n_features, use_activ=True, block='cli', name='GenUNet3D')
        is_gen = True

    # DISCRIMINATOR NETWORK
    elif net_name == 'n_layers':
        net = NLayerDiscriminator(input_nc, output_nc=output_nc,
                                  ndf=n_features, n_layers=3,
                                  norm_layer=nn.InstanceNorm3d, use_bias=True)

    else:
        raise NotImplementedError('Model name [%s] is not recognized' % net_name)

    return init_network(net, gpu_ids, is_gen)


# ==================================================================================================================== #
#
#  3D U-NET for segmentation of source data
#
# ==================================================================================================================== #
class Segmentation3DUNET(object):
    """
    Class for 3D U-NET that segments the source (dHCP) data
    """

    # ============================================================================
    def __init__(self, args):
        """
        Constructor
        :param args:
        """

        # Parameters setup
        #####################################################
        self.vol_size = (args.crop_height, args.crop_width, args.crop_depth)

        if args.gpu_ids is not None:
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        # Define the segmentation network
        #####################################################
        self.Seg = define_Net(input_nc=1,
                              output_nc=args.n_classes,
                              n_features=args.seg_features,
                              net_name=args.seg_net)

        utils.print_networks([self.Seg], ['Seg'])

        # Define Losses criterias
        #####################################################
        self.GDL = GeneralisedDiceLoss

        # Optimizers
        #####################################################
        self.s_optimizer = torch.optim.Adam(self.Seg.parameters(), lr=args.lr)

        self.s_lr_scheduler = torch.optim.lr_scheduler.CyclicLR(self.s_optimizer,
                                                                base_lr=args.lr / 1000,
                                                                max_lr=args.lr,
                                                                mode='triangular2',
                                                                step_size_up=args.epochs // 6,
                                                                cycle_momentum=False)

        # Data loaders for SOURCE DOMAIN (A)
        #####################################################
        transformed_dataset_train_domA = SegmentationDataLoader(
            csv_file=os.path.join(args.csv_dir, args.train_A_csv),
            root_dir=args.root_dir_dHCP,
            shuffle=True,
            is_augment=args.is_augment,
            transform=transforms.Compose([SplitLab(args.n_classes),
                                          RandomCrop(output_size=self.vol_size,
                                                     is_random=True,
                                                     n_classes=args.n_classes),
                                          ToTensor()]))

        transformed_dataset_valid_domA = SegmentationDataLoader(
            csv_file=os.path.join(args.csv_dir, args.valid_A_csv),
            root_dir=args.root_dir_dHCP,
            shuffle=True,
            is_augment=True,
            transform=transforms.Compose([SplitLab(args.n_classes),
                                          RandomCrop(output_size=self.vol_size,
                                                     is_random=True,
                                                     n_classes=args.n_classes),
                                          ToTensor()]))

        transformed_dataset_test_domA = SegmentationDataLoader(
            csv_file=os.path.join(args.csv_dir, args.test_A_csv),
            root_dir=args.root_dir_dHCP,
            shuffle=False,
            is_augment=False,
            transform=transforms.Compose([SplitLab(args.n_classes),
                                          RandomCrop(output_size=self.vol_size,
                                                     is_random=False,
                                                     n_classes=args.n_classes),
                                          ToTensor()]))

        # Data loaders for TARGET DOMAIN (B)
        #####################################################
        transformed_dataset_train_domB = SegmentationDataLoader(
            csv_file=os.path.join(args.csv_dir, args.train_B_csv),
            root_dir=args.root_dir_EPRIME,
            shuffle=True, is_augment=False,
            transform=transforms.Compose([SplitLab(args.n_classes),
                                          RandomCrop(output_size=self.vol_size,
                                                     is_random=True, n_classes=args.n_classes),
                                          ToTensor()]))

        transformed_dataset_valid_domB = SegmentationDataLoader(
            csv_file=os.path.join(args.csv_dir, args.valid_B_csv),
            root_dir=args.root_dir_EPRIME,
            shuffle=True, is_augment=False,
            transform=transforms.Compose([SplitLab(args.n_classes),
                                          RandomCrop(output_size=self.vol_size,
                                                     is_random=True, n_classes=args.n_classes),
                                          ToTensor()]))

        transformed_dataset_test_domB = SegmentationDataLoader(
            csv_file=os.path.join(args.csv_dir, args.test_B_csv),
            root_dir=args.root_dir_EPRIME,
            shuffle=False, is_augment=False,
            transform=transforms.Compose([SplitLab(args.n_classes),
                                          RandomCrop(output_size=self.vol_size,
                                                     is_random=False, n_classes=args.n_classes),
                                          ToTensor()]))

        self.dataloaders = {
            'train-A': DataLoader(transformed_dataset_train_domA, batch_size=args.batch_size,
                                  shuffle=True, num_workers=2*args.batch_size),
            'valid-A': DataLoader(transformed_dataset_valid_domA, batch_size=args.batch_size,
                                  shuffle=True, num_workers=2*args.batch_size),
            'test-A': DataLoader(transformed_dataset_test_domA, batch_size=1,
                                  shuffle=False, num_workers=1),

            'train-B': DataLoader(transformed_dataset_train_domB, batch_size=1,
                                  shuffle=True, num_workers=2 * args.batch_size),
            'valid-B': DataLoader(transformed_dataset_valid_domB, batch_size=1,
                                  shuffle=True, num_workers=2 * args.batch_size),
            'test-B': DataLoader(transformed_dataset_test_domB, batch_size=1,
                                 shuffle=False, num_workers=1)
        }

        # Check if results folder exists
        #####################################################
        if not os.path.isdir(args.results_dir):
            os.makedirs(args.results_dir)

        # Check if checkpoint folder exists
        #####################################################
        if not os.path.isdir(args.checkpoint_dir):
            os.makedirs(args.checkpoint_dir)

        # Try loading checkpoint
        #####################################################
        try:
            ckpt = utils.load_checkpoint('%s/latest.ckpt' % (args.checkpoint_dir))
            self.start_epoch = ckpt['epoch']
            self.losses_train = ckpt['losses_train']
            self.Seg.load_state_dict(ckpt['Seg'])
            self.s_optimizer.load_state_dict(ckpt['s_optimizer'])
        except:
            print('    [!] No checkpoint, starting from scratch!')
            self.start_epoch = 0
            self.losses_train = []


    # ============================================================================
    def train(self, args):
        """
        Train the network
        :param args:
        :return:
        """

        # Variables for train
        #####################################################
        best_segmentation_loss = 1e10
        plot_step = 1

        # Train (Go through each epoch
        #####################################################
        for epoch in range(self.start_epoch, args.epochs):

            # Print learning rate for each epoch
            lr = self.s_optimizer.param_groups[0]['lr']
            print("\n")
            print('            LEARNING RATE = %.7f' % lr)
            print("\n")

            # Save time to calculate how long it took
            start_time = time.time()

            # Metrics to store during training
            metrics = {'seg_loss_train': [], 'seg_loss_valid': [], 'lr': [lr]}

            # Set plotted to false at the start of each epoch
            plotted = False

            # For each epoch set the validation losses to 0
            seg_loss_valid = 0.0

            # Go through each data point TRAIN/VALID from domain A
            #####################################################
            for phase in ['train-A', 'valid-A']:

                for i, data_point in enumerate(self.dataloaders[phase]):

                    # # Uncomment for quick check
                    if i > 5:
                        break

                    # step
                    ##################################################
                    len_dataloader = len(self.dataloaders[phase])
                    step = epoch * len_dataloader + i + 1

                    # Fetch some data
                    ##################################################
                    t2w_gt = utils.cuda(Variable(data_point['image']))
                    seg_gt = utils.cuda(Variable(data_point['lab']))

                    # TRAIN
                    ##################################################
                    if phase == 'train-A':
                        # Set optimiser to zero grad
                        ##################################################
                        self.s_optimizer.zero_grad()

                        # Forward pass through UNet
                        ##################################################
                        pred_logits = self.Seg(t2w_gt)
                        pred_probs = torch.softmax(pred_logits, dim=1, dtype=torch.float32)

                        # Dice Loss
                        ###################################################
                        seg_loss = self.GDL(pred_probs, seg_gt, include_background=True) * args.lamda_seg

                        # Store metrics
                        ###################################################
                        metrics['seg_loss_train'].append(seg_loss.item())

                        # Update unet
                        ###################################################
                        seg_loss.backward()
                        self.s_optimizer.step()

                    # VALIDATE
                    #######################################################
                    else:
                        self.Seg.eval()

                        with torch.no_grad():
                            # Forward pass through UNet
                            ##################################################
                            pred_logits = self.Seg(t2w_gt)
                            pred_probs = torch.softmax(pred_logits, dim=1, dtype=torch.float32)

                            # Dice Loss
                            ###################################################
                            seg_loss = self.GDL(pred_probs, seg_gt, include_background=True) * args.lamda_seg

                            # Store metrics
                            ###################################################
                            metrics['seg_loss_valid'].append(seg_loss.item())

                            # Save valid losses here:
                            seg_loss_valid += seg_loss.item()

                        # Plot some images
                        #######################################################
                        if epoch % plot_step == 0 and not plotted:
                            plotted = True
                            utils.plot_seg_img(args, epoch, seg_gt, pred_probs, t2w_gt)

                        # Save best after all validation steps
                        #######################################################
                        if i == (args.validation_steps - 1):
                            seg_loss_valid /= args.validation_steps

                            print("\n")
                            print(f"        > Average Seg Loss     {seg_loss_valid:6.3f} ")

                            # Save best
                            if best_segmentation_loss > seg_loss_valid and epoch > 0:

                                best_segmentation_loss = seg_loss_valid
                                print(f"        > Best Seg Loss So Far {best_segmentation_loss:6.3f} ")

                                # Override the latest checkpoint for best generator loss
                                utils.save_checkpoint({'epoch': epoch + 1,
                                                       'Seg': self.Seg.state_dict(),
                                                       's_optimizer': self.s_optimizer.state_dict()},
                                                      '%s/latest_best_loss.ckpt' % (args.checkpoint_dir))

                                # Write in a file
                                with open('%s/README' % (args.checkpoint_dir), 'w') as f:
                                    f.write('Epoch: %d | Seg Loss: %f \n' % (epoch + 1, seg_loss_valid))

                            print("\n")

                            break

                    # PRINT STATS
                    ###################################################
                    time_elapsed = time.time() - start_time
                    print("              %s Epoch: (%3d) (%5d/%5d) (%3d) | Seg Loss:%.2e | %.0fm %.2fs" %
                          (phase.upper(), epoch, i + 1, len_dataloader, step,
                           seg_loss, time_elapsed // 60, time_elapsed % 60))

            # Append the metrics to losses_train
            ######################################
            self.losses_train.append(metrics)

            # Override the latest checkpoint at the end of an epoch
            #######################################################
            utils.save_checkpoint({'epoch': epoch + 1,
                                   'Seg': self.Seg.state_dict(),
                                   's_optimizer': self.s_optimizer.state_dict(),
                                   'losses_train': self.losses_train},
                                  '%s/latest.ckpt' % (args.checkpoint_dir))

            # Update learning rates
            ########################
            self.s_lr_scheduler.step()

        return self.losses_train


# ==================================================================================================================== #
#
#  3D U-NET for segmentation of source data with UDA in latent space
#
# ==================================================================================================================== #
class Segmentation3DLatent(object):
    """
    Class for UDA with latent space
    """
    # ============================================================================
    def __init__(self, args):

        # Define the network
        #####################################################
        self.Seg = define_Net(input_nc=1, output_nc=args.n_classes,
                              n_features=args.seg_features, net_name=args.seg_net)

        self.Dis = define_Net(input_nc=240, output_nc=2,
                              n_features=args.ndf, net_name=args.dis_net)

        utils.print_networks([self.Seg, self.Dis], ['Seg', 'Dis'])


        # print("-------------------")
        # print(self.Seg)
        # print("-------------------\n\n")
        #
        # print("-------------------")
        # print(self.Dis)
        # print("-------------------\n\n")


        # Define Losses criterias
        #####################################################
        self.GDL = GeneralisedDiceLoss
        self.NCC = NCCLoss
        self.MSE = MSELoss()
        self.BCE = BCELoss()
        self.CE = CrossEntropyLoss()
        self.L1 = L1Loss()
        self.SmoothL1 = SmoothL1Loss()

        # Optimizers
        #####################################################
        self.s_optimizer = torch.optim.Adam(self.Seg.parameters(), lr=args.lr)
        self.d_optimizer = torch.optim.Adam(self.Dis.parameters(), lr=args.lr, betas=(0.5, 0.999))

        self.s_lr_scheduler = torch.optim.lr_scheduler.CyclicLR(self.s_optimizer,
                                                                base_lr=args.lr / 1000,
                                                                max_lr=args.lr,
                                                                mode='triangular2',
                                                                step_size_up=args.epochs // 6,
                                                                cycle_momentum=False)

        self.d_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=self.d_optimizer,
                                                                lr_lambda=schedulers.LambdaLR(args.epochs, 0,
                                                                                              args.decay_epoch).step)

        # Data loaders for SOURCE DOMAIN (A)
        #####################################################
        transformed_dataset_train_domA = SegmentationDataLoader(
            csv_file=os.path.join(args.csv_dir, args.train_A_csv),
            root_dir=args.root_dir_dHCP,
            shuffle=True, is_augment=args.is_augment,
            transform=transforms.Compose([SplitLab(args.n_classes),
                                          RandomCrop(output_size=(args.crop_height,
                                                                  args.crop_width,
                                                                  args.crop_depth),
                                                     is_random=True, n_classes=args.n_classes),
                                          ToTensor()]))

        transformed_dataset_valid_domA = SegmentationDataLoader(
            csv_file=os.path.join(args.csv_dir, args.valid_A_csv),
            root_dir=args.root_dir_dHCP,
            shuffle=True, is_augment=False,
            transform=transforms.Compose([SplitLab(args.n_classes),
                                          RandomCrop(output_size=(args.crop_height,
                                                                  args.crop_width,
                                                                  args.crop_depth),
                                                     is_random=True, n_classes=args.n_classes),
                                          ToTensor()]))

        transformed_dataset_test_domA = SegmentationDataLoader(
            csv_file=os.path.join(args.csv_dir, args.test_A_csv),
            root_dir=args.root_dir_dHCP,
            shuffle=False, is_augment=False,
            transform=transforms.Compose([SplitLab(args.n_classes),
                                          RandomCrop(output_size=(args.crop_height,
                                                                  args.crop_width,
                                                                  args.crop_depth),
                                                     is_random=False, n_classes=args.n_classes),
                                          ToTensor()]))

        # Data loaders for TARGET DOMAIN (B)
        #####################################################
        transformed_dataset_train_domB = SegmentationDataLoader(
            csv_file=os.path.join(args.csv_dir, args.train_B_csv),
            root_dir=args.root_dir_EPRIME,
            shuffle=True, is_augment=False,
            transform=transforms.Compose([SplitLabEPRIME(args.n_classes),
                                          RandomCrop(output_size=(args.crop_height,
                                                                  args.crop_width,
                                                                  args.crop_depth),
                                                     is_random=True, n_classes=args.n_classes),
                                          ToTensor()]))

        transformed_dataset_valid_domB = SegmentationDataLoader(
            csv_file=os.path.join(args.csv_dir, args.valid_B_csv),
            root_dir=args.root_dir_EPRIME,
            shuffle=True, is_augment=False,
            transform=transforms.Compose([SplitLabEPRIME(args.n_classes),
                                          RandomCrop(output_size=(args.crop_height,
                                                                  args.crop_width,
                                                                  args.crop_depth),
                                                     is_random=True, n_classes=args.n_classes),
                                          ToTensor()]))

        transformed_dataset_test_domB = SegmentationDataLoader(
            csv_file=os.path.join(args.csv_dir, args.test_B_csv),
            root_dir=args.root_dir_EPRIME,
            shuffle=False, is_augment=False,
            transform=transforms.Compose([SplitLabEPRIME(args.n_classes),
                                          RandomCrop(output_size=(args.crop_height,
                                                                  args.crop_width,
                                                                  args.crop_depth),
                                                     is_random=False, n_classes=args.n_classes),
                                          ToTensor()]))

        self.dataloaders = {
            'train-A': DataLoader(transformed_dataset_train_domA, batch_size=args.batch_size,
                                  shuffle=True, num_workers=4 * args.batch_size),
            'valid-A': DataLoader(transformed_dataset_valid_domA, batch_size=args.batch_size,
                                  shuffle=True, num_workers=4 * args.batch_size),
            'test-A': DataLoader(transformed_dataset_test_domA, batch_size=1,
                                 shuffle=False, num_workers=1),

            'train-B': DataLoader(transformed_dataset_train_domB, batch_size=args.batch_size,
                                  shuffle=True, num_workers=4 * args.batch_size),
            'valid-B': DataLoader(transformed_dataset_valid_domB, batch_size=args.batch_size,
                                  shuffle=True, num_workers=4 * args.batch_size),
            'test-B': DataLoader(transformed_dataset_test_domB, batch_size=1,
                                 shuffle=False, num_workers=1)
        }

        # Check if results folder exists
        #####################################################
        if not os.path.isdir(args.results_dir):
            os.makedirs(args.results_dir)

        # Try loading checkpoint
        #####################################################
        if not os.path.isdir(args.checkpoint_dir):
            os.makedirs(args.checkpoint_dir)

        self.alpha_max = args.alpha_max
        self.e1 = args.e1
        self.e2 = args.e2

        try:
            ckpt = utils.load_checkpoint('%s/latest.ckpt' % (args.checkpoint_dir))
            self.start_epoch = ckpt['epoch']
            self.losses_train = ckpt['losses_train']
            self.Seg.load_state_dict(ckpt['Seg'])
            self.Dis.load_state_dict(ckpt['Dis'])
            self.s_optimizer.load_state_dict(ckpt['s_optimizer'])
            self.d_optimizer.load_state_dict(ckpt['d_optimizer'])
        except:
            print('    [!] No checkpoint, starting from scratch!')

            self.start_epoch = 0
            self.losses_train = []


    # ============================================================================
    def train(self, args):
        """
        Train the network
        :param args:
        :return:
        """

        # Variables for train
        #####################################################
        best_segmentation_loss = 1e10
        plot_step = 1

        # Train (Go through each epoch
        #####################################################
        for epoch in range(self.start_epoch, args.epochs):

            # Print learning rate for each epoch
            lr = self.s_optimizer.param_groups[0]['lr']
            print("\n")
            print('            LEARNING RATE SEG = %.7f' % lr)
            lr = self.d_optimizer.param_groups[0]['lr']
            print('            LEARNING RATE DIS = %.7f' % lr)
            print("\n")

            # Save time to calculate how long it took
            start_time = time.time()

            # Metrics to store during training
            metrics = {'dis_loss_train': [], 'adv_loss_train': [], 'seg_loss_train': [], 'total_loss_train': [],
                       'adv_loss_valid': [], 'seg_loss_valid': [], 'total_loss_valid': [],
                       'acc_train': [], 'acc_valid': [], 'lr': [lr]}

            # Set plotted to false at the start of each epoch
            plotted = False

            # Schedule the training
            if epoch < self.e1:
                alpha = 0
            elif epoch <= self.e2:
                alpha = self.alpha_max * (epoch - self.e1) / (self.e2 - self.e1)
            else:
                alpha = self.alpha_max

            print('            ALPHA = %.7f' % alpha)

            # For each epoch set the validation losses to 0
            seg_loss_valid = 0.0
            acc_loss_valid = 0.0

            # Go through each data point TRAIN/VALID
            #####################################################
            for phase in ['train', 'valid']:

                for i, (data_point_a, data_point_b) in enumerate(zip(self.dataloaders[phase+'-A'],
                                                                     self.dataloaders[phase+'-B'])):

                    # step
                    len_dataloader = np.minimum(len(self.dataloaders[phase+'-A']),
                                                len(self.dataloaders[phase+'-B']))

                    if i >= (len_dataloader - 1):
                        break

                    step = epoch * len_dataloader + i + 1

                    # Fetch some data
                    ##################################################
                    t2w_gt_a = utils.cuda(Variable(data_point_a['image']))
                    seg_gt_a = utils.cuda(Variable(data_point_a['lab']))

                    t2w_gt_b = utils.cuda(Variable(data_point_b['image']))   # EPRIME
                    seg_gt_b = data_point_b['lab']

                    # TRAIN
                    ##################################################
                    if phase == 'train':

                        # # Uncomment for quick check
                        # if i > 5:
                        #     break

                        ############################################################################
                        # The discriminator is TRAIN, the segmentor is EVAL
                        ############################################################################
                        self.Seg.eval()
                        self.Dis.train()

                        # The input of the discriminator is composed of batches of
                        # Target and Source Domain, so the idea is to classify them
                        ######################################
                        inputs_model_adv_a = utils.cuda(Variable(t2w_gt_a))  # dHCP
                        inputs_model_adv_b = utils.cuda(Variable(t2w_gt_b))  # ePRIME

                        # We need the U-NET branches to be able to classify / to feed to the discriminator
                        # This is equivalent to get h(x) from the model (basically takes the 4 last
                        # decoder representations from the U-net)
                        # Input to the unet is the paired source/target patches
                        # and outputs the decoder arms
                        ######################################

                        # For domain A
                        ######################################
                        _, dec4, dec3, dec2, dec1 = self.Seg(inputs_model_adv_a, True)
                        # Which are then interpolated to the same size
                        dec1 = F.interpolate(dec1, size=dec3.size()[2:], mode='trilinear', align_corners=False)
                        dec2 = F.interpolate(dec2, size=dec3.size()[2:], mode='trilinear', align_corners=False)
                        dec3 = F.interpolate(dec3, size=dec3.size()[2:], mode='trilinear', align_corners=False)
                        dec4 = F.interpolate(dec4, size=dec3.size()[2:], mode='trilinear', align_corners=False)
                        # Concatenate them
                        inputs_discriminator_a = torch.cat((dec1, dec2, dec3, dec4), 1)

                        # For domain B
                        ######################################
                        _, dec4, dec3, dec2, dec1 = self.Seg(inputs_model_adv_b, True)
                        # Which are then interpolated to the same size
                        dec1 = F.interpolate(dec1, size=dec3.size()[2:], mode='trilinear', align_corners=False)
                        dec2 = F.interpolate(dec2, size=dec3.size()[2:], mode='trilinear', align_corners=False)
                        dec3 = F.interpolate(dec3, size=dec3.size()[2:], mode='trilinear', align_corners=False)
                        dec4 = F.interpolate(dec4, size=dec3.size()[2:], mode='trilinear', align_corners=False)
                        # Concatenate them
                        inputs_discriminator_b = torch.cat((dec1, dec2, dec3, dec4), 1)

                        # Forward pass through DISCRIMINATOR
                        ######################################
                        outputs_discriminator_a = self.Dis(utils.cuda(Variable(inputs_discriminator_a)))
                        outputs_discriminator_b = self.Dis(utils.cuda(Variable(inputs_discriminator_b)))

                        # Labels discriminator
                        # We generate the labels to train the discriminator (0: for source domain 1: target domain)
                        ######################################
                        labels_discriminator_a = utils.cuda(Variable(torch.zeros_like(
                                                            outputs_discriminator_a[:, 0, :, :, :]).type(torch.LongTensor)))
                        labels_discriminator_b = utils.cuda(Variable(torch.ones_like(
                                                            outputs_discriminator_b[:, 0, :, :, :]).type(torch.LongTensor)))

                        # DISCRIMINATOR loss
                        ######################################
                        dis_loss = (self.CE(outputs_discriminator_a, labels_discriminator_a) +
                                    self.CE(outputs_discriminator_b, labels_discriminator_b) ) * 0.5 * args.lamda_adv

                        # Store metrics
                        ######################################
                        metrics['dis_loss_train'].append(dis_loss.item())

                        # Update
                        ######################################
                        self.d_optimizer.zero_grad()
                        dis_loss.backward()
                        self.d_optimizer.step()


                        ############################################################################
                        # The segmentor is TRAIN, the discriminator is EVAL
                        ############################################################################
                        self.Dis.eval()
                        self.Seg.train()

                        # Do a forward pass on the inputs model adversarial
                        # (Target and Source Domain separate)
                        ######################################

                        # For domain A
                        ######################################
                        pred_seg, dec4, dec3, dec2, dec1 = self.Seg(inputs_model_adv_a, True)

                        # Which are then interpolated to the same size
                        dec1 = F.interpolate(dec1, size=dec3.size()[2:], mode='trilinear', align_corners=False)
                        dec2 = F.interpolate(dec2, size=dec3.size()[2:], mode='trilinear', align_corners=False)
                        dec3 = F.interpolate(dec3, size=dec3.size()[2:], mode='trilinear', align_corners=False)
                        dec4 = F.interpolate(dec4, size=dec3.size()[2:], mode='trilinear', align_corners=False)
                        # Concatenate them
                        inputs_discriminator_a = torch.cat((dec1, dec2, dec3, dec4), 1)

                        # For domain A
                        ######################################
                        _, dec4, dec3, dec2, dec1 = self.Seg(inputs_model_adv_b, True)

                        # Which are then interpolated to the same size
                        dec1 = F.interpolate(dec1, size=dec3.size()[2:], mode='trilinear', align_corners=False)
                        dec2 = F.interpolate(dec2, size=dec3.size()[2:], mode='trilinear', align_corners=False)
                        dec3 = F.interpolate(dec3, size=dec3.size()[2:], mode='trilinear', align_corners=False)
                        dec4 = F.interpolate(dec4, size=dec3.size()[2:], mode='trilinear', align_corners=False)
                        # Concatenate them
                        inputs_discriminator_b = torch.cat((dec1, dec2, dec3, dec4), 1)

                        # Forward pass through DISCRIMINATOR
                        ######################################
                        outputs_discriminator_a = self.Dis(utils.cuda(Variable(inputs_discriminator_a)))
                        outputs_discriminator_b = self.Dis(utils.cuda(Variable(inputs_discriminator_b)))

                        # Labels discriminator
                        # We generate the labels to train the discriminator (0: for source domain 1: target domain)
                        ######################################
                        labels_discriminator_a = utils.cuda(Variable(torch.zeros_like(
                                                            outputs_discriminator_a[:, 0, :, :, :]).type(torch.LongTensor)))
                        labels_discriminator_b = utils.cuda(Variable(torch.ones_like(
                                                            outputs_discriminator_b[:, 0, :, :, :]).type(torch.LongTensor)))

                        # DISCRIMINATOR loss
                        ######################################
                        loss_adv = (self.CE(outputs_discriminator_a, labels_discriminator_a) +
                                    self.CE(outputs_discriminator_b, labels_discriminator_b) ) * 0.5 * args.lamda_adv

                        # Calculate dice score between predicted segmentation and ground truth segmentation
                        ######################################
                        seg_loss = self.GDL(torch.softmax(pred_seg, dim=1, dtype=torch.float32),
                                            seg_gt_a, include_background=True) * args.lamda_seg

                        # Calculate accuracy
                        ######################################
                        acc_adv_a = ((torch.argmax(torch.softmax(outputs_discriminator_a, dim=1, dtype=torch.float32),
                                                 dim=1) - labels_discriminator_a) == 0).sum().float() / \
                                   labels_discriminator_a.view(-1, 1).shape[0]
                        acc_adv_b = ((torch.argmax(torch.softmax(outputs_discriminator_b, dim=1, dtype=torch.float32),
                                                 dim=1) - labels_discriminator_b) == 0).sum().float() / \
                                   labels_discriminator_b.view(-1, 1).shape[0]
                        acc_adv = (acc_adv_a + acc_adv_b) * 0.5

                        # Total loss
                        ######################################
                        total_loss = seg_loss - alpha * loss_adv

                        # Store metrics
                        ######################################
                        metrics['adv_loss_train'].append(loss_adv.item())
                        metrics['seg_loss_train'].append(seg_loss.item())
                        metrics['total_loss_train'].append(total_loss.item())
                        metrics['acc_train'].append(acc_adv.item())

                        # Update
                        ######################################
                        self.s_optimizer.zero_grad()
                        total_loss.backward()
                        self.s_optimizer.step()

                    # VALIDATE
                    #######################################################
                    else:
                        self.Seg.eval()
                        self.Dis.eval()

                        inputs_model_adv_a = utils.cuda(Variable(t2w_gt_a))  # dHCP
                        inputs_model_adv_b = utils.cuda(Variable(t2w_gt_b))  # ePRIME

                        with torch.no_grad():
                            # Forward pass through SEGMENTOR
                            ######################################


                            # Forward pass on the inputs model adversarial
                            # (Target and Source Domain separate)
                            ######################################

                            # For domain A
                            ######################################
                            pred_seg, dec4, dec3, dec2, dec1 = self.Seg(inputs_model_adv_a, True)

                            # Which are then interpolated to the same size
                            dec1 = F.interpolate(dec1, size=dec3.size()[2:], mode='trilinear', align_corners=False)
                            dec2 = F.interpolate(dec2, size=dec3.size()[2:], mode='trilinear', align_corners=False)
                            dec3 = F.interpolate(dec3, size=dec3.size()[2:], mode='trilinear', align_corners=False)
                            dec4 = F.interpolate(dec4, size=dec3.size()[2:], mode='trilinear', align_corners=False)

                            # Concatenate them
                            inputs_discriminator_a = torch.cat((dec1, dec2, dec3, dec4), 1)

                            # For domain B
                            ######################################
                            pred_seg_b, dec4, dec3, dec2, dec1 = self.Seg(inputs_model_adv_b, True)
                            # Which are then interpolated to the same size
                            dec1 = F.interpolate(dec1, size=dec3.size()[2:], mode='trilinear', align_corners=False)
                            dec2 = F.interpolate(dec2, size=dec3.size()[2:], mode='trilinear', align_corners=False)
                            dec3 = F.interpolate(dec3, size=dec3.size()[2:], mode='trilinear', align_corners=False)
                            dec4 = F.interpolate(dec4, size=dec3.size()[2:], mode='trilinear', align_corners=False)
                            # Concatenate them
                            inputs_discriminator_b = torch.cat((dec1, dec2, dec3, dec4), 1)

                            # Forward pass through DISCRIMINATOR
                            ######################################
                            outputs_discriminator_a = self.Dis(utils.cuda(Variable(inputs_discriminator_a)))
                            outputs_discriminator_b = self.Dis(utils.cuda(Variable(inputs_discriminator_b)))

                            # Labels discriminator
                            # We generate the labels to train the discriminator (0: for source domain 1: target domain)
                            ######################################
                            labels_discriminator_a = utils.cuda(Variable(torch.zeros_like(
                                outputs_discriminator_a[:, 0, :, :, :]).type(torch.LongTensor)))
                            labels_discriminator_b = utils.cuda(Variable(torch.ones_like(
                                outputs_discriminator_b[:, 0, :, :, :]).type(torch.LongTensor)))

                            # DISCRIMINATOR loss
                            ######################################
                            loss_adv = (self.CE(outputs_discriminator_a, labels_discriminator_a) +
                                        self.CE(outputs_discriminator_b, labels_discriminator_b)) * 0.5 * args.lamda_adv

                            # Calculate dice score between predicted segmentation and ground truth segmentation
                            ######################################
                            seg_loss = self.GDL(torch.softmax(pred_seg, dim=1, dtype=torch.float32),
                                                seg_gt_a, include_background=True) * args.lamda_seg

                            # Calculate Accuracy
                            ######################################
                            acc_adv_a = ((torch.argmax(
                                torch.softmax(outputs_discriminator_a, dim=1, dtype=torch.float32),
                                dim=1) - labels_discriminator_a) == 0).sum().float() / \
                                        labels_discriminator_a.view(-1, 1).shape[0]
                            acc_adv_b = ((torch.argmax(
                                torch.softmax(outputs_discriminator_b, dim=1, dtype=torch.float32),
                                dim=1) - labels_discriminator_b) == 0).sum().float() / \
                                        labels_discriminator_b.view(-1, 1).shape[0]
                            acc_adv = (acc_adv_a + acc_adv_b) * 0.5

                            # Total loss
                            ######################################
                            total_loss = seg_loss - alpha * loss_adv

                            # Store metrics
                            ######################################
                            metrics['adv_loss_valid'].append(loss_adv.item())
                            metrics['seg_loss_valid'].append(seg_loss.item())
                            metrics['total_loss_valid'].append(total_loss.item())
                            metrics['acc_valid'].append(acc_adv.item())

                            # Save valid losses here
                            ######################################
                            seg_loss_valid += seg_loss.item()
                            acc_loss_valid += acc_adv.item()

                        # Plot some images
                        #######################################################
                        if epoch % plot_step == 0 and not plotted:
                            plotted = True

                            pred_seg_dHCP = torch.softmax(pred_seg, dim=1, dtype=torch.float32)
                            pred_seg_EPRIME = torch.softmax(pred_seg_b, dim=1, dtype=torch.float32)

                            utils.plot_seg_latent(args, epoch,
                                                  t2w_gt_a, seg_gt_a, pred_seg_dHCP,
                                                  t2w_gt_b, seg_gt_b, pred_seg_EPRIME)

                        # Save best after all validation steps
                        #######################################################
                        if i == (args.validation_steps - 1):
                            seg_loss_valid /= args.validation_steps
                            acc_loss_valid /= args.validation_steps

                            print("\n")
                            print(f"        > Average Seg Loss     {seg_loss_valid:6.3f} ")
                            print(f"        > Average Acc Loss     {acc_loss_valid:6.3f} ")

                            # Save best
                            ######################################
                            if best_segmentation_loss > seg_loss_valid and \
                                    acc_loss_valid < 0.8 and epoch > 0:

                                best_segmentation_loss = seg_loss_valid
                                print(f"        > Best Seg Loss So Far {best_segmentation_loss:6.3f} ")
                                print(f"        > Current Accuracy     {acc_loss_valid:6.3f} ")

                                # Override the latest checkpoint for best generator loss
                                ######################################
                                utils.save_checkpoint({'epoch': epoch + 1,
                                                       'Seg': self.Seg.state_dict(),
                                                       'Dis': self.Dis.state_dict(),
                                                       'd_optimizer': self.d_optimizer.state_dict(),
                                                       's_optimizer': self.s_optimizer.state_dict()},
                                                      '%s/latest_best_loss.ckpt' % (args.checkpoint_dir))

                                # Write in a file
                                ######################################
                                with open('%s/README' % (args.checkpoint_dir), 'w') as f:
                                    f.write('Epoch: %d | Seg Loss: %f | Acc Loss: %f \n' % (
                                        epoch + 1, seg_loss_valid, acc_loss_valid))

                            # Stop early -- Don't go through all the validation set
                            break

                    # PRINT STATS
                    ###################################################
                    time_elapsed = time.time() - start_time
                    print("              %s Epoch: (%3d) (%5d/%5d) (%3d) | Acc Loss:%.2e | Adv Loss:%.2e | Seg Loss:%.2e | %.0fm %.2fs" %
                          (phase.upper(), epoch, i + 1, len_dataloader, step,
                           acc_adv, loss_adv, seg_loss, time_elapsed // 60, time_elapsed % 60))

            # Append the metrics to losses_train
            ######################################
            self.losses_train.append(metrics)

            # Override the latest checkpoint at the end of an epoch
            #######################################################
            utils.save_checkpoint({'epoch': epoch + 1,
                                   'Seg': self.Seg.state_dict(),
                                   'Dis': self.Dis.state_dict(),
                                   's_optimizer': self.s_optimizer.state_dict(),
                                   'd_optimizer': self.d_optimizer.state_dict(),
                                   'losses_train': self.losses_train},
                                  '%s/latest.ckpt' % (args.checkpoint_dir))

            # Update learning rates
            ######################################
            self.s_lr_scheduler.step()
            self.d_lr_scheduler.step()

        return self.losses_train


# ==================================================================================================================== #
#
#  3D U-NET for segmentation of source data with UDA in image space
#
# ==================================================================================================================== #
class Segmentation3DImage(object):
    """
    Class for UDA with image space
    """
    # ============================================================================
    def __init__(self, args):

        # Define the network
        #####################################################
        self.Seg = define_Net(input_nc=1, output_nc=args.n_classes,
                              n_features=args.seg_features, net_name=args.seg_net)

        self.Gen = define_Net(input_nc=1, output_nc=1,
                              n_features=args.ngf, net_name=args.gen_net)

        self.Dis = define_Net(input_nc=1, output_nc=1,
                              n_features=args.ndf, net_name=args.dis_net)

        utils.print_networks([self.Seg, self.Gen, self.Dis], ['Seg', 'Gen', 'Dis'])


        # print("-------------------")
        # print(self.Seg)
        # print("-------------------\n\n")
        #
        # print("-------------------")
        # print(self.Gen)
        # print("-------------------\n\n")
        #
        # print("-------------------")
        # print(self.Dis)
        # print("-------------------\n\n")


        # Define Loss criterias
        #####################################################
        self.GDL = GeneralisedDiceLoss
        self.NCC = NCCLoss
        self.MSE = nn.MSELoss()
        self.L1 = nn.L1Loss()
        self.SmoothL1 = nn.SmoothL1Loss()

        # Optimizers
        #####################################################
        self.s_optimizer = torch.optim.Adam(self.Seg.parameters(), lr=args.lr)
        self.d_optimizer = torch.optim.Adam(self.Dis.parameters(), lr=args.lr/10.0,
                                            betas=(0.5, 0.999), weight_decay=1e-5)
        self.g_optimizer = torch.optim.Adam(self.Gen.parameters(), lr=args.lr/10.0,
                                            betas=(0.5, 0.999), weight_decay=1e-5)

        self.s_lr_scheduler = torch.optim.lr_scheduler.CyclicLR(self.s_optimizer,
                                                                base_lr=args.lr / 100,  # args.lr / 1000,
                                                                max_lr=args.lr,
                                                                mode='triangular2',
                                                                step_size_up=args.epochs // 6,
                                                                cycle_momentum=False)
        self.d_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=self.d_optimizer,
                                                                lr_lambda=schedulers.LambdaLR(args.epochs, 0,
                                                                                              args.decay_epoch).step)
        self.g_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=self.g_optimizer,
                                                                lr_lambda=schedulers.LambdaLR(args.epochs, 0,
                                                                                              args.decay_epoch).step)

        # Data loaders for SOURCE DOMAIN (A)
        #####################################################
        transformed_dataset_train_domA = SegmentationDataLoader(
            csv_file=os.path.join(args.csv_dir, args.train_A_csv),
            root_dir=args.root_dir_dHCP,
            shuffle=True, is_augment=args.is_augment,
            transform=transforms.Compose([SplitLab(args.n_classes),
                                          RandomCrop(output_size=(args.crop_height,
                                                                  args.crop_width,
                                                                  args.crop_depth),
                                                     is_random=True, n_classes=args.n_classes),
                                          ToTensor()]))

        transformed_dataset_valid_domA = SegmentationDataLoader(
            csv_file=os.path.join(args.csv_dir, args.valid_A_csv),
            root_dir=args.root_dir_dHCP,
            shuffle=True, is_augment=False,
            transform=transforms.Compose([SplitLab(args.n_classes),
                                          RandomCrop(output_size=(args.crop_height,
                                                                  args.crop_width,
                                                                  args.crop_depth),
                                                     is_random=True, n_classes=args.n_classes),
                                          ToTensor()]))

        transformed_dataset_test_domA = SegmentationDataLoader(
            csv_file=os.path.join(args.csv_dir, args.test_A_csv),
            root_dir=args.root_dir_dHCP,
            shuffle=False, is_augment=False,
            transform=transforms.Compose([SplitLab(args.n_classes),
                                          RandomCrop(output_size=(args.crop_height,
                                                                  args.crop_width,
                                                                  args.crop_depth),
                                                     is_random=False, n_classes=args.n_classes),
                                          ToTensor()]))

        # Data loaders for TARGET DOMAIN (B)
        #####################################################
        transformed_dataset_train_domB = SegmentationDataLoader(
            csv_file=os.path.join(args.csv_dir, args.train_B_csv),
            root_dir=args.root_dir_EPRIME,
            shuffle=True, is_augment=False,
            transform=transforms.Compose([SplitLabEPRIME(args.n_classes),
                                          RandomCrop(output_size=(args.crop_height,
                                                                  args.crop_width,
                                                                  args.crop_depth),
                                                     is_random=True, n_classes=args.n_classes),
                                          ToTensor()]))

        transformed_dataset_valid_domB = SegmentationDataLoader(
            csv_file=os.path.join(args.csv_dir, args.valid_B_csv),
            root_dir=args.root_dir_EPRIME,
            shuffle=True, is_augment=False,
            transform=transforms.Compose([SplitLabEPRIME(args.n_classes),
                                          RandomCrop(output_size=(args.crop_height,
                                                                  args.crop_width,
                                                                  args.crop_depth),
                                                     is_random=True, n_classes=args.n_classes),
                                          ToTensor()]))

        transformed_dataset_test_domB = SegmentationDataLoader(
            csv_file=os.path.join(args.csv_dir, args.test_B_csv),
            root_dir=args.root_dir_EPRIME,
            shuffle=False, is_augment=False,
            transform=transforms.Compose([SplitLabEPRIME(args.n_classes),
                                          RandomCrop(output_size=(args.crop_height,
                                                                  args.crop_width,
                                                                  args.crop_depth),
                                                     is_random=False, n_classes=args.n_classes),
                                          ToTensor()]))


        self.dataloaders = {
            'train-A': DataLoader(transformed_dataset_train_domA, batch_size=args.batch_size,
                                  shuffle=True, num_workers=8 * args.batch_size),
            'valid-A': DataLoader(transformed_dataset_valid_domA, batch_size=args.batch_size,
                                  shuffle=True, num_workers=8 * args.batch_size),
            'test-A': DataLoader(transformed_dataset_test_domA, batch_size=1,
                                 shuffle=False, num_workers=1),

            'train-B': DataLoader(transformed_dataset_train_domB, batch_size=args.batch_size,
                                  shuffle=True, num_workers=8 * args.batch_size),
            'valid-B': DataLoader(transformed_dataset_valid_domB, batch_size=args.batch_size,
                                  shuffle=True, num_workers=8 * args.batch_size),
            'test-B': DataLoader(transformed_dataset_test_domB, batch_size=1,
                                 shuffle=False, num_workers=1),
        }

        # Check if results folder exists
        #####################################################
        if not os.path.isdir(args.results_dir):
            os.makedirs(args.results_dir)

        # Try loading checkpoint
        #####################################################
        if not os.path.isdir(args.checkpoint_dir):
            os.makedirs(args.checkpoint_dir)

        try:
            ckpt = utils.load_checkpoint('%s/latest.ckpt' % (args.checkpoint_dir))
            self.start_epoch = ckpt['epoch']
            self.losses_train = ckpt['losses_train']
            self.Seg.load_state_dict(ckpt['Seg'])
            self.Gen.load_state_dict(ckpt['Gen'])
            self.Dis.load_state_dict(ckpt['Dis'])
            self.s_optimizer.load_state_dict(ckpt['s_optimizer'])
            self.d_optimizer.load_state_dict(ckpt['d_optimizer'])
            self.g_optimizer.load_state_dict(ckpt['g_optimizer'])
        except:
            print('    [!] No checkpoint, starting from scratch!')

            self.start_epoch = 0
            self.losses_train = []


    # ============================================================================
    def train(self, args):
        """
        Train the network
        :param args:
        :return:
        """

        # Variables for train
        #####################################################
        best_segmentation_loss = 1e10
        best_generator_loss = 1e10
        a_fake_sample = utils.Sample_from_Pool()
        plot_step = 1

        # Train (Go through each epoch
        #####################################################
        for epoch in range(self.start_epoch, args.epochs):

            # Print learning rate for each epoch
            lr = self.s_optimizer.param_groups[0]['lr']
            print("\n")
            print('            LEARNING RATE SEG = %.7f' % lr)
            lr = self.d_optimizer.param_groups[0]['lr']
            print('            LEARNING RATE DIS = %.7f' % lr)
            lr = self.g_optimizer.param_groups[0]['lr']
            print('            LEARNING RATE GEN = %.7f' % lr)
            print("\n")

            # Save time to calculate how long it took
            start_time = time.time()

            # Metrics to store during training
            metrics = {'seg_loss_train': [], 'adv_loss_train': [], 'gen_loss_train': [], 'ncc_loss_train': [],
                       'dis_loss_train': [],
                       'seg_loss_valid': [], 'adv_loss_valid': [], 'gen_loss_valid': [], 'ncc_loss_valid': [],
                       'dis_loss_valid': []}

            # Set plotted to false at the start of each epoch
            plotted = False

            # For each epoch set the validation losses to 0
            seg_loss_valid = 0.0
            gen_loss_valid = 0.0

            # Go through each data point TRAIN/VALID
            #####################################################
            for phase in ['train', 'valid']:

                for i, (data_point_a, data_point_b) in enumerate(zip(self.dataloaders[phase+'-A'],
                                                                     self.dataloaders[phase+'-B'])):

                    # step
                    len_dataloader = np.minimum(len(self.dataloaders[phase+'-A']),
                                                len(self.dataloaders[phase+'-B']))

                    step = epoch * len_dataloader + i + 1

                    # Fetch some data
                    ##################################################
                    t2w_gt_a = utils.cuda(Variable(data_point_a['image']))
                    seg_gt_a = utils.cuda(Variable(data_point_a['lab']))

                    t2w_gt_b = utils.cuda(Variable(data_point_b['image']))   # EPRIME

                    # TRAIN
                    ##################################################
                    if phase == 'train':

                        # # Uncomment for quick check
                        if i > 5:
                            break

                        ##################################################
                        ##########################  Generator Computations
                        ##################################################
                        set_grad([self.Dis], False)
                        self.g_optimizer.zero_grad()
                        self.s_optimizer.zero_grad()

                        # Forward pass through generator
                        ##################################################
                        t2w_a_fake = self.Gen(t2w_gt_a)

                        # Forward pass through segmentation network
                        ##################################################
                        seg_a_real_pred = torch.softmax(self.Seg(t2w_gt_a), dim=1, dtype=torch.float32)
                        seg_a_fake_pred = torch.softmax(self.Seg(t2w_a_fake), dim=1, dtype=torch.float32)

                        # NCC Loss between input and output
                        ###################################################
                        if epoch < 5:
                            ncc_loss_pred = self.NCC(t2w_a_fake, t2w_gt_a) * args.lamda_ncc * 1000.0
                        else:
                            ncc_loss_pred = self.NCC(t2w_a_fake, t2w_gt_a) * args.lamda_ncc

                        # Adversarial losses
                        ###################################################
                        t2w_a_fake_dis = self.Dis(t2w_a_fake)
                        real_label = utils.cuda(Variable(torch.ones(t2w_a_fake_dis.size())))
                        adv_gen_loss = self.MSE(t2w_a_fake_dis, real_label)

                        # Dice Loss
                        ###################################################
                        if epoch < 5:
                            seg_loss = self.GDL(seg_a_real_pred, seg_gt_a, include_background=True) * args.lamda_seg
                        else:
                            seg_loss = (self.GDL(seg_a_fake_pred, seg_gt_a, include_background=True) +
                                        self.GDL(seg_a_real_pred, seg_gt_a, include_background=True)) * 0.5 * args.lamda_seg

                        # Total generators losses
                        ###################################################
                        gen_loss = adv_gen_loss + seg_loss + ncc_loss_pred

                        # Store metrics
                        metrics['ncc_loss_train'].append(ncc_loss_pred.item())
                        metrics['seg_loss_train'].append(seg_loss.item())
                        metrics['adv_loss_train'].append(adv_gen_loss.item())
                        metrics['gen_loss_train'].append(gen_loss.item())

                        # Update generators
                        ###################################################
                        gen_loss.backward()
                        self.g_optimizer.step()
                        self.s_optimizer.step()

                        #################################################
                        #####################  Discriminator Computations
                        #################################################
                        set_grad([self.Dis], True)
                        self.d_optimizer.zero_grad()

                        # Sample from history of generated images
                        #################################################
                        t2w_a_fake = Variable(torch.Tensor(a_fake_sample([t2w_a_fake.cpu().data.numpy()])[0]))
                        t2w_a_fake = utils.cuda(t2w_a_fake)

                        # Forward pass through discriminator
                        #################################################
                        t2w_a_fake_dis = self.Dis(t2w_a_fake)
                        t2w_b_real_dis = self.Dis(t2w_gt_b)
                        real_label = utils.cuda(Variable(torch.ones(t2w_b_real_dis.size())))
                        fake_label = utils.cuda(Variable(torch.zeros(t2w_a_fake_dis.size())))

                        # Discriminator losses
                        ##################################################
                        dis_real_loss = self.MSE(t2w_b_real_dis, real_label)
                        dis_fake_loss = self.MSE(t2w_a_fake_dis, fake_label)

                        # Total discriminators losses
                        dis_loss = (dis_real_loss + dis_fake_loss) * 0.5

                        # Store metrics
                        metrics['dis_loss_train'].append(dis_loss.item())

                        # Update discriminators
                        ##################################################
                        dis_loss.backward()
                        self.d_optimizer.step()

                    # VALIDATE
                    #######################################################
                    else:
                        self.Seg.eval()
                        self.Gen.eval()
                        self.Dis.eval()

                        with torch.no_grad():
                            # Forward pass through generator
                            ##################################################
                            t2w_a_fake = self.Gen(t2w_gt_a)

                            # Forward pass through segmentation network
                            ##################################################
                            input_seg = torch.cat([t2w_a_fake, t2w_gt_a], dim=0)
                            output_seg = torch.cat([seg_gt_a, seg_gt_a], dim=0)
                            seg_a_fake_pred = torch.softmax(self.Seg(input_seg), dim=1, dtype=torch.float32)

                            # NCC Loss between input and output
                            ###################################################
                            ncc_loss_pred = self.NCC(t2w_a_fake, t2w_gt_a) * args.lamda_ncc

                            # Adversarial losses
                            ###################################################
                            t2w_a_fake_dis = self.Dis(t2w_a_fake)
                            real_label = utils.cuda(Variable(torch.ones(t2w_a_fake_dis.size())))
                            adv_gen_loss = self.MSE(t2w_a_fake_dis, real_label)

                            # Dice Loss
                            ###################################################
                            seg_loss = self.GDL(seg_a_fake_pred, output_seg, include_background=True) * args.lamda_seg

                            # Total generator loss
                            ###################################################
                            gen_loss = adv_gen_loss + seg_loss + ncc_loss_pred

                            # Forward pass through discriminator
                            #################################################
                            t2w_a_fake_dis = self.Dis(t2w_a_fake)
                            t2w_b_real_dis = self.Dis(t2w_gt_b)
                            real_label = utils.cuda(Variable(torch.ones(t2w_b_real_dis.size())))
                            fake_label = utils.cuda(Variable(torch.zeros(t2w_a_fake_dis.size())))

                            # Discriminator losses
                            ##################################################
                            dis_real_loss = self.MSE(t2w_b_real_dis, real_label)
                            dis_fake_loss = self.MSE(t2w_a_fake_dis, fake_label)

                            # Total discriminators losses
                            dis_loss = (dis_real_loss + dis_fake_loss) * 0.5

                            metrics['ncc_loss_valid'].append(ncc_loss_pred.item())
                            metrics['seg_loss_valid'].append(seg_loss.item())
                            metrics['adv_loss_valid'].append(adv_gen_loss.item())
                            metrics['gen_loss_valid'].append(gen_loss.item())
                            metrics['dis_loss_valid'].append(dis_loss.item())

                            # Save valid losses here:
                            seg_loss_valid += seg_loss.item()
                            gen_loss_valid += gen_loss.item()

                        # Plot some images
                        #######################################################
                        if epoch % plot_step == 0 and not plotted:
                            plotted = True
                            utils.plot_seg_img_fake(args, epoch,
                                                    seg_gt_a, seg_a_fake_pred,
                                                    t2w_gt_a, t2w_a_fake)

                        # Save best after all validation steps
                        #######################################################
                        if i == (args.validation_steps - 1):
                            seg_loss_valid /= args.validation_steps
                            gen_loss_valid /= args.validation_steps

                            print("\n")
                            print(f"        > Average Seg Loss     {seg_loss_valid:6.3f} ")
                            print(f"        > Average Gen Loss     {gen_loss_valid:6.3f} ")

                            # Save best
                            if best_segmentation_loss > seg_loss_valid and \
                                    best_generator_loss > gen_loss_valid and epoch > 0:

                                best_segmentation_loss = seg_loss_valid
                                print(f"        > Best Seg Loss So Far {best_segmentation_loss:6.3f} ")

                                best_generator_loss = gen_loss_valid
                                print(f"        > Best Gen Loss So Far {best_generator_loss:6.3f} ")

                                # Override the latest checkpoint for best generator loss
                                utils.save_checkpoint({'epoch': epoch + 1,
                                                       'Seg': self.Seg.state_dict(),
                                                       'Gen': self.Gen.state_dict(),
                                                       'Dis': self.Dis.state_dict(),
                                                       'g_optimizer': self.g_optimizer.state_dict(),
                                                       'd_optimizer': self.d_optimizer.state_dict(),
                                                       's_optimizer': self.s_optimizer.state_dict()},
                                                      '%s/latest_best_loss.ckpt' % (args.checkpoint_dir))

                                # Write in a file
                                with open('%s/README' % (args.checkpoint_dir), 'w') as f:
                                    f.write('Epoch: %d | Seg Loss: %f | Gen Loss: %f \n' % (
                                        epoch + 1, seg_loss_valid, gen_loss_valid))

                            # Stop early -- Don't go through all the validation set
                            break

                    # PRINT STATS
                    ###################################################
                    time_elapsed = time.time() - start_time
                    print("              %s Epoch: (%3d) (%5d/%5d) (%3d) | Gen Loss:%.2e | Dis Loss:%.2e | Seg Loss:%.2e | NCC Loss:%.2e | %.0fm %.2fs" %
                          (phase.upper(), epoch, i + 1, len_dataloader, step,
                           gen_loss, dis_loss, seg_loss, ncc_loss_pred.item(), time_elapsed // 60, time_elapsed % 60))

            # Append the metrics to losses_train
            ######################################
            self.losses_train.append(metrics)

            # Override the latest checkpoint at the end of an epoch
            #######################################################
            utils.save_checkpoint({'epoch': epoch + 1,
                                   'Seg': self.Seg.state_dict(),
                                   'Gen': self.Gen.state_dict(),
                                   'Dis': self.Dis.state_dict(),
                                   's_optimizer': self.s_optimizer.state_dict(),
                                   'g_optimizer': self.g_optimizer.state_dict(),
                                   'd_optimizer': self.d_optimizer.state_dict(),
                                   'losses_train': self.losses_train},
                                  '%s/latest.ckpt' % (args.checkpoint_dir))

            # Update learning rates
            ########################
            self.s_lr_scheduler.step()
            self.g_lr_scheduler.step()
            self.d_lr_scheduler.step()

        return self.losses_train


# ==================================================================================================================== #
#
#  3D U-NET for cortical parcellation
#
# ==================================================================================================================== #
class Segmentation3DUNETCortex11(object):
    """
    Class for cortical parcellation network
    """
    # ============================================================================
    def __init__(self, args):

        n_cortex_classes = args.n_classes
        n_input_channels = 2

        # Define the network
        #####################################################
        self.Seg = define_Net(input_nc=n_input_channels, output_nc=n_cortex_classes + 1,  # 11 + 1 background
                              n_features=args.seg_features, net_name=args.seg_net)

        utils.print_networks([self.Seg], ['Seg'])

        # Define Loss criterias
        #####################################################
        self.GDL = GeneralisedDiceLoss

        # Optimizers
        #####################################################
        self.s_optimizer = torch.optim.Adam(self.Seg.parameters(), lr=args.lr)

        self.s_lr_scheduler = torch.optim.lr_scheduler.CyclicLR(self.s_optimizer,
                                                                base_lr=args.lr / 1000,
                                                                max_lr=args.lr,
                                                                mode='triangular2',
                                                                step_size_up=args.epochs // 6,
                                                                cycle_momentum=False)

        # DATA Loaders
        #####################################################
        transformed_dataset_train_domA = SegmentationDataLoaderCortex(
            csv_file=os.path.join(args.csv_dir, args.train_A_csv),
            root_dir=args.root_dir_dHCP,
            shuffle=True,
            is_augment=args.is_augment,
            transform=transforms.Compose([SplitLabCortex(n_input_channels, n_cortex_classes),
                                          RandomCropCortex(output_size=(args.crop_height,
                                                                        args.crop_width,
                                                                        args.crop_depth),
                                                           is_random=False,
                                                           n_classes=n_input_channels),
                                          ToTensor()]))

        transformed_dataset_valid_domA = SegmentationDataLoaderCortex(
            csv_file=os.path.join(args.csv_dir, args.valid_A_csv),
            root_dir=args.root_dir_dHCP,
            shuffle=True,
            is_augment=False,
            transform=transforms.Compose([SplitLabCortex(n_input_channels, n_cortex_classes),
                                          RandomCropCortex(output_size=(args.crop_height,
                                                                        args.crop_width,
                                                                        args.crop_depth),
                                                           is_random=False,
                                                           n_classes=n_input_channels),
                                          ToTensor()]))

        transformed_dataset_test_domA = SegmentationDataLoaderCortex(
            csv_file=os.path.join(args.csv_dir, args.test_A_csv),
            root_dir=args.root_dir_dHCP,
            shuffle=False,
            is_augment=False,
            transform=transforms.Compose([SplitLabCortex(n_input_channels, n_cortex_classes),
                                          RandomCropCortex(output_size=(args.crop_height,
                                                                        args.crop_width,
                                                                        args.crop_depth),
                                                           is_random=False,
                                                           n_classes=n_input_channels),
                                          ToTensor()]))

        self.dataloaders = {
            'train-A': DataLoader(transformed_dataset_train_domA, batch_size=args.batch_size,
                                  shuffle=True, num_workers=4 * args.batch_size),
            'valid-A': DataLoader(transformed_dataset_valid_domA, batch_size=args.batch_size,
                                  shuffle=True, num_workers=4 * args.batch_size),
            'test-A': DataLoader(transformed_dataset_test_domA, batch_size=1,
                                  shuffle=False, num_workers=1)
        }

        # Check if results folder exists
        #####################################################
        if not os.path.isdir(args.results_dir):
            os.makedirs(args.results_dir)

        # Try loading checkpoint
        #####################################################
        if not os.path.isdir(args.checkpoint_dir):
            os.makedirs(args.checkpoint_dir)

        try:
            ckpt = utils.load_checkpoint('%s/latest.ckpt' % (args.checkpoint_dir))
            self.start_epoch = ckpt['epoch']
            self.losses_train = ckpt['losses_train']
            self.Seg.load_state_dict(ckpt['Seg'])
            self.s_optimizer.load_state_dict(ckpt['s_optimizer'])
        except:
            print('    [!] No checkpoint, starting from scratch!')

            self.start_epoch = 0
            self.losses_train = []


    # ============================================================================
    def train(self, args):
        """
        Train the network
        :param args:
        :return:
        """

        # Variables for train
        #####################################################
        best_segmentation_loss = 1e10
        plot_step = 1

        # Train (Go through each epoch
        #####################################################
        for epoch in range(self.start_epoch, args.epochs):

            # Print learning rate for each epoch
            lr = self.s_optimizer.param_groups[0]['lr']
            print("\n")
            print('            LEARNING RATE = %.7f' % lr)
            print("\n")

            # Save time to calculate how long it took
            start_time = time.time()

            # Metrics to store during training
            metrics = {'seg_loss_train': [], 'seg_loss_valid': [], 'lr': [lr]}

            # Set plotted to false at the start of each epoch
            plotted = False

            # For each epoch set the validation losses to 0
            seg_loss_valid = 0.0

            # Go through each data point TRAIN/VALID
            #####################################################
            for phase in ['train-A', 'valid-A']:

                for i, data_point in enumerate(self.dataloaders[phase]):

                    # step
                    len_dataloader = len(self.dataloaders[phase])
                    step = epoch * len_dataloader + i + 1

                    # Fetch some data
                    ##################################################
                    seg_gt_in = utils.cuda(Variable(data_point['lab']))
                    seg_gt_out = utils.cuda(Variable(data_point['lab_cortex']))

                    # TRAIN
                    ##################################################
                    if phase == 'train-A':

                        # # Uncomment for quick check
                        # if i > 5:
                        #     break

                        # Set optimiser to zero grad
                        ##################################################
                        self.s_optimizer.zero_grad()

                        # Forward pass through UNet
                        ##################################################
                        seg_pr = torch.softmax(self.Seg(seg_gt_in), dim=1, dtype=torch.float32)

                        # Dice Loss
                        ###################################################
                        seg_loss = self.GDL(seg_pr, seg_gt_out, include_background=True, weighted=False) * args.lamda_seg

                        metrics['seg_loss_train'].append(seg_loss.item())

                        # Update unet
                        ###################################################
                        seg_loss.backward()
                        self.s_optimizer.step()

                    # VALIDATE
                    #######################################################
                    else:
                        self.Seg.eval()

                        with torch.no_grad():
                            # Forward pass through UNet
                            ##################################################
                            seg_pr = torch.softmax(self.Seg(seg_gt_in), dim=1, dtype=torch.float32)

                            # Dice Loss
                            ###################################################
                            seg_loss = self.GDL(seg_pr, seg_gt_out, include_background=False, weighted=False) * args.lamda_seg
                            metrics['seg_loss_valid'].append(seg_loss.item())

                            # Save valid losses here:
                            seg_loss_valid += seg_loss.item()

                        # Plot some images
                        #######################################################
                        if epoch % plot_step == 0 and not plotted:
                            plotted = True
                            utils.plot_seg_cortex(args, epoch, seg_gt_out, seg_pr, seg_gt_in)

                        # Save best after all validation steps
                        #######################################################
                        if i == (args.validation_steps - 1):
                            seg_loss_valid /= args.validation_steps

                            print("\n")
                            print(f"        > Average Seg Loss     {seg_loss_valid:6.3f} ")

                            # Save best
                            if best_segmentation_loss > seg_loss_valid and epoch > 0:

                                best_segmentation_loss = seg_loss_valid
                                print(f"        > Best Seg Loss So Far {best_segmentation_loss:6.3f} ")

                                # Override the latest checkpoint for best generator loss
                                utils.save_checkpoint({'epoch': epoch + 1,
                                                       'Seg': self.Seg.state_dict(),
                                                       's_optimizer': self.s_optimizer.state_dict()},
                                                      '%s/latest_best_loss.ckpt' % (args.checkpoint_dir))

                                # Write in a file
                                with open('%s/README' % (args.checkpoint_dir), 'w') as f:
                                    f.write('Epoch: %d | Seg Loss: %f \n' % (epoch + 1, seg_loss_valid))

                            # Stop early -- Don't go through all the validation set
                            break

                    # PRINT STATS
                    ###################################################
                    time_elapsed = time.time() - start_time
                    print("              %s Epoch: (%3d) (%5d/%5d) (%3d) | Seg Loss:%.2e | %.0fm %.2fs" %
                          (phase.upper(), epoch, i + 1, len_dataloader, step,
                           seg_loss, time_elapsed // 60, time_elapsed % 60))

            # Append the metrics to losses_train
            ######################################
            self.losses_train.append(metrics)

            # Override the latest checkpoint at the end of an epoch
            #######################################################
            utils.save_checkpoint({'epoch': epoch + 1,
                                   'Seg': self.Seg.state_dict(),
                                   's_optimizer': self.s_optimizer.state_dict(),
                                   'losses_train': self.losses_train},
                                  '%s/latest.ckpt' % (args.checkpoint_dir))

            # Update learning rates
            ########################
            self.s_lr_scheduler.step()

        return self.losses_train
