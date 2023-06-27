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
# networks.py
#
##############################################################################
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np
from functools import partial

ACTIVATION_FUNCTIONS = ['ReLU', 'LeakyReLU', 'ELU', 'PReLU']
ALLOWED_STR = ['b', 'r', 'l', 'e', 'p' 'c', 'i']


# ==================================================================================================================== #
#
#  3D UNet
#
#  SOME BUILDING BLOCKS FROM HERE:
#  Link: https://github.com/wolny/pytorch-3dunet/blob/master/pytorch3dunet/unet3d/buildingblocks.py
#
# ==================================================================================================================== #
def conv3d(in_channels, out_channels, kernel_size, bias, padding):
    return nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)


def create_conv(in_channels, out_channels, kernel_size, order, padding):
    """
    Create convolutional block
    Link: https://github.com/wolny/pytorch-3dunet/blob/master/pytorch3dunet/unet3d/buildingblocks.py

    :param in_channels: number of input channels
    :param out_channels: number of output channels
    :param kernel_size: size of convolutional kernel
    :param order: order of things, e.g.
            'cr' -> conv + ReLU
            'gcr' -> groupnorm + conv + ReLU
            'cl' -> conv + LeakyReLU
            'cli' -> conv + LeakyReLU + instancenorm
            'ce' -> conv + ELU
            'bcr' -> batchnorm + conv + ReLU
    :param padding:
    :return:
    """

    assert 'c' in order, "    [!] Conv layer MUST be present"
    assert order[0] not in 'rle', "    [!] Non-linearity cannot be the first operation in the layer"

    modules = []
    for i, char in enumerate(order):
        if char == 'r':
            modules.append(('ReLU', nn.ReLU(inplace=True)))
        elif char == 'l':
            modules.append(('LeakyReLU', nn.LeakyReLU(negative_slope=0.2, inplace=True)))
        elif char == 'e':
            modules.append(('ELU', nn.ELU(inplace=True)))
        elif char == 'p':
            modules.append(('PReLU', nn.PReLU()))
        elif char == 'c':
            # add learnable bias only in the absence of batchnorm
            bias = not ('b' in order)
            modules.append(('conv', conv3d(in_channels, out_channels, kernel_size, bias, padding=padding)))
        elif char == 'b':
            is_before_conv = i < order.index('c')
            if is_before_conv:
                modules.append(('batchnorm', nn.BatchNorm3d(in_channels)))
            else:
                modules.append(('batchnorm', nn.BatchNorm3d(out_channels)))
        elif char == 'i':
            is_before_conv = i < order.index('c')
            if is_before_conv:
                modules.append(('instancenorm', nn.InstanceNorm3d(in_channels)))
            else:
                modules.append(('instancenorm', nn.InstanceNorm3d(out_channels)))
        else:
            print_error_msg =  f"    [!] Unsupported layer type '{char}'. MUST be one of {ALLOWED_STR}"
            print_error_msg += f" and ALLOWED activations are {ACTIVATION_FUNCTIONS}"
            raise ValueError(print_error_msg)

    return modules


class SingleConv(nn.Sequential):
    """
    Create single convolution
    Link: https://github.com/wolny/pytorch-3dunet/blob/master/pytorch3dunet/unet3d/buildingblocks.py
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, order='cli', padding=1):
        super(SingleConv, self).__init__()

        for name, module in create_conv(in_channels, out_channels, kernel_size, order, padding=padding):
            self.add_module(name, module)


class DoubleConv(nn.Sequential):
    """
    Create double convolution
    Link: https://github.com/wolny/pytorch-3dunet/blob/master/pytorch3dunet/unet3d/buildingblocks.py
    """

    def __init__(self, in_channels, out_channels, encoder, kernel_size=3, order='cli', padding=1):
        super(DoubleConv, self).__init__()
        if encoder:
            # we're in the encoder path
            conv1_in_channels = in_channels
            conv1_out_channels = out_channels // 2
            if conv1_out_channels < in_channels:
                conv1_out_channels = in_channels
            conv2_in_channels, conv2_out_channels = conv1_out_channels, out_channels
        else:
            # we're in the decoder path, decrease the number of channels in the 1st convolution
            conv1_in_channels, conv1_out_channels = in_channels, out_channels
            conv2_in_channels, conv2_out_channels = out_channels, out_channels

        # conv1
        self.add_module('SingleConv1',
                        SingleConv(conv1_in_channels, conv1_out_channels, kernel_size, order, padding=padding))
        # conv2
        self.add_module('SingleConv2',
                        SingleConv(conv2_in_channels, conv2_out_channels, kernel_size, order, padding=padding))


class UNetModel(nn.Module):

    def __init__(self,
                 num_channels,
                 num_classes,
                 n_features,
                 use_activ=False,
                 block='cpi',
                 name='UNet3D'):
        super(UNetModel, self).__init__()

        self.name = name
        print(f"    Network chosen is {name}")

        # Set the attributes
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.use_activ = use_activ

        # Set network parameters:
        self.n_features = n_features  # =[16, 32, 64, 128, 256]
        self.k_size = 3

        # Create network
        self.avgpool = nn.AvgPool3d(kernel_size=2)
        self.maxpool = nn.MaxPool3d(kernel_size=2)

        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        self.dconv_1 = DoubleConv(in_channels=self.num_channels, out_channels=self.n_features[0],
                                  encoder=True, kernel_size=self.k_size, order=block)
        self.dconv_2 = SingleConv(in_channels=self.n_features[0], out_channels=self.n_features[1],
                                  kernel_size=self.k_size, order=block)
        self.dconv_3 = SingleConv(in_channels=self.n_features[1], out_channels=self.n_features[2],
                                  kernel_size=self.k_size, order=block)
        self.dconv_4 = SingleConv(in_channels=self.n_features[2], out_channels=self.n_features[3],
                                  kernel_size=self.k_size, order=block)

        self.centre = DoubleConv(in_channels=self.n_features[3], out_channels=self.n_features[4],
                                 encoder=False, kernel_size=self.k_size, order=block)

        self.uconv_1 = SingleConv(in_channels=self.n_features[4] + self.n_features[3],
                                  out_channels=self.n_features[3],
                                  kernel_size=self.k_size, order=block)
        self.uconv_2 = SingleConv(in_channels=self.n_features[2] + self.n_features[3],
                                  out_channels=self.n_features[2],
                                  kernel_size=self.k_size, order=block)
        self.uconv_3 = SingleConv(in_channels=self.n_features[1] + self.n_features[2],
                                  out_channels=self.n_features[1],
                                  kernel_size=self.k_size, order=block)
        self.uconv_4 = DoubleConv(in_channels=self.n_features[0] + self.n_features[1],
                                  out_channels=self.n_features[0],
                                  encoder=False, kernel_size=self.k_size, order=block)

        self.conv_last = nn.Conv3d(in_channels=self.n_features[0],
                                   out_channels=self.num_classes, kernel_size=1)

        self.activ_tanh= nn.Tanh()


    def forward(self, x, return_branches=False, return_encoder_branches=False):
        enc1 = self.dconv_1(x)
        x = self.avgpool(enc1)

        enc2 = self.dconv_2(x)
        x = self.maxpool(enc2)

        enc3 = self.dconv_3(x)
        x = self.maxpool(enc3)

        enc4 = self.dconv_4(x)
        x = self.maxpool(enc4)

        x = self.centre(x)

        dec4 = self.uconv_1(torch.cat([self.upsample(x), enc4], 1))
        dec3 = self.uconv_2(torch.cat([self.upsample(dec4), enc3], 1))
        dec2 = self.uconv_3(torch.cat([self.upsample(dec3), enc2], 1))
        dec1 = self.uconv_4(torch.cat([self.upsample(dec2), enc1], 1))

        final = self.conv_last(dec1)

        if self.use_activ:
            final = self.activ_tanh(final)

        if return_encoder_branches:
            return final, enc1, enc2, enc3, enc4, dec4, dec3, dec2, dec1

        if return_branches:
            return final, dec4, dec3, dec2, dec1
        else:
            return final


# ==================================================================================================================== #
#
#  DISCRIMINATOR ARCHITECTURE
#
#  SOME BUILDING BLOCKS FROM HERE:
#  Link: https://github.com/arnab39/cycleGAN-PyTorch/blob/master/arch/discriminators.py
#
# ==================================================================================================================== #
def conv_norm_lrelu(in_dim, out_dim, kernel_size, stride = 1, padding=0,
                    norm_layer = nn.InstanceNorm3d, bias = False):
    return nn.Sequential(
        nn.Conv3d(in_dim, out_dim, kernel_size, stride, padding, bias = bias),
        norm_layer(out_dim), nn.LeakyReLU(0.2, True))


class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, output_nc=1, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm3d, use_bias=False):
        super(NLayerDiscriminator, self).__init__()

        dis_model = [nn.Conv3d(input_nc, ndf, kernel_size=4, stride=2, padding=1),
                     nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            dis_model += [conv_norm_lrelu(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=2,
                                               norm_layer= norm_layer, padding=1, bias=use_bias)]
        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        dis_model += [conv_norm_lrelu(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=1,
                                               norm_layer= norm_layer, padding=1, bias=use_bias)]
        dis_model += [nn.Conv3d(ndf * nf_mult, output_nc, kernel_size=4, stride=1, padding=1)]

        self.dis_model = nn.Sequential(*dis_model)

    def forward(self, input):
        return self.dis_model(input)

