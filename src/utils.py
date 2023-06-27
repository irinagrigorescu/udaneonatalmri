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
# utils.py
#
##############################################################################
import torch
import numpy as np
import copy
import os
import matplotlib.pyplot as plt


# ==================================================================================================================== #
#
#  GLOBAL VARIABLES
#
# ==================================================================================================================== #
CORTEX_IDS = [ 5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
               20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]

CORTEX_IDS_11 = [[5,7,9,11,13,15, 25, 27, 29, 31],
                 [6,8,10,12,14,16,24, 26, 28, 30],
                 [20], [21], [22], [23], [32,33,34,35],
                 [36], [37], [38], [39]]

CORTEX_IDS_11_NAMES = ['LTemp', 'RTemp',
                       'LIns', 'RIns', 'ROcc', 'LOcc', 'Cing',
                       'RFront', 'LFront', 'RPar', 'LPar']

TISSUE_LABELS_DICT = {
    7: ['bck', 'csf', 'cGM', 'wm', 'cereb', 'dGM', 'bstem'],
    5: ['bck', 'csf', 'cGM', 'wm', 'dGM'],
    32: [0] + [x for x in CORTEX_IDS],
    11: ['bck'] + CORTEX_IDS_11_NAMES
}

# ==================================================================================================================== #
#
#  PLOTTING FUNCTIONS
#
# ==================================================================================================================== #
def plot_seg_img(args_, epoch_, seg_gt_, seg_pr_, t2w_gt_):
    # Figure
    plt.figure(figsize=(12, 8))

    # GT Segmentation
    plt.subplot(3, 1, 1)
    seg_plot = torch.zeros((seg_gt_.shape[2], args_.n_classes * seg_gt_.shape[3]))
    for i in range(seg_gt_.shape[1]):
        seg_plot[:, i * seg_gt_.shape[3]:(i+1) * seg_gt_.shape[3]] = \
            seg_gt_[0, i, :, :, seg_gt_.shape[4] // 2].cpu().data
    plt.imshow(seg_plot.numpy(), vmin=0.0, vmax=1.0)
    plt.colorbar()
    plt.ylabel('GT Seg')
    plt.xticks([])
    plt.yticks([])
    plt.title('E = ' + str(epoch_ + 1))

    # Pred Segmentation
    plt.subplot(3, 1, 2)
    seg_plot = torch.zeros((seg_gt_.shape[2], args_.n_classes * seg_gt_.shape[3]))
    for i in range(seg_pr_.shape[1]):
        seg_plot[:, i * seg_gt_.shape[3]:(i+1) * seg_gt_.shape[3]] = \
            seg_pr_[0, i, :, :, seg_gt_.shape[4] // 2].cpu().data
    plt.imshow(seg_plot.numpy(), vmin=0.0, vmax=1.0)
    plt.ylabel('PR Seg')
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()

    # Image
    plt.subplot(3, 1, 3)
    seg_plot = torch.zeros((seg_gt_.shape[2], args_.n_classes * seg_gt_.shape[3]))
    for i in range(seg_gt_.shape[1]):
        seg_plot[:, i * seg_gt_.shape[3]:(i+1) * seg_gt_.shape[3]] = \
            t2w_gt_[0, 0, :, :, seg_gt_.shape[4] // 2].cpu().data
    plt.imshow(seg_plot.numpy(), vmin=-1.0, vmax=1.0)
    plt.ylabel('GT T2w')
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()

    plt.savefig(args_.checkpoint_dir + '/Example_E' + str(epoch_+1) + '.png',
                dpi=200, bbox_inches='tight', transparent=True)
    plt.close('all')


def plot_seg_latent(args_, epoch_, t2w_a_, seg_gt_, seg_pr_, t2w_b_, seg_gt_b_, seg_pr_b_):
    # Figure
    plt.figure(figsize=(12, 10))

    # dhcp
    plt.subplot(6, 1, 1)
    to_plot = torch.zeros((t2w_a_.shape[2], args_.n_classes * t2w_a_.shape[3]))
    for i in range(args_.n_classes):
        to_plot[:, i * t2w_a_.shape[3]:i * t2w_a_.shape[3] + t2w_a_.shape[3]] = \
            t2w_a_[0, 0, :, :, t2w_a_.shape[4]//2].cpu().data
    plt.imshow(to_plot.numpy(), vmin=-1.0, vmax=1.0)
    plt.colorbar()
    plt.ylabel('dHCP')
    plt.xticks([])
    plt.yticks([])
    plt.title('E = ' + str(epoch_ + 1) + ' ' + args_.exp_name)

    # seg_orig gt
    plt.subplot(6, 1, 2)
    to_plot = torch.zeros((t2w_a_.shape[2], args_.n_classes * t2w_a_.shape[3]))
    for i in range(args_.n_classes):
        to_plot[:, i * t2w_a_.shape[3]:i * t2w_a_.shape[3] + t2w_a_.shape[3]] = \
            seg_gt_[0, i, :, :, seg_gt_.shape[4]//2].cpu().data
    plt.imshow(to_plot.numpy(), vmin=0.0, vmax=1.0)
    plt.colorbar()
    plt.ylabel('Seg GT')
    plt.xticks([])
    plt.yticks([])

    # seg_orig pr
    plt.subplot(6, 1, 3)
    to_plot = torch.zeros((t2w_a_.shape[2], args_.n_classes * t2w_a_.shape[3]))
    for i in range(args_.n_classes):
        to_plot[:, i * t2w_a_.shape[3]:i * t2w_a_.shape[3] + t2w_a_.shape[3]] = \
            seg_pr_[0, i, :, :, seg_pr_.shape[4]//2].cpu().data
    plt.imshow(to_plot.numpy(), vmin=-1.0, vmax=1.0)
    plt.colorbar()
    plt.ylabel('Seg PR')
    plt.xticks([])
    plt.yticks([])

    # eprime
    plt.subplot(6, 1, 4)
    to_plot = torch.zeros((t2w_a_.shape[2], args_.n_classes * t2w_a_.shape[3]))
    for i in range(args_.n_classes):
        to_plot[:, i * t2w_a_.shape[3]:i * t2w_a_.shape[3] + t2w_a_.shape[3]] = \
            t2w_b_[0, 0, :, :, t2w_b_.shape[4]//2].cpu().data
    plt.imshow(to_plot.numpy(), vmin=-1.0, vmax=1.0)
    plt.colorbar()
    plt.ylabel('EPRIME')
    plt.xticks([])
    plt.yticks([])

    # seg_orig pr
    plt.subplot(6, 1, 6)
    to_plot = torch.zeros((t2w_a_.shape[2], args_.n_classes * t2w_a_.shape[3]))
    for i in range(args_.n_classes):
        to_plot[:, i * t2w_a_.shape[3]:i * t2w_a_.shape[3] + t2w_a_.shape[3]] = \
            seg_pr_b_[0, i, :, :, seg_pr_b_.shape[4]//2].cpu().data
    plt.imshow(to_plot.numpy(), vmin=0.0, vmax=1.0)
    plt.colorbar()
    plt.ylabel('Seg EPRIME PR')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(6, 1, 5)
    to_plot = torch.zeros((t2w_a_.shape[2], args_.n_classes * t2w_a_.shape[3]))
    for i in range(args_.n_classes):
        to_plot[:, i * t2w_a_.shape[3]:i * t2w_a_.shape[3] + t2w_a_.shape[3]] = \
            seg_gt_b_[0, i, :, :, seg_gt_b_.shape[4]//2].cpu().data
    plt.imshow(to_plot.numpy(), vmin=0.0, vmax=1.0)
    plt.colorbar()
    plt.ylabel('Seg EPRIME GT')
    plt.xticks([])
    plt.yticks([])

    plt.savefig(args_.checkpoint_dir + '/Example_E' + str(epoch_ + 1) + '.png',
                dpi=200, bbox_inches='tight', transparent=True)
    plt.close('all')


def plot_seg_img_fake(args_, epoch_, seg_gt_, seg_pr_, t2w_gt_, t2w_fake_):
    # Figure
    plt.figure(figsize=(16, 8))

    # GT Segmentation
    plt.subplot(4, 1, 1)
    seg_plot = torch.zeros((seg_gt_.shape[2], args_.n_classes * seg_gt_.shape[3]))
    for i in range(seg_gt_.shape[1]):
        seg_plot[:, i * seg_gt_.shape[3]:(i + 1) * seg_gt_.shape[3]] = \
            seg_gt_[0, i, :, :, seg_gt_.shape[4] // 2].cpu().data
    plt.imshow(seg_plot.numpy(), vmin=0.0, vmax=1.0)
    plt.colorbar()
    plt.ylabel('GT Seg')
    plt.xticks([])
    plt.yticks([])
    plt.title('E = ' + str(epoch_ + 1))

    # Pred Segmentation
    plt.subplot(4, 1, 2)
    seg_plot = torch.zeros((seg_gt_.shape[2], args_.n_classes * seg_gt_.shape[3]))
    for i in range(seg_pr_.shape[1]):
        seg_plot[:, i * seg_gt_.shape[3]:(i + 1) * seg_gt_.shape[3]] = \
            seg_pr_[0, i, :, :, seg_gt_.shape[4] // 2].cpu().data
    plt.imshow(seg_plot.numpy(), vmin=0.0, vmax=1.0)
    plt.ylabel('PR Seg')
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()

    # Image
    plt.subplot(4, 1, 3)
    seg_plot = torch.zeros((seg_gt_.shape[2], args_.n_classes * seg_gt_.shape[3]))
    for i in range(seg_gt_.shape[1]):
        seg_plot[:, i * seg_gt_.shape[3]:(i + 1) * seg_gt_.shape[3]] = \
            t2w_gt_[0, 0, :, :, seg_gt_.shape[4] // 2].cpu().data
    plt.imshow(seg_plot.numpy(), vmin=-1.0, vmax=1.0)
    plt.ylabel('GT T2w')
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()

    # Image fake
    plt.subplot(4, 1, 4)
    seg_plot = torch.zeros((seg_gt_.shape[2], args_.n_classes * seg_gt_.shape[3]))
    for i in range(seg_gt_.shape[1]):
        seg_plot[:, i * seg_gt_.shape[3]:(i + 1) * seg_gt_.shape[3]] = \
            t2w_fake_[0, 0, :, :, seg_gt_.shape[4] // 2].cpu().data
    plt.imshow(seg_plot.numpy(), vmin=-1.0, vmax=1.0)
    plt.ylabel('Fake T2w')
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()

    plt.savefig(args_.checkpoint_dir + '/Example_E' + str(epoch_ + 1) + '.png',
                dpi=200, bbox_inches='tight', transparent=True)
    plt.close('all')


def plot_seg_cortex(args_, epoch_, seg_gt_out_, seg_pr_, seg_gt_in_):

    n_classes, nx, ny, nz = seg_gt_out_.shape[1:]
    vvmax = n_classes

    # Figure
    plt.figure(figsize=(18, 6))

    # Image
    plt.subplot(2, 4, 1)
    mask = np.argmax(seg_gt_in_[0, :, :, :, nz//2].cpu().data, axis=0)
    plt.imshow(mask.numpy(), vmin=0.0, vmax=1.0)
    plt.colorbar()
    plt.ylabel('GT INPUT')
    plt.xticks([])
    plt.yticks([])
    plt.title('E = ' + str(epoch_ + 1))

    # GT Segmentation
    plt.subplot(2, 4, 2)
    seg_plot = np.argmax(seg_gt_out_[0, :, :, :, nz//2].cpu().data, axis=0)
    plt.imshow(seg_plot.numpy(), vmin=0.0, vmax=vvmax, cmap='jet')
    plt.colorbar()
    plt.ylabel('GT Seg')
    plt.xticks([])
    plt.yticks([])

    # Pred Segmentation
    plt.subplot(2, 4, 3)
    seg_plot_ = np.argmax(seg_pr_[0, :, :, :, nz//2].cpu().data, axis=0)
    plt.imshow(seg_plot_.numpy(), vmin=0.0, vmax=vvmax, cmap='jet')
    plt.colorbar()
    plt.ylabel('PR Seg')
    plt.xticks([])
    plt.yticks([])

    # Difference Segmentation
    plt.subplot(2, 4, 4)
    temp_ = seg_plot.numpy() - seg_plot_.numpy()
    plt.imshow(temp_, vmin=-np.max(np.abs(temp_)), vmax=np.max(np.abs(temp_)), cmap='jet')
    plt.colorbar()
    plt.ylabel('GT - PR Seg')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(2, 4, 5)
    seg_plot = seg_gt_in_[0, 0, :, :, nz//2].cpu().data
    plt.imshow(seg_plot.numpy(), vmin=0.0, vmax=1.0)
    plt.colorbar()
    plt.ylabel('GT INPUT')
    plt.xticks([])
    plt.yticks([])
    plt.title('E = ' + str(epoch_ + 1))

    # GT Segmentation
    plt.subplot(2, 4, 6)
    seg_plot = seg_gt_out_[0, 0, :, :, nz // 2].cpu().data
    plt.imshow(seg_plot.numpy(), vmin=0.0, vmax=1.0)
    plt.colorbar()
    plt.ylabel('GT Seg')
    plt.xticks([])
    plt.yticks([])

    # Pred Segmentation
    plt.subplot(2, 4, 7)
    seg_plot_ = seg_pr_[0, 0, :, :, nz // 2].cpu().data
    plt.imshow(seg_plot_.numpy(), vmin=0.0, vmax=1.0)
    plt.colorbar()
    plt.ylabel('PR Seg')
    plt.xticks([])
    plt.yticks([])

    # Difference Segmentation
    plt.subplot(2, 4, 8)
    temp_ = seg_plot.numpy() - seg_plot_.numpy()
    plt.imshow(temp_, vmin=-np.max(np.abs(temp_)), vmax=np.max(np.abs(temp_)), cmap='jet')
    plt.colorbar()
    plt.ylabel('GT - PR Seg')
    plt.xticks([])
    plt.yticks([])

    plt.savefig(args_.checkpoint_dir + '/Example_E' + str(epoch_ + 1) + '.png',
                dpi=200, bbox_inches='tight', transparent=True)
    plt.close('all')


def plot_losses_train(args, losses_train, title_plot):
    # Get some variables about the train
    ####################
    n_epochs_train = len(losses_train)
    keys_train = list(losses_train[0].keys())
    n_iter_train = len(losses_train[0][keys_train[0]])
    print(keys_train)
    print(len(keys_train))

    # Average losses
    ####################
    losses_train_mean = {key_: [] for key_ in keys_train}
    losses_train_std = {key_: [] for key_ in keys_train}
    for epoch_ in losses_train:
        for key_ in keys_train:
            losses_train_mean[key_].append(np.mean(epoch_[key_]))
            losses_train_std[key_].append(np.std(epoch_[key_]))

    # Plot losses
    ####################
    import matplotlib.pyplot as plt

    plt.figure(figsize=(20, 18))
    for i_, key_ in enumerate(keys_train):
        plt.subplot(6, 2, i_ + 1)
        plt.fill_between(np.arange(1, n_epochs_train),
                         [x - y for x, y in zip(losses_train_mean[key_][1:],
                                                losses_train_std[key_][1:])],
                         [x + y for x, y in zip(losses_train_mean[key_][1:],
                                                losses_train_std[key_][1:])],
                         alpha=0.2)
        plt.semilogy(np.arange(0, n_epochs_train), losses_train_mean[key_])
        plt.grid()
        plt.xlabel('epochs')
        plt.ylabel(key_)
        if i_ == 0:
            # plt.ylim([1e+1, 1e+2])
            plt.title(args.exp_name)

        if i_ >= 11:
            break

    plt.savefig(args.results_dir + '/' + title_plot + str(n_epochs_train) + '.png',
                dpi=200, bbox_inches='tight', transparent=True)
    plt.close('all')


# ==================================================================================================================== #
#
#  CLASS FOR ARGUMENTS
#
# ==================================================================================================================== #
class ArgumentsTrainInferenceSeg():
    """
    Arguments for the experiments
    """
    def __init__(self,
                 epochs=100,
                 decay_epoch=1,
                 batch_size=1,
                 lr=0.002,
                 validation_steps=15,
                 training=False,
                 testing=False,
                 gpu_ids=0,
                 crop_height=128,
                 crop_width=128,
                 crop_depth=128,
                 root_dir_dHCP='/path/to/domain/A/data/',
                 root_dir_EPRIME='/path/to/domain/B/data/',
                 csv_dir='path/to/csv/file/',
                 train_A_csv='domA_data_train.csv',
                 valid_A_csv='domA_data_valid.csv',
                 test_A_csv='domA_data_test_paper.csv',
                 train_B_csv='domB_data_train.csv',
                 valid_B_csv='domB_data_valid.csv',
                 test_B_csv='domB_data_test_paper.csv',
                 results_dir='/path/to/results/dir/',
                 checkpoint_dir='/path/to/checkpoints/dir/',
                 n_classes=7,
                 lamda_seg=1, lamda_adv=10, lamda_ncc=10,
                 seg_net='unet3D', seg_features=[16, 32, 64, 128],
                 gen_net='unet_128', ngf=64,
                 dis_net='n_layers', ndf=64,
                 alpha_max=1.0, e1=5, e2=50,
                 is_augment=True,
                 exp_name='test'):

        self.epochs = epochs
        self.decay_epoch = decay_epoch
        self.batch_size = batch_size
        self.lr = lr

        self.validation_steps = validation_steps
        self.training = training
        self.testing = testing

        self.gpu_ids = gpu_ids

        self.crop_height = crop_height
        self.crop_width = crop_width
        self.crop_depth = crop_depth

        self.exp_name = exp_name

        self.root_dir_dHCP = root_dir_dHCP
        self.root_dir_EPRIME = root_dir_EPRIME

        self.csv_dir = csv_dir

        self.train_A_csv = train_A_csv
        self.valid_A_csv = valid_A_csv
        self.test_A_csv = test_A_csv

        self.train_B_csv = train_B_csv
        self.valid_B_csv = valid_B_csv
        self.test_B_csv = test_B_csv

        self.results_dir = results_dir
        self.checkpoint_dir = checkpoint_dir

        self.seg_net = seg_net
        self.seg_features = seg_features

        self.gen_net = gen_net
        self.ngf = ngf

        self.dis_net = dis_net
        self.ndf = ndf

        self.alpha_max = alpha_max
        self.e1 = e1
        self.e2 = e2

        self.lamda_seg = lamda_seg
        self.lamda_adv = lamda_adv
        self.lamda_ncc = lamda_ncc

        self.n_classes = n_classes
        self.is_augment = is_augment


# ==================================================================================================================== #
#
#  HELPER FUNCTIONS
#
# ==================================================================================================================== #
def print_networks(nets, names):
    """
    Print network parameters
    Link: https://github.com/arnab39/cycleGAN-PyTorch/blob/ecdc9735e426992056c70e78f3143bb6d538e48c/utils.py

    :param nets:
    :param names:
    :return:
    """
    print('    ------------Number of Parameters---------------')

    for i, net in enumerate(nets):
        num_params = 0
        for param in net.parameters():
            num_params += param.numel()
        print('    [Network %s] Total number of parameters : %.3f M' % (names[i], num_params / 1e6))

    print('    -----------------------------------------------')


def save_checkpoint(state, save_path):
    """
    To save the checkpoint

    Link: https://github.com/arnab39/cycleGAN-PyTorch/blob/ecdc9735e426992056c70e78f3143bb6d538e48c/utils.py
    :param state:
    :param save_path:
    :return:
    """
    torch.save(state, save_path)


def load_checkpoint(ckpt_path, map_location=None):
    """
    To load the checkpoint

    Link: https://github.com/arnab39/cycleGAN-PyTorch/blob/ecdc9735e426992056c70e78f3143bb6d538e48c/utils.py
    :param ckpt_path:
    :param map_location:
    :return:
    """
    ckpt = torch.load(ckpt_path, map_location=map_location)
    print('    [*] Loading checkpoint from %s succeed!' % ckpt_path)
    return ckpt


def cuda(xs):
    """
    Make cuda tensor

    Link: https://github.com/arnab39/cycleGAN-PyTorch/blob/ecdc9735e426992056c70e78f3143bb6d538e48c/utils.py
    :param xs:
    :return:
    """
    if torch.cuda.is_available():
        if not isinstance(xs, (list, tuple)):
            return xs.cuda()
        else:
            return [x.cuda() for x in xs]


class Sample_from_Pool(object):
    """
    To store 50 generated image in a pool and sample from it when it is full
    (Shrivastava et al’s strategy)

    Link: https://github.com/arnab39/cycleGAN-PyTorch/blob/ecdc9735e426992056c70e78f3143bb6d538e48c/utils.py

    """
    def __init__(self, max_elements=50):
        self.max_elements = max_elements
        self.cur_elements = 0
        self.items = []

    def __call__(self, in_items):
        return_items = []
        for in_item in in_items:
            if self.cur_elements < self.max_elements:
                self.items.append(in_item)
                self.cur_elements = self.cur_elements + 1
                return_items.append(in_item)
            else:
                if np.random.ranf() > 0.5:
                    idx = np.random.randint(0, self.max_elements)
                    tmp = copy.copy(self.items[idx])
                    self.items[idx] = in_item
                    return_items.append(tmp)
                else:
                    return_items.append(in_item)
        return return_items

