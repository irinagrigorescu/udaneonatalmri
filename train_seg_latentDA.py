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
# train_seg_baseline.py
#
##############################################################################
from src.utils import ArgumentsTrainInferenceSeg, plot_losses_train
from src import models as md


# ====================================================================================================================

N_epochs = 31
N_epochs_adv = 20  # 5
N_epochs_fin = 50  # 50
alpha=0.03

# Prepare arguments
############################################################
args = ArgumentsTrainInferenceSeg(epochs=N_epochs,
                                  decay_epoch=1,
                                  batch_size=1,
                                  lr=0.002,
                                  gpu_ids=0,
                                  crop_height=128,
                                  crop_width=128,
                                  crop_depth=128,
                                  lamda_seg=10.0, lamda_adv=1.0,
                                  validation_steps=8,
                                  training=True,
                                  root_dir_dHCP='/path/to/dhcp/data/',
                                  root_dir_EPRIME='/path/to/eprime/data/',
                                  csv_dir='~/example_csv_files/',
                                  train_A_csv='train_A.csv',
                                  valid_A_csv='valid_A.csv',
                                  test_A_csv='test_A.csv',
                                  train_B_csv='train_B.csv',
                                  valid_B_csv='valid_B.csv',
                                  test_B_csv='test_B.csv',
                                  results_dir='/path/to/results/',
                                  checkpoint_dir='/path/to/checkpoints/',
                                  exp_name='test',
                                  n_classes=7,
                                  seg_net='unet3D', seg_features=[16, 32, 64, 128, 256],
                                  dis_net='n_layers', ndf=64,
                                  alpha_max=alpha, e1=N_epochs_adv, e2=N_epochs_fin,
                                  is_augment=True)

args.gpu_ids = [0]

if args.training:
    print("Training")
    model = md.Segmentation3DLatent(args)

    # Run train
    ####################
    losses_train = model.train(args)

    # Plot losses
    ####################
    plot_losses_train(args, losses_train, 'fig_losses_train_E')

