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
# dataloaders.py
#
##############################################################################

from __future__ import print_function, division
import os
import torch
import torchio
from torchvision.transforms import Compose
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class SegmentationDataLoader(Dataset):
    """
    Dataset loader for segmentation
    """
    def __init__(self, csv_file, root_dir, is_augment=False, shuffle=True, transform=None):
        """
        Constructor

        :param csv_file: Path to the csv file with GA, AS, gender, filename
        :param root_dir: Path to data
        :param is_augment: To augment or not
        :param shuffle: if True reshuffles indices for every get item
        :param transform: optional transform to be applied
        """
        self.data_file = pd.read_csv(csv_file)
        self.input_folder = root_dir
        self.shuffle = shuffle
        self.indices = np.arange(len(self.data_file))  # indices of the data [0 ... N-1]
        self.extra_transform = transform
        self.is_augment = is_augment

        # Get the transforms (preprocessing only if is_augment is False)
        self.transform = self.get_transform()

        # Shuffle indices if shuffle is set to True
        self.shuffle_indices()


    def get_transform(self):
        """
        Getter for both transformations
        :return:
        """

        self.preprocessing = self.get_preprocessing()
        self.augmentation = self.get_augmentation()

        # Put them all together
        if self.is_augment:
            return Compose(self.preprocessing + self.augmentation)
        else:
            return Compose(self.preprocessing)


    def get_augmentation(self):
        """
        Getter for augmentation
        :return:
        """

        self.to_motion = torchio.transforms.RandomMotion(degrees=2.0,
                                                         translation=2.0,  # 3.0
                                                         num_transforms=1,
                                                         p=0.75)
        self.to_spike = torchio.transforms.RandomSpike(num_spikes=1,
                                                       intensity=0.2,
                                                       p=0.75)
        self.to_affine = torchio.transforms.RandomAffine(scales=(0.8, 1.2),
                                                         degrees=(10),
                                                         isotropic=False,
                                                         default_pad_value='minimum',
                                                         p=0.75)
        self.to_bias_field = torchio.transforms.RandomBiasField(coefficients=(-0.5, 0.5),
                                                                order=3,
                                                                p=0.75)

        # Choose which augmentation to do
        augmentation_choice = np.random.choice([0, 1, 2, 3, 4])
        if augmentation_choice == 0:
            aug_img = [self.to_affine]
        elif augmentation_choice == 1:
            aug_img = [self.to_motion]
        elif augmentation_choice == 2:
            aug_img = [self.to_spike]
        elif augmentation_choice == 3:
            aug_img = [self.to_bias_field]
        else:
            aug_img = [self.to_affine, self.to_motion]

        return aug_img


    def get_preprocessing(self):
        """
        Getter for preprocessing
        :return:
        """

        # Canonical reorientation and resampling
        to_ras = torchio.transforms.ToCanonical()
        # Resampling to 0.5 isotropic (testing for lower resolution, change to 0.5 below)
        to_iso = torchio.transforms.Resample((1.0, 1.0, 1.0))
        # Z-Normalisation
        to_znorm = torchio.transforms.ZNormalization()
        # Rescaling
        self.to_rescl = torchio.transforms.RescaleIntensity(out_min_max=(0.0, 1.0))

        return [to_ras, to_iso, to_znorm, self.to_rescl]


    def __len__(self):
        """
        Number of elements per epoch
        :return:
        """
        return len(self.data_file)


    def __getitem__(self, item):

        if torch.is_tensor(item):
            item = item.tolist()

        self.shuffle_indices()
        item = self.indices[item]

        # Get image filename
        img_name = os.path.join(self.input_folder,
                                self.data_file.iloc[item, 0])
        # Get segmentation filename
        lab_name = os.path.join(self.input_folder,
                                self.data_file.iloc[item, 1])

        # Create torchio subject
        subject = torchio.Subject(
            t2w=torchio.Image(img_name, torchio.INTENSITY),
            label=torchio.Image(lab_name, torchio.LABEL),
        )
        dataset = torchio.SubjectsDataset([subject])[0]

        # Transform subject
        transformed_subj = self.transform(dataset)

        # Create sample
        sample = {'image': transformed_subj['t2w']['data'][0, :, :, :].numpy().astype(np.float32),
                  'lab': transformed_subj['label']['data'][0, :, :, :].numpy().astype(np.float32),
                  'GA': float(self.data_file.iloc[item, 2]),
                  'AS': float(self.data_file.iloc[item, 3]),
                  'gender': self.data_file.iloc[item, 4],
                  'name': self.data_file.iloc[item, 0].split('_T2w')[0],
                  'img_aff': transformed_subj['t2w']['affine'],
                  'seg_aff': transformed_subj['label']['affine']}

        # Extra transform if it exists
        if self.extra_transform:
            sample = self.extra_transform(sample)

        return sample


    def shuffle_indices(self):
        """
        Shuffle indices in case self.shuffle is True
        :return:
        """

        if self.shuffle:
            np.random.shuffle(self.indices)
        else:
            self.indices = np.arange(len(self.data_file))


class SegmentationDataLoaderCortex(Dataset):
    """
    Dataset loader for cortical parcellation networks
    """
    def __init__(self, csv_file, root_dir, is_augment=False, shuffle=True, transform=None):
        """
        Constructor
        :param csv_file: Path to the csv file with GA, AS, gender, filename
        :param root_dir: Path to data
        :param shuffle: if True reshuffles indices for every get item
        :param transform: optional transform to be applied
        """
        self.data_file = pd.read_csv(csv_file)
        self.input_folder = root_dir
        self.shuffle = shuffle
        self.indices = np.arange(len(self.data_file))  # indices of the data [0 ... N-1]
        self.extra_transform = transform
        self.is_augment = is_augment

        # Get the transforms (preprocessing only if is_augment is False)
        self.transform = self.get_transform()

        # Shuffle indices if shuffle is set to True
        self.shuffle_indices()

    def get_transform(self):
        """
        Getter for both transformations
        :return:
        """

        self.preprocessing = self.get_preprocessing()
        self.augmentation = self.get_augmentation()

        # Put them all together
        if self.is_augment:
            return Compose(self.preprocessing + self.augmentation)
        else:
            return Compose(self.preprocessing)

    def get_augmentation(self):
        """
        Getter for augmentation
        :return:
        """

        self.to_affine = torchio.transforms.RandomAffine(scales=(0.8, 1.2),
                                                         degrees=(10),
                                                         isotropic=False,
                                                         default_pad_value='minimum',
                                                         p=0.75)

        augmentation_choice = np.random.choice([0, 1])

        if augmentation_choice == 0:
            aug_img = [self.to_affine, self.to_rescl]
        else:
            aug_img = [self.to_rescl]

        return aug_img

    def get_preprocessing(self):
        """
        Getter for preprocessing
        :return:
        """

        # Canonical reorientation and resampling
        to_ras = torchio.transforms.ToCanonical()
        # Resampling to 0.5 isotropic (testing for lower resolution, change to 0.5 below)
        to_iso = torchio.transforms.Resample((1.0, 1.0, 1.0))
        # Rescaling
        self.to_rescl = torchio.transforms.RescaleIntensity(out_min_max=(0.0, 1.0))

        return [to_ras, to_iso, self.to_rescl]


    def __len__(self):
        """
        Number of elements per epoch
        :return:
        """
        return len(self.data_file)

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()

        self.shuffle_indices()
        item = self.indices[item]

        lab_name = os.path.join(self.input_folder, self.data_file.iloc[item, 1])
        lab_cortex_name = os.path.join(self.input_folder, self.data_file.iloc[item, 1].split('_tissue')[0] + '_all.nii.gz')

        # Read data:
        subject = torchio.Subject(
            label=torchio.Image(lab_name, torchio.LABEL),
            labelcortex=torchio.Image(lab_cortex_name, torchio.LABEL),
        )
        dataset = torchio.SubjectsDataset([subject])[0]

        # Transform subject
        transformed_subj = self.transform(dataset)

        # Create sample
        lab_ = transformed_subj['label']['data'][0, :, :, :].numpy().astype(np.float32)
        lab_cortex_ = transformed_subj['labelcortex']['data'][0, :, :, :].numpy().astype(np.float32)
        lab_aff = transformed_subj['label']['affine']
        ga_baby = float(self.data_file.iloc[item, 2])
        as_baby = float(self.data_file.iloc[item, 3])
        ge_baby = self.data_file.iloc[item, 4]
        subj_name = self.data_file.iloc[item, 0].split('_T2w')[0]

        # Create sample_img:
        sample = {'lab': lab_,
                  'lab_cortex': lab_cortex_,
                  'GA': ga_baby,
                  'AS': as_baby,
                  'gender': ge_baby,
                  'name': subj_name,
                  'seg_aff': lab_aff}

        # Extra transform if it exists
        if self.extra_transform:
            sample = self.extra_transform(sample)

        return sample

    def shuffle_indices(self):
        """
        Shuffle indices in case self.shuffle is True
        :return:
        """
        if self.shuffle:
            np.random.shuffle(self.indices)
        else:
            self.indices = np.arange(len(self.data_file))


