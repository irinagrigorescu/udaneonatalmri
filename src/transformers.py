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
# transformers.py
#
##############################################################################

from __future__ import print_function, division
import torch
import numpy as np
from src.utils import CORTEX_IDS, CORTEX_IDS_11


# ==================================================================================================================== #
#
#  TO TENSOR
#
# ==================================================================================================================== #
class ToTensor(object):
    """
    Convert ndarrays in sample to Tensors
    """

    def __call__(self, sample):
        if 'image' not in sample.keys():
            if 'lab_cortex' not in sample.keys():
                lab = sample['lab']
                current_sample_dict = {'lab': (torch.from_numpy(lab)).float(),
                                       'GA': np.expand_dims(sample['GA'], 0).astype(dtype=np.float32),
                                       'AS': np.expand_dims(sample['AS'], 0).astype(dtype=np.float32),
                                       'gender': sample['gender'],
                                       'name': sample['name'],
                                       'seg_aff': sample['seg_aff']}
            else:
                lab, lab_cortex = sample['lab'], sample['lab_cortex']
                current_sample_dict = {'lab': (torch.from_numpy(lab)).float(),
                                       'lab_cortex': (torch.from_numpy(lab_cortex)).float(),
                                       'GA': np.expand_dims(sample['GA'], 0).astype(dtype=np.float32),
                                       'AS': np.expand_dims(sample['AS'], 0).astype(dtype=np.float32),
                                       'gender': sample['gender'],
                                       'name': sample['name'],
                                       'seg_aff': sample['seg_aff']}

        else:
            image = sample['image']
            # Transform image from H x W x D to 1 x H x W x D
            image = np.expand_dims(image, 0).astype(dtype=np.float32)

            current_sample_dict = {'image': (torch.from_numpy(image)).float(),
                                   'GA': np.expand_dims(sample['GA'], 0).astype(dtype=np.float32),
                                   'AS': np.expand_dims(sample['AS'], 0).astype(dtype=np.float32),
                                   'gender': sample['gender'],
                                   'name': sample['name'],
                                   'img_aff': sample['img_aff']}

        # Add segmentation
        if 'lab' in sample.keys():
            lab = sample['lab']

            current_sample_dict['lab'] = (torch.from_numpy(lab)).float()
            current_sample_dict['seg_aff'] = sample['seg_aff']

        return current_sample_dict


# ==================================================================================================================== #
#
#  RANDOM CROPS
#
# ==================================================================================================================== #
class RandomCrop(object):
    """
    Randomly crop the image and label in a sample_img
    """

    def __init__(self, output_size, is_random=True, is_test=False, n_classes=9, is_normalize=False):
        self.is_random = is_random
        self.is_normalize = is_normalize
        self.n_classes = n_classes
        self.is_test = is_test

        # Check it's instance of a tuple or int
        assert isinstance(output_size, (int, tuple))

        # If int make into a tuple
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size, output_size)
        # Else check it has 3 sizes
        else:
            assert len(output_size) == 3
            self.output_size = output_size

    def __call__(self, sample):
        image, lab = sample['image'], sample['lab']

        # Get image size
        h, w, d = image.shape[:3]

        if self.is_test:
            new_h, new_w, new_d = h, w, d
        else:
            new_h, new_w, new_d = self.output_size

        # Pad image in case new_i > i
        pad_value = np.min(image)
        if new_h >= h:
            pad_ = (new_h - h) // 2 + 1
            image = np.pad(image, ((pad_, pad_), (0, 0), (0, 0)), 'constant', constant_values=pad_value)
            lab1 = np.pad(lab[[0], :, :, :], ((0, 0), (pad_, pad_), (0, 0), (0, 0)),
                          'constant', constant_values=1)
            lab2 = np.pad(lab[np.arange(1, self.n_classes), :, :, :], ((0, 0), (pad_, pad_), (0, 0), (0, 0)),
                          'constant', constant_values=0)
            lab = np.concatenate([lab1, lab2], axis=0)

        if new_w >= w:
            pad_ = (new_w - w) // 2 + 1
            image = np.pad(image, ((0, 0), (pad_, pad_), (0, 0)), 'constant', constant_values=pad_value)
            lab1 = np.pad(lab[[0], :, :, :], ((0, 0), (0, 0), (pad_, pad_), (0, 0)),
                          'constant', constant_values=1)
            lab2 = np.pad(lab[np.arange(1, self.n_classes), :, :, :], ((0, 0), (0, 0), (pad_, pad_), (0, 0)),
                          'constant', constant_values=0)
            lab = np.concatenate([lab1, lab2], axis=0)

        if new_d >= d:
            pad_ = (new_d - d) // 2 + 1
            image = np.pad(image, ((0, 0), (0, 0), (pad_, pad_)), 'constant', constant_values=pad_value)
            lab1 = np.pad(lab[[0], :, :, :], ((0, 0), (0, 0), (0, 0), (pad_, pad_)),
                          'constant', constant_values=1)
            lab2 = np.pad(lab[np.arange(1, self.n_classes), :, :, :], ((0, 0), (0, 0), (0, 0), (pad_, pad_)),
                          'constant', constant_values=0)
            lab = np.concatenate([lab1, lab2], axis=0)
        h, w, d = image.shape[:3]

        if self.is_random:
            # Get patch starting point
            patch_x = np.random.randint(0, h - new_h)
            patch_y = np.random.randint(0, w - new_w)
            patch_z = np.random.randint(0, d - new_d)
        else:
            # Calculate centre of mass
            coords_x, coords_y, coords_z = np.meshgrid(np.arange(0, w),
                                                       np.arange(0, h),
                                                       np.arange(0, d))
            lab_brain = np.zeros((h, w, d))
            lab_brain[lab[0, ...] == 0] = 1
            coords_x = np.round(np.sum(coords_x * lab_brain) / np.sum(lab_brain))
            coords_y = np.round(np.sum(coords_y * lab_brain) / np.sum(lab_brain))
            coords_z = np.round(np.sum(coords_z * lab_brain) / np.sum(lab_brain))

            # Calculate start point of patch
            patch_y = 0 if (int(coords_x - new_h // 2) < 0 or int(coords_x + new_h // 2) >= h) \
                else int(coords_x - new_h // 2)
            patch_x = 0 if (int(coords_y - new_w // 2) < 0 or int(coords_y + new_w // 2) >= w) \
                else int(coords_y - new_w // 2)
            patch_z = 0 if (int(coords_z - new_d // 2) < 0 or int(coords_z + new_d // 2) >= d) \
                else int(coords_z - new_d // 2)

        # Create new image
        if self.is_normalize:
            image = (image - np.min(image)) / \
                    (np.max(image) - np.min(image))

        image = image[patch_x: patch_x + new_h,
                      patch_y: patch_y + new_w,
                      patch_z: patch_z + new_d]
        # Create new lab
        lab = lab[:,
                  patch_x: patch_x + new_h,
                  patch_y: patch_y + new_w,
                  patch_z: patch_z + new_d]

        return {'image': image,
                'lab': lab,
                'GA': sample['GA'],
                'AS': sample['AS'],
                'gender': sample['gender'],
                'name': sample['name'],
                'img_aff': sample['img_aff'],
                'seg_aff': sample['seg_aff']}


class RandomCropCortex(object):
    """
    Randomly crop the image and label in a sample_img
    """

    def __init__(self, output_size, is_random=True, is_test=False, n_classes=2):
        self.is_random = is_random
        self.n_classes = n_classes
        self.is_test = is_test

        # Check it's instance of a tuple or int
        assert isinstance(output_size, (int, tuple))

        # If int make into a tuple
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size, output_size)
        # Else check it has 3 sizes
        else:
            assert len(output_size) == 3
            self.output_size = output_size

    def __call__(self, sample):
        lab, lab_cortex, old_lab = sample['lab'], sample['lab_cortex'], sample['old_lab']

        # Get image size
        h, w, d = lab.shape[1:]

        n_classes_cortex = lab_cortex.shape[0]

        if self.is_test:
            new_h, new_w, new_d = h, w, d
        else:
            new_h, new_w, new_d = self.output_size

        # Pad image in case new_i > i
        if new_h > h:
            pad_ = (new_h - h) // 2 + 1

            old_lab = np.pad(old_lab, ((pad_, pad_), (0, 0), (0, 0)), 'constant', constant_values=0)

            lab1 = np.pad(lab[[0], :, :, :], ((0, 0), (pad_, pad_), (0, 0), (0, 0)),
                          'constant', constant_values=1)
            lab2 = np.pad(lab[np.arange(1, self.n_classes), :, :, :], ((0, 0), (pad_, pad_), (0, 0), (0, 0)),
                          'constant', constant_values=0)
            lab = np.concatenate([lab1, lab2], axis=0)

            lab1_ = np.pad(lab_cortex[[0], :, :, :], ((0, 0), (pad_, pad_), (0, 0), (0, 0)),
                          'constant', constant_values=1)
            lab2_ = np.pad(lab_cortex[np.arange(1, n_classes_cortex), :, :, :], ((0, 0), (pad_, pad_), (0, 0), (0, 0)),
                          'constant', constant_values=0)
            lab_cortex = np.concatenate([lab1_, lab2_], axis=0)

        if new_w > w:
            pad_ = (new_w - w) // 2 + 1

            old_lab = np.pad(old_lab, ((0, 0), (pad_, pad_), (0, 0)), 'constant', constant_values=0)

            lab1 = np.pad(lab[[0], :, :, :], ((0, 0), (0, 0), (pad_, pad_), (0, 0)),
                          'constant', constant_values=1)
            lab2 = np.pad(lab[np.arange(1, self.n_classes), :, :, :], ((0, 0), (0, 0), (pad_, pad_), (0, 0)),
                          'constant', constant_values=0)
            lab = np.concatenate([lab1, lab2], axis=0)

            lab1_ = np.pad(lab_cortex[[0], :, :, :], ((0, 0), (0, 0), (pad_, pad_), (0, 0)),
                          'constant', constant_values=1)
            lab2_ = np.pad(lab_cortex[np.arange(1, n_classes_cortex), :, :, :], ((0, 0), (0, 0), (pad_, pad_), (0, 0)),
                          'constant', constant_values=0)
            lab_cortex = np.concatenate([lab1_, lab2_], axis=0)

        if new_d > d:
            pad_ = (new_d - d) // 2 + 1

            old_lab = np.pad(old_lab, ((0, 0), (0, 0), (pad_, pad_)), 'constant', constant_values=0)

            lab1 = np.pad(lab[[0], :, :, :], ((0, 0), (0, 0), (0, 0), (pad_, pad_)),
                          'constant', constant_values=1)
            lab2 = np.pad(lab[np.arange(1, self.n_classes), :, :, :], ((0, 0), (0, 0), (0, 0), (pad_, pad_)),
                          'constant', constant_values=0)
            lab = np.concatenate([lab1, lab2], axis=0)

            lab1_ = np.pad(lab_cortex[[0], :, :, :], ((0, 0), (0, 0), (0, 0), (pad_, pad_)),
                          'constant', constant_values=1)
            lab2_ = np.pad(lab_cortex[np.arange(1, n_classes_cortex), :, :, :], ((0, 0), (0, 0), (0, 0), (pad_, pad_)),
                          'constant', constant_values=0)
            lab_cortex = np.concatenate([lab1_, lab2_], axis=0)


        # Get the new ones
        h, w, d = lab.shape[1:]

        if self.is_random:
            # Get patch starting point
            patch_x = np.random.randint(0, h - new_h)
            patch_y = np.random.randint(0, w - new_w)
            patch_z = np.random.randint(0, d - new_d)
        else:
            # Calculate centre of mass
            coords_x, coords_y, coords_z = np.meshgrid(np.arange(0, w),
                                                       np.arange(0, h),
                                                       np.arange(0, d))
            lab_brain = np.zeros((h, w, d))
            lab_brain[old_lab != 0] = 1
            coords_x = np.round(np.sum(coords_x * lab_brain) / np.sum(lab_brain))
            coords_y = np.round(np.sum(coords_y * lab_brain) / np.sum(lab_brain))
            coords_z = np.round(np.sum(coords_z * lab_brain) / np.sum(lab_brain))

            # Calculate start point of patch
            patch_y = 0 if (int(coords_x - new_h // 2) < 0 or int(coords_x + new_h // 2) >= h) \
                else int(coords_x - new_h // 2)
            patch_x = 0 if (int(coords_y - new_w // 2) < 0 or int(coords_y + new_w // 2) >= w) \
                else int(coords_y - new_w // 2)
            patch_z = 0 if (int(coords_z - new_d // 2) < 0 or int(coords_z + new_d // 2) >= d) \
                else int(coords_z - new_d // 2)

        # Create new lab
        lab = lab[:,
                  patch_x: patch_x + new_h,
                  patch_y: patch_y + new_w,
                  patch_z: patch_z + new_d]

        lab_cortex = lab_cortex[:,
                  patch_x: patch_x + new_h,
                  patch_y: patch_y + new_w,
                  patch_z: patch_z + new_d]

        return {'lab': lab, 'lab_cortex': lab_cortex,
                'GA': sample['GA'],
                'AS': sample['AS'],
                'gender': sample['gender'],
                'name': sample['name'],
                'seg_aff': sample['seg_aff']}


# ==================================================================================================================== #
#
#  SPLIT LABS
#
# ==================================================================================================================== #
class SplitLabEPRIME(object):
    """
    Split lab eprime
    from:
        0   Background
        1	CSF
        2	Cortical gray matter
        3	White matter
        4	Background (skull area)
        5	Ventricles
        6	Cerebellum + BStem
        7	Deep Gray Matter
    to 7 labels:
        0   0+4 Background
        1	1+5 CSF
        2	2   cGM
        3	3   WM
        4	6   Cerebellum + Bstem
        5	7   dGM + Hippocampi and Amygdala
        6	6   ??? Cerebellum + Bstem
    """

    def __init__(self, n_classes):
        self.name = 'splitlab'
        self.n_classes = n_classes

    def __call__(self, sample):
        lab = sample['lab']

        # Check maximum value is 7
        assert np.max(lab) == 7

        # Get lab size:
        h, w, d = lab.shape[:3]

        # Create self.n_classes different classes
        new_lab = np.zeros((self.n_classes, h, w, d))

        # Treat background differently:
        background = np.zeros_like(lab, dtype=np.float32)
        background[lab == 0] = 1.0
        background[lab == 4] = 1.0
        new_lab[0, :, :, :] = background

        # Treat CSF differently:
        csf = np.zeros_like(lab, dtype=np.float32)
        csf[lab == 1] = 1.0
        csf[lab == 5] = 1.0
        new_lab[1, :, :, :] = csf

        # Treat all the others
        id_ = 0
        id_new = [2, 3, 4, 5, 6]
        for id_lab in [2, 3, 6, 7, 6]:
            lab_temp = np.zeros_like(lab, dtype=np.float32)
            lab_temp[lab == id_lab] = 1.0

            new_lab[id_new[id_], :, :, :] = lab_temp
            id_ += 1

        return {'image': sample['image'],
                'lab': new_lab,
                'GA': sample['GA'],
                'AS': sample['AS'],
                'gender': sample['gender'],
                'name': sample['name'],
                'img_aff': sample['img_aff'],
                'seg_aff': sample['seg_aff']}


class SplitLab(object):
    """
    Split lab
    from:
        0   Background
        1	CSF
        2	Cortical gray matter
        3	White matter
        4	Background (skull area)
        5	Ventricles
        6	Cerebellum
        7	Deep Gray Matter
        8	Brainstem
        9	Hippocampi and Amygdala
    to:
        0   0+4 Background
        1	1   CSF
        2	2   cGM
        3	3   WM
        5	4   Ventricles
        6	5   Cerebellum
        7	6   dGM
        8	7   Brainstem
        9	8   Hippocampi and Amygdala
    or to 7 labels:
        0   0+4 Background
        1	1+5 CSF
        2	2   cGM
        3	3   WM
        4	6   Cerebellum
        5	7+9 dGM + Hippocampi and Amygdala
        6	8   Brainstem
    """

    def __init__(self, n_classes):
        self.name = 'splitlab'
        self.n_classes = n_classes

    def __call__(self, sample):
        lab = sample['lab']

        # Check maximum value is 9
        assert np.max(lab) == 9

        # Get lab size:
        h, w, d = lab.shape[:3]

        if self.n_classes != 7:
            # Create self.n_classes different classes
            new_lab = np.zeros((7, h, w, d))
        else:
            new_lab = np.zeros((self.n_classes, h, w, d))

        # Treat background differently:
        background = np.zeros_like(lab, dtype=np.float32)
        background[lab == 0] = 1.0
        background[lab == 4] = 1.0
        new_lab[0, :, :, :] = background

        # Treat CSF differently:
        csf = np.zeros_like(lab, dtype=np.float32)
        csf[lab == 1] = 1.0
        csf[lab == 5] = 1.0
        new_lab[1, :, :, :] = csf

        # Treat dGM differently:
        dgm = np.zeros_like(lab, dtype=np.float32)
        dgm[lab == 7] = 1.0
        dgm[lab == 9] = 1.0
        new_lab[5, :, :, :] = dgm

        # Treat all the others
        id_ = 0
        id_new = [2, 3, 4, 6]
        for id_lab in [2, 3, 6, 8]:  # [1, 2, 3, 5, 6, 7, 8, 9]:  # [ 7, 2, 3]
            lab_temp = np.zeros_like(lab, dtype=np.float32)
            lab_temp[lab == id_lab] = 1.0

            new_lab[id_new[id_], :, :, :] = lab_temp
            id_ += 1

        # Choose only labels 0, 1, 2, 3, and 5
        if self.n_classes == 5:
            new_lab = np.concatenate((new_lab[0:4, :, :, :],
                                      new_lab[[5], :, :, :]), axis=0)

        elif self.n_classes != 7:
            print("    [!] Invalid number of classes in SplitLab class")
            return 0

        return {'image': sample['image'],
                'lab': new_lab,
                'GA': sample['GA'],
                'AS': sample['AS'],
                'gender': sample['gender'],
                'name': sample['name'],
                'img_aff': sample['img_aff'],
                'seg_aff': sample['seg_aff']}


class SplitLabCortex(object):
    """
    Split lab
    from:
        0   Background
        2	Cortical gray matter
        3	White matter
        4	Background (skull area)
        5	Ventricles
        6	Cerebellum
        7	Deep Gray Matter
        8	Brainstem
        9	Hippocampi and Amygdala
    to:
        0   0+4 Background
        1	2   cGM
    """

    def __init__(self, n_classes, n_cortex_classes=32, is_predicted_folder=False):
        self.name = 'splitlab'
        self.n_classes = n_classes
        self.n_cortex_classes = n_cortex_classes
        self.is_predicted_folder = is_predicted_folder

    def __call__(self, sample):
        lab, subject_name = sample['lab'], sample['name']

        # Get lab size:
        h, w, d = lab.shape[:3]

        # # # # # # # Create new Lab   ---  cortex
        new_lab = np.zeros((self.n_classes, h, w, d))

        # Treat cGM
        cGM = np.zeros_like(lab, dtype=np.float32)
        cGM[lab == 2] = 1.0
        new_lab[1, :, :, :] = cGM

        # Treat background differently:
        background = np.ones_like(lab, dtype=np.float32)
        new_lab[0, :, :, :] = np.abs(background - cGM)


        # Split cortex into the 32 or 11 classes
        if self.n_cortex_classes == 32:
            new_lab_cortex = np.zeros((len(CORTEX_IDS ) +1, h, w, d))
            new_lab_cortex[0, :, :, :] = new_lab[0, :, :, :]
            for ii, id_cortex in enumerate(CORTEX_IDS):
                temp_ = np.zeros_like(sample['lab_cortex'], dtype=np.float32)
                temp_[sample['lab_cortex'] == id_cortex] = 1.0
                new_lab_cortex[ii + 1, :, :, :] = temp_
        else:
            # self.n_cortex_classes == 11:
            new_lab_cortex = np.zeros((len(CORTEX_IDS_11 ) +1, h, w, d))
            new_lab_cortex[0, :, :, :] = new_lab[0, :, :, :]
            for ii, id_cortices in enumerate(CORTEX_IDS_11):
                temp_ = np.zeros_like(sample['lab_cortex'], dtype=np.float32)
                for id_cortex in id_cortices:
                    temp_[sample['lab_cortex'] == id_cortex] = 1.0
                new_lab_cortex[ii + 1, :, :, :] = temp_


        return {'lab': new_lab,
                'lab_cortex': new_lab_cortex,
                'old_lab': lab,
                'GA': sample['GA'],
                'AS': sample['AS'],
                'gender': sample['gender'],
                'name': sample['name'],
                'seg_aff': sample['seg_aff']}

