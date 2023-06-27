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
# losses.py
#
##############################################################################
import torch


def gd_loss(pred_seg, target_seg, include_background=False, weighted=True, eps=1e-6):
    """
    Generalised Dice Loss

    :param pred_seg:
    :param target_seg:
    :param include_background:
    :param eps:
    :return:
    """
    ba_size = target_seg.size(0)
    n_class = target_seg.size(1)

    ground = target_seg.float().view(ba_size, n_class, -1)
    pred = pred_seg.view(ba_size, n_class, -1)

    score = 0.0

    if include_background:
        id_start = 0
    else:
        id_start = 1

    for batch in range(0, ba_size):
        b_ground = ground[batch, id_start:]
        b_pred = pred[batch, id_start:]

        b_ground_sum = torch.sum(b_ground, dim=1).view(-1, 1)       # dimensions to reduce

        if weighted:
            weights = torch.reciprocal(b_ground_sum.pow(2))
            weights = torch.clamp(weights, 1e-12, 1.0)
        else:
            weights = torch.ones_like(b_ground_sum)

        intersection = b_ground * b_pred
        intersection = weights * intersection

        sums = b_ground + b_pred
        sums = weights * sums
        score = score + (2.0 * intersection.sum(1) + eps) / (sums.sum(1) + eps)


    return 1.0 - score.mean() / ba_size



def ncc(pred_img, target_img, eps=1e-5):
    """
    Global NCC loss
    :param pred_img:
    :param target_img:
    :param eps:
    :return:
    """
    y_true_dm = target_img - torch.mean(target_img)
    y_pred_dm = pred_img - torch.mean(pred_img)

    ncc_num = torch.sum(y_true_dm * y_pred_dm)
    ncc_den = torch.sqrt(torch.sum(torch.pow(y_true_dm, 2)) * torch.sum(torch.pow(y_pred_dm, 2)) + eps)

    return 1.0 - ncc_num / ncc_den