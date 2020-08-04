# -*- coding: utf-8 -*-
"""
# @Time    : May/25/2020
# @Author  : zhx
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from medpy import metric
from utilites.softmax_helper import softmax_torch
from utilites.tensor_utilites import sum_tensor, mean_tensor


def BCDHWtoNC(output, target):
    target = target.long()
    num_classes = output.size()[1]
    # i, j = 1, 2
    # while j < len(output.shape):
    #     output = output.transpose(i, j)
    #     i = j
    #     j += 1
    output = output.permute([0, 2, 3, 4, 1]).contiguous()
    output = output.view(-1, num_classes)  # (N, C)
    target = target.view(-1, )  # (N, )
    assert output.size(0) == target.size(0), "Excepted net output batch_size {} to match target batch_size{}".format(
        output.size(0), target.size(0))
    return output, target


class CrossentropyND(nn.CrossEntropyLoss):
    def forward(self, output, target):
        output, target = BCDHWtoNC(output, target)
        return super(CrossentropyND, self).forward(output, target)


def get_tp_fp_fn_tn(output, target, axes=None, square=False):
    if axes is None:
        axes = list(range(2, len(output.size())))
    output_shape = output.shape  # B, C, D, H, W
    target_shape = target.shape  # B, D, H, W
    with torch.no_grad():
        if len(output_shape) != len(target_shape):
            target = target.view((target_shape[0], 1, *target_shape[1:]))
        target = target.long()
        target_onehot = torch.zeros(output_shape)
        if output.device.type == "cuda":
            target_onehot = target_onehot.cuda(output.device.index)
        target_onehot.scatter_(1, target, 1).float()
    # output.shape  target.shape == B, C, D, H, W
    tp = output * target_onehot
    fp = output * (1 - target_onehot)
    fn = (1 - output) * target_onehot
    tn = (1 - output) * (1 - target_onehot)
    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2
        tn = tn ** 2
    tp = sum_tensor(tp, axes, keepdim=False)
    fp = sum_tensor(fp, axes, keepdim=False)
    fn = sum_tensor(fn, axes, keepdim=False)
    tn = sum_tensor(tn, axes, keepdim=False)
    return tp, fp, fn, tn


class SoftDiceLoss(nn.Module):
    def __init__(self, apply_softmax=None, smooth=1e-5, square=False):
        super(SoftDiceLoss, self).__init__()
        self.apply_softmax = apply_softmax # the last layer is softmax, so this apply_softmax variables is setted False.
        self.square = square
        self.smooth = smooth

    def forward(self, output, target):
        output_shape = output.shape  # B, C, D, H, W

        if self.apply_softmax is not None:
            output = self.apply_softmax(output)

        axes = list(range(2, len(output_shape)))
        tp, fp, fn, _ = get_tp_fp_fn_tn(output, target, axes, square=False)

        nominator = 2 * tp + self.smooth
        denominator = 2 * tp + fp + fn + self.smooth

        dc = nominator / denominator

        dc = dc[:, 1:]  # exclude background dice
        dc = dc.mean()
        return 1-dc


class DiceLossCrossEntropy(nn.Module):
    def __init__(self, soft_dice_kwargs=None, ce_kwargs=None, weight_ce=1, weight_dice=1):
        super(DiceLossCrossEntropy, self).__init__()
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice
        soft_dice_kwargs = soft_dice_kwargs if soft_dice_kwargs is not None else {}
        ce_kwargs = ce_kwargs if ce_kwargs is not None else {}
        self.dc = SoftDiceLoss(apply_softmax=softmax_torch, **soft_dice_kwargs)
        self.ce = CrossentropyND(**ce_kwargs)

    def forward(self, output, target):
        '''
        :param output:  shape: (B, C, D, H, W)
        :param target:  shape: (B, D, H, W)
        :return:
        '''
        dc_loss = self.dc(output, target) if self.weight_dice != 0 else 0
        ce_loss = self.ce(output, target) if self.weight_ce != 0 else 0
        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result


class DiceLossFocalLoss(nn.Module):
    def __init__(self, soft_dice_kwargs=None, fl_kwargs=None, weight_ce=1, weight_dice=1):
        super(DiceLossFocalLoss, self).__init__()
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice
        soft_dice_kwargs = soft_dice_kwargs if soft_dice_kwargs is not None else {}
        fl_kwargs = fl_kwargs if fl_kwargs is not None else {}
        self.dc = SoftDiceLoss(apply_softmax=softmax_torch, **soft_dice_kwargs)
        self.fl = FocalLossMultiClass(**fl_kwargs)

    def forward(self, output, target):
        '''
        :param output:  shape: (B, C, D, H, W)
        :param target:  shape: (B, D, H, W)
        :return:
        '''
        dc_loss = self.dc(output, target) if self.weight_dice != 0 else 0
        fl_loss = self.fl(output, target) if self.weight_ce != 0 else 0
        result = self.weight_ce * fl_loss + self.weight_dice * dc_loss
        return result


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, class_weight=None, apply_softmax=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        if isinstance(class_weight, list) or isinstance(class_weight, tuple):
            class_weight = torch.tensor(class_weight, dtype=torch.float32, requires_grad=False).cuda()
        self.class_weight = class_weight
        self.reduction = reduction
        self.apply_softmax = apply_softmax

    def forward(self, output, target):
        if self.apply_softmax is not None:
            output = self.apply_softmax(output)
        # convert output to pseudo probability
        output, target = BCDHWtoNC(output, target)

        probs = torch.stack([output[i, t] for i, t in enumerate(target)])
        # probs = torch.sigmoid(probs)
        focal_weight = torch.pow(1 - probs, self.gamma)

        # add focal weight to cross entropy
        ce_loss = F.cross_entropy(output, target, weight=self.class_weight, reduction='none')
        focal_loss = focal_weight * ce_loss

        if self.reduction == 'mean':
            focal_loss = (focal_loss / focal_weight.sum()).sum()
        elif self.reduction == 'sum':
            focal_loss = focal_loss.sum()

        return focal_loss


class FocalLossMultiClass(nn.Module):
    def __init__(self, alpha=None, gamma=2, use_alpha=False, size_average=True):
        super(FocalLossMultiClass, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        if use_alpha:
            self.alpha = torch.tensor(alpha).cuda()

        self.softmax = softmax_torch
        self.use_alpha = use_alpha
        self.size_average = size_average

    def forward(self, output, target):
        output = self.softmax(output)
        output, target = BCDHWtoNC(output, target)
        prob = output.clamp(min=0.0001, max=1.0)

        target_ = torch.zeros(target.size(0), output.size(-1)).cuda()
        target_.scatter_(1, target.view(-1, 1).long(), 1.)

        if self.use_alpha:
            batch_loss = - self.alpha.float() * torch.pow(1 - prob,
                                                           self.gamma).float() * prob.log().float() * target_.float()
        else:
            batch_loss = - torch.pow(1 - prob, self.gamma).float() * prob.log().float() * target_.float()

        batch_loss = batch_loss.sum(dim=1)

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()

        return loss


class DC_and_topk_loss(nn.Module):
    def __init__(self, soft_dice_kwargs=None, ce_kwargs=None, aggregate="sum", square_dice=False):
        super(DC_and_topk_loss, self).__init__()
        self.aggregate = aggregate
        soft_dice_kwargs = soft_dice_kwargs if soft_dice_kwargs is not None else {}
        ce_kwargs = ce_kwargs if ce_kwargs is not None else {}
        self.ce = TopKLoss(**ce_kwargs)
        self.dc = SoftDiceLoss(apply_softmax=softmax_torch, **soft_dice_kwargs)

    def forward(self, net_output, target):
        dc_loss = self.dc(net_output, target)
        ce_loss = self.ce(net_output, target)
        if self.aggregate == "sum":
            result = ce_loss + dc_loss
        else:
            raise NotImplementedError("nah son") # reserved for other stuff (later?)
        return result


class TopKLoss(CrossentropyND):
    """
    Network has to have NO LINEARITY!
    """
    def __init__(self, weight=None, ignore_index=-100, k=10):
        self.k = k
        super(TopKLoss, self).__init__(weight, False, ignore_index, reduce=False)

    def forward(self, inp, target):
        target = target[:, 0].long()
        res = super(TopKLoss, self).forward(inp, target)
        num_voxels = np.prod(res.shape, dtype=np.int64)
        res, _ = torch.topk(res.view((-1, )), int(num_voxels * self.k / 100), sorted=False)
        return res.mean()


def dice_coff(output, target, smooth=1):  # output: B,C,D,H,W  target: B,D,H,W
    assert len(output.shape)==5 and len(target.shape)==4, "output shape must B,C,D,H,W, target shape must B,D,H,W, now is {}, {}".format(output.shape, target.shape)
    num_classes = output.shape[1]
    output_softmax = softmax_torch(output)
    output_seg = output_softmax.argmax(1)
    axes = tuple(range(1, len(target.shape)))
    tp_hard = torch.zeros((target.shape[0], num_classes - 1)).to(output_seg.device.index)
    fp_hard = torch.zeros((target.shape[0], num_classes - 1)).to(output_seg.device.index)
    fn_hard = torch.zeros((target.shape[0], num_classes - 1)).to(output_seg.device.index)
    for c in range(1, num_classes):
        tp_hard[:, c - 1] = sum_tensor((output_seg == c).float() * (target == c).float(), axes=axes)
        fp_hard[:, c - 1] = sum_tensor((output_seg == c).float() * (target != c).float(), axes=axes)
        fn_hard[:, c - 1] = sum_tensor((output_seg != c).float() * (target == c).float(), axes=axes)

    tp_hard = tp_hard.sum(0, keepdim=False).detach().cpu().numpy()
    fp_hard = fp_hard.sum(0, keepdim=False).detach().cpu().numpy()
    fn_hard = fn_hard.sum(0, keepdim=False).detach().cpu().numpy()

    fore_ground_dc = list((2 * tp_hard + smooth) / (2 * tp_hard + fp_hard + fn_hard + smooth))
    tp_hard = list(tp_hard)
    fp_hard = list(fp_hard)
    fn_hard = list(fn_hard)
    return fore_ground_dc, tp_hard, fp_hard, fn_hard


class ConfusionMatrix:
    def __init__(self, prediction=None, reference=None):

        self.tp = None
        self.fp = None
        self.tn = None
        self.fn = None
        self.size = None
        self.reference_empty = None
        self.reference_full = None
        self.prediction_empty = None
        self.prediction_full = None
        self.set_reference(reference)
        self.set_prediction(prediction)

    def set_prediction(self, prediction):
        self.prediction = prediction
        self.reset()

    def set_reference(self, reference):
        self.reference = reference
        self.reset()

    def reset(self):

        self.tp = None
        self.fp = None
        self.tn = None
        self.fn = None
        self.size = None
        self.prediction_empty = None
        self.prediction_full = None
        self.reference_empty = None
        self.reference_full = None

    def compute(self):

        if self.prediction is None or self.reference is None:
            raise ValueError("'prediction' and 'reference' must both be set to compute confusion matrix.")

        assert self.prediction.shape == self.reference.shape, "Shape mismatch: {} and {}".format(self.prediction.shape, self.reference.shape)

        self.tp = int(((self.prediction != 0) * (self.reference != 0)).sum())
        self.fp = int(((self.prediction != 0) * (self.reference == 0)).sum())
        self.tn = int(((self.prediction == 0) * (self.reference == 0)).sum())
        self.fn = int(((self.prediction == 0) * (self.reference != 0)).sum())
        self.size = int(np.prod(self.reference.shape, dtype=np.int64))
        self.prediction_empty = not np.any(self.prediction)
        self.prediction_full = np.all(self.prediction)
        self.reference_empty = not np.any(self.reference)
        self.reference_full = np.all(self.reference)

    def get_matrix(self):

        for entry in (self.tp, self.fp, self.tn, self.fn):
            if entry is None:
                self.compute()
                break

        return self.tp, self.fp, self.tn, self.fn

    def get_size(self):

        if self.size is None:
            self.compute()
        return self.size

    def get_existence(self):

        for case in (self.prediction_empty, self.prediction_full, self.reference_empty, self.reference_full):
            if case is None:
                self.compute()
                break

        return self.prediction_empty, self.prediction_full, self.reference_empty, self.reference_full


def dice(prediction=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """2TP / (2TP + FP + FN)"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(prediction, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()
    prediction_empty, prediction_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if prediction_empty and reference_empty:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0.

    return float(2. * tp / (2 * tp + fp + fn))


def jaccard(prediction=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """TP / (TP + FP + FN)"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(prediction, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()
    prediction_empty, prediction_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if prediction_empty and reference_empty:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0.

    return float(tp / (tp + fp + fn))


def precision(prediction=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """TP / (TP + FP)"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(prediction, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()
    prediction_empty, prediction_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if prediction_empty:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0.

    return float(tp / (tp + fp))


def sensitivity(prediction=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """TP / (TP + FN)"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(prediction, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()
    prediction_empty, prediction_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if reference_empty:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0.

    return float(tp / (tp + fn))


def recall(prediction=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """TP / (TP + FN)"""

    return sensitivity(prediction, reference, confusion_matrix, nan_for_nonexisting, **kwargs)


def specificity(prediction=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """TN / (TN + FP)"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(prediction, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()
    prediction_empty, prediction_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if reference_full:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0.

    return float(tn / (tn + fp))


def accuracy(prediction=None, reference=None, confusion_matrix=None, **kwargs):
    """(TP + TN) / (TP + FP + FN + TN)"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(prediction, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()

    return float((tp + tn) / (tp + fp + tn + fn))


def fscore(prediction=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, beta=1., **kwargs):
    """(1 + b^2) * TP / ((1 + b^2) * TP + b^2 * FN + FP)"""

    precision_ = precision(prediction, reference, confusion_matrix, nan_for_nonexisting)
    recall_ = recall(prediction, reference, confusion_matrix, nan_for_nonexisting)

    return (1 + beta*beta) * precision_ * recall_ /\
        ((beta*beta * precision_) + recall_)


def false_positive_rate(prediction=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """FP / (FP + TN)"""

    return 1 - specificity(prediction, reference, confusion_matrix, nan_for_nonexisting)


def false_omission_rate(prediction=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """FN / (TN + FN)"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(prediction, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()
    prediction_empty, prediction_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if prediction_full:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0.

    return float(fn / (fn + tn))


def false_negative_rate(prediction=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """FN / (TP + FN)"""

    return 1 - sensitivity(prediction, reference, confusion_matrix, nan_for_nonexisting)


def true_negative_rate(prediction=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """TN / (TN + FP)"""

    return specificity(prediction, reference, confusion_matrix, nan_for_nonexisting)


def false_discovery_rate(prediction=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """FP / (TP + FP)"""

    return 1 - precision(prediction, reference, confusion_matrix, nan_for_nonexisting)


def negative_predictive_value(prediction=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """TN / (TN + FN)"""

    return 1 - false_omission_rate(prediction, reference, confusion_matrix, nan_for_nonexisting)


def total_positives_prediction(prediction=None, reference=None, confusion_matrix=None, **kwargs):
    """TP + FP"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(prediction, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()

    return tp + fp


def total_negatives_prediction(prediction=None, reference=None, confusion_matrix=None, **kwargs):
    """TN + FN"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(prediction, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()

    return tn + fn


def total_positives_reference(prediction=None, reference=None, confusion_matrix=None, **kwargs):
    """TP + FN"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(prediction, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()

    return tp + fn


def total_negatives_reference(prediction=None, reference=None, confusion_matrix=None, **kwargs):
    """TN + FP"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(prediction, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()

    return tn + fp


def hausdorff_distance(prediction=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, voxel_spacing=None, connectivity=1, **kwargs):

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(prediction, reference)

    prediction_empty, prediction_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if prediction_empty or prediction_full or reference_empty or reference_full:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0

    prediction, reference = confusion_matrix.prediction, confusion_matrix.reference

    return metric.hd(prediction, reference, voxel_spacing, connectivity)


def hausdorff_distance_95(prediction=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, voxel_spacing=None, connectivity=1, **kwargs):

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(prediction, reference)

    prediction_empty, prediction_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if prediction_empty or prediction_full or reference_empty or reference_full:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0

    prediction, reference = confusion_matrix.prediction, confusion_matrix.reference

    return metric.hd95(prediction, reference, voxel_spacing, connectivity)


def avg_surface_distance(prediction=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, voxel_spacing=None, connectivity=1, **kwargs):

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(prediction, reference)

    prediction_empty, prediction_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if prediction_empty or prediction_full or reference_empty or reference_full:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0

    prediction, reference = confusion_matrix.prediction, confusion_matrix.reference

    return metric.asd(prediction, reference, voxel_spacing, connectivity)


def avg_surface_distance_symmetric(prediction=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, voxel_spacing=None, connectivity=1, **kwargs):

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(prediction, reference)

    prediction_empty, prediction_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if prediction_empty or prediction_full or reference_empty or reference_full:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0

    prediction, reference = confusion_matrix.prediction, confusion_matrix.reference

    return metric.assd(prediction, reference, voxel_spacing, connectivity)


ALL_METRICS = {
    "False Positive Rate": false_positive_rate,
    "Dice": dice,
    "Jaccard": jaccard,
    "Hausdorff Distance": hausdorff_distance,
    "Hausdorff Distance 95": hausdorff_distance_95,
    "Precision": precision,
    "Recall": recall,
    "Avg. Symmetric Surface Distance": avg_surface_distance_symmetric,
    "Avg. Surface Distance": avg_surface_distance,
    "Accuracy": accuracy,
    "False Omission Rate": false_omission_rate,
    "Negative Predictive Value": negative_predictive_value,
    "False Negative Rate": false_negative_rate,
    "True Negative Rate": true_negative_rate,
    "False Discovery Rate": false_discovery_rate,
    "Total Positives Test": total_positives_prediction,
    "Total Negatives Test": total_negatives_prediction,
    "Total Positives Reference": total_positives_reference,
    "total Negatives Reference": total_negatives_reference
}


if __name__ == '__main__':
    from network_architecture.APLSNet import APLSNet
    net = APLSNet(1, 6, [32, 40, 48, 56]).cuda()
    loss = DiceLossFocalLoss()
    inputs = torch.randn(1, 1, 64, 92, 92, dtype=torch.float32, requires_grad=True).cuda()
    outputs = net(inputs)
    targets = torch.randint(6, size=(1, 64, 92, 92), dtype=torch.float32).cuda()
    ls = loss(outputs, targets)
    ls.backward()
    print(ls)
