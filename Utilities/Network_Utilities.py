import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from collections import OrderedDict
import numpy as np
import os
import time
from torch.nn.modules.module import _addindent
from torchvision import models
import datetime



"""============================="""
"""======= Base functions ======"""
"""============================="""
def gimme_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params




"""=============================="""
"""====== Numpy Metrics ========="""
"""=============================="""
def Dice(inp,target, eps=0.000001):
    input_flatten = inp.flatten()
    target_flatten= target.flatten()

    overlap       = np.sum(input_flatten*target_flatten)
    return np.clip(((2.*overlap)/(np.sum(target_flatten)+np.sum(input_flatten)+eps)),1e-4,0.9999)


"""======================================="""
"""============ Loss Provider ============"""
"""======================================="""
class Loss_Provider(nn.Module):
    """
    NOTE: Each loss has the input variable >inp< to denote the data input. If the segmentation classes
    are directly encoded into the numerical mask values, the ground truth input is denoted >target<. If
    this is done via channel information, use >target_one_hot<.
    """
    def __init__(self, opt):
        super(Loss_Provider, self).__init__()
        self.loss_choice    = opt.Training['loss_func']

        if self.loss_choice  =="binary_dice":
            self.loss_func      = binary_dice_loss(opt)
        if self.loss_choice  =="weighted_binary_dice":
            self.loss_func      = weighted_binary_dice_loss(opt)
        elif self.loss_choice=="multiclass_dice":
            self.loss_func      = multiclass_dice_loss(opt)
        elif self.loss_choice=="binary_pwce":
            self.loss_func      = binary_pwce_loss(opt)
        elif self.loss_choice=="multiclass_pwce":
            self.loss_func      = multiclass_pwce_loss(opt)
        elif self.loss_choice=="multiclass_combined":
            self.loss_func      = multiclass_combined_loss(opt)
        elif self.loss_choice=="focal_pwce":
            self.loss_func      = pw_focal_loss(opt)
        elif self.loss_choice=="focal_dice":
            self.loss_func      = dice_focal_loss(opt)
        else:
            raise NotImplementedError("This choice of loss has not been implemented: {}".format(self.loss_choice))

    def forward(self, inp, **kwargs):
        return self.loss_func(inp, **kwargs)




"""================================"""
"""========= Loss Functions ======="""
"""================================"""
class binary_dice_loss(nn.Module):
    """
    Smooth surrogate loss for Binary Dice Score Evaluation.
    """
    def __init__(self, opt):
        super(binary_dice_loss, self).__init__()
        self.epsilon         = opt.Training['epsilon']
        self.require_weightmaps          = False
        self.require_one_hot             = False
        self.require_single_channel_mask  = True
        self.require_single_channel_input = True

        self.pars = opt

    def forward(self, inp, target):
        """
        Arguments:
            inp:    Network prediction of shape (BS,1,W,H) in [0,1]
            target: Ground Truth of size (BS,1,W,H) in [0,1]
        """
        bs          = inp.size()[0]
        inp, target = inp.view(bs,-1).type(torch.FloatTensor).to(self.pars.device), target.view(bs,-1).type(torch.FloatTensor).to(self.pars.device)

        intersection = torch.sum(inp * target,dim=1)+self.epsilon
        union        = torch.sum(inp,dim=1) + torch.sum(target,dim=1)+self.epsilon
        return torch.mean(-2.*torch.clamp(intersection/union,1e-7,0.9999))


class multiclass_dice_loss(nn.Module):
    """
    Surrogate loss for multiclass dice loss by computing the dice score per channel.
    """
    def __init__(self, opt):
        super(multiclass_dice_loss, self).__init__()
        self.epsilon      = opt.Training['epsilon']
        self.weight_score = torch.from_numpy(np.array(opt.Training['weight_score'])).type(torch.FloatTensor).to(opt.device) if opt.Training['weight_score'] is not None else None

        self.require_weightmaps  = False
        self.require_one_hot     = True
        self.require_single_channel_mask  = False
        self.require_single_channel_input = False

        self.pars = opt

    def forward(self, inp, target_one_hot):
        """
        Arguments:
            inp:            Network prediction of shape (BS,C,W,H) in [0,1]
            target_one_hot: Ground Truth of size (BS,Classes,W,H) in [0,1]
        """
        bs,ch = inp.size()[:2]

        inp            = inp.type(torch.FloatTensor).to(self.pars.device).view(bs,ch,-1)
        target_one_hot = target_one_hot.type(torch.FloatTensor).to(self.pars.device).view(bs,ch,-1)

        intersection = torch.sum(inp*target_one_hot, dim=2)
        union        = torch.sum(inp,dim=2) + torch.sum(target_one_hot,dim=2)+self.epsilon
        if self.weight_score is not None:
            weight_score = torch.stack([self.weight_score for _ in range(bs)],dim=0)
            dice_loss    = torch.mean(-1.*torch.mean(2. * intersection*weight_score/union,dim=1))
        else:
            dice_loss    = torch.mean(-1.*torch.mean(2. * intersection/union,dim=1))
        return dice_loss





class weighted_binary_dice_loss(nn.Module):
    """
    Surrogate loss for weighted binary dice loss by computing the dice score per channel.
    """
    def __init__(self, opt):
        super(weighted_binary_dice_loss, self).__init__()
        self.epsilon      = opt.Training['epsilon']
        self.weight_score = torch.from_numpy(np.array(opt.Training['weight_score'])).type(torch.FloatTensor).to(opt.device) if opt.Training['weight_score'] is not None else None
        self.wmap_weight  = opt.Training['wmap_weight']

        self.require_weightmaps  = True
        self.require_one_hot     = False
        self.require_single_channel_mask  = True
        self.require_single_channel_input = True

        self.pars = opt

    def forward(self, inp, target, wmap):
        """
        Arguments:
            inp:            Network prediction of shape (BS,C,W,H) in [0,1]
            target_one_hot: Ground Truth of size (BS,Classes,W,H) in [0,1]
        """

        bs,ch = inp.size()[:2]

        bs          = inp.size()[0]
        inp, target = inp.view(bs,-1).type(torch.FloatTensor).to(self.pars.device), target.view(bs,-1).type(torch.FloatTensor).to(self.pars.device)
        wmap = wmap.view(bs,-1).type(torch.FloatTensor).to(self.pars.device)

        intersection = torch.sum(inp * target * (1+wmap)**self.wmap_weight,dim=1)+self.epsilon
        union        = torch.sum(inp,dim=1) + torch.sum(target,dim=1)+self.epsilon
        dice_loss    = torch.mean(-1.*torch.clamp(2*intersection/union,1e-7,0.9999))

        intersection = torch.sum((1-inp) * (1-target)* (1+wmap)**self.wmap_weight,dim=1)+self.epsilon
        union        = torch.sum(inp,dim=1) + torch.sum(target,dim=1)+self.epsilon
        dice_loss    += torch.mean(-1.*torch.clamp(2*intersection/union,1e-7,0.9999))

        return dice_loss


class binary_pwce_loss(nn.Module):
    """
    Pixel-Weighted For Binary CrossEntropyLoss.
    """
    def __init__(self, opt):
        super(binary_pwce_loss, self).__init__()

        self.wmap_weight   = opt.Training['wmap_weight']
        self.loss          = nn.BCELoss(size_average=False, reduce=False)

        self.require_weightmaps  = True if self.wmap_weight else False
        self.require_one_hot     = False
        self.require_single_channel_mask  = True
        self.require_single_channel_input = True
        self.pars                = opt

    def forward(self, inp, target, wmap=None):
        """
        Arguments:
            inp:    Network predictions of shape (BS,Classes,W,H)
            target: Ground Truth of shape (BS,Classes,W,H)
        """
        bs = inp.size()[0]

        inp     = inp.view(bs,-1).type(torch.FloatTensor).to(self.pars.device)
        target  = target.view(bs,-1).type(torch.FloatTensor).to(self.pars.device)
        wmap    = wmap.view(bs,-1).type(torch.FloatTensor).to(self.pars.device) if wmap is not None else torch.zeros_like(target).to(self.pars.device)

        return torch.mean(self.loss(inp, target)*(wmap+1.)**self.wmap_weight)


class multiclass_pwce_loss(nn.Module):
    """
    Pixel-Weighted Multiclass CrossEntropyLoss.
    """
    def __init__(self, opt):
        super(multiclass_pwce_loss, self).__init__()

        self.pars                = opt

        self.wmap_weight   = opt.Training['wmap_weight']
        self.loss          = nn.CrossEntropyLoss(weight=torch.Tensor(self.pars.Training['class_weights']).to(self.pars.device), size_average=False, reduce=False)

        self.require_weightmaps          = True if self.wmap_weight else False
        self.require_one_hot             = False
        self.require_single_channel_mask  = True
        self.require_single_channel_input = False


    def forward(self, inp, target, wmap=None):
        bs,ch = inp.size()[:2]

        inp     = inp.view(bs,ch,-1).type(torch.FloatTensor).to(self.pars.device)
        target  = target.view(bs,-1).type(torch.LongTensor).to(self.pars.device)
        wmap    = wmap.view(bs,-1).type(torch.FloatTensor).to(self.pars.device) if wmap is not None else torch.zeros_like(target).type(torch.FloatTensor).to(self.pars.device)

        return torch.mean(self.loss(inp, target)*(wmap+1.)**self.wmap_weight)


class multiclass_combined_loss(nn.Module):
    """
    Pixel-Weighted Multiclass CrossEntropyLoss over Multiclass Dice Loss.
    """
    def __init__(self, opt):
        super(multiclass_combined_loss, self).__init__()

        self.wmap_weight   = opt.Training['wmap_weight']
        self.pwce_loss     = multiclass_pwce_loss(opt)
        self.dice_loss     = multiclass_dice_loss(opt)

        self.require_weightmaps           = True if self.wmap_weight else False
        self.require_one_hot              = self.dice_loss.require_one_hot or self.pwce_loss.require_one_hot
        self.require_single_channel_mask  = self.dice_loss.require_single_channel_mask or self.pwce_loss.require_single_channel_mask
        self.require_single_channel_input = self.dice_loss.require_single_channel_input or self.pwce_loss.require_single_channel_input

        self.pars                = opt

    def forward(self, inp, target, target_one_hot, wmap=None):
        pwce_loss = self.pwce_loss(inp, target, wmap)
        dice_loss = self.dice_loss(inp, target_one_hot)

        return pwce_loss/(-1*dice_loss)


class pw_focal_loss(nn.Module):
    def __init__(self, opt):
        super(pw_focal_loss, self).__init__()
        self.gamma       = opt.Training['focal_gamma']
        self.epsilon     = opt.Training['epsilon']
        self.wmap_weight = opt.Training['wmap_weight']

        self.require_weightmaps = True
        self.require_one_hot    = True
        self.require_single_channel_mask  = False
        self.require_single_channel_input = False

        self.pars = opt

    def forward(self, inp, target_one_hot, wmap=None):
        """
        inp:    Input with shape (BS,C,W,H)
        target_one_hot: Target with shape (BS,C,W,H) in [0,1]
        """
        bs,ch = inp.size()[:2]

        inp             = inp.view(bs,ch,-1).type(torch.FloatTensor).to(self.pars.device)
        target_one_hot  = target_one_hot.view(bs,ch,-1).type(torch.FloatTensor).to(self.pars.device)
        if wmap is not None: wmap = wmap.type(torch.FloatTensor).to(self.pars.device)

        inp   = inp.clamp(self.epsilon, 1. - self.epsilon)

        loss = -1 * target_one_hot * torch.log(inp) * (1-inp)**self.gamma
        loss = torch.mean(torch.mean(loss, dim=2),dim=1).view(bs,-1)
        if wmap is not None: loss = loss*(1+wmap.view(bs,-1))**self.wmap_weight
        loss = torch.mean(loss)

        return loss


class dice_focal_loss(nn.Module):
    def __init__(self, opt):
        super(dice_focal_loss, self).__init__()
        self.gamma        = opt.Training['focal_gamma']
        self.epsilon      = opt.Training['epsilon']
        self.weight_score = None

        self.require_weightmaps = False
        self.require_one_hot    = True
        self.require_single_channel_mask  = False
        self.require_single_channel_input = False

        self.pars = opt

    def forward(self, inp, target_one_hot, wmap=None):
        """
        inp:    Input with shape (BS,C,W,H)
        target_one_hot: Target with shape (BS,C,W,H) in [0,1]
        """
        bs,ch = inp.size()[:2]

        inp            = inp.type(torch.FloatTensor).to(self.pars.device).view(bs,ch,-1)
        target_one_hot = target_one_hot.type(torch.FloatTensor).to(self.pars.device).view(bs,ch,-1)
        if wmap is not None: wmap = wmap.type(torch.FloatTensor).to(self.pars.device)

        inp   = inp.clamp(self.epsilon, 1. - self.epsilon)

        intersection = torch.sum(inp*target_one_hot, dim=2)
        union        = torch.sum(inp,dim=2) + torch.sum(target_one_hot,dim=2)+self.epsilon
        if self.weight_score is not None:
            weight_score = torch.stack([self.weight_score for _ in range(bs)],dim=0)
            dice_loss    = torch.mean(-1.*torch.mean(2. * intersection*weight_score/union,dim=1))
        else:
            dice_loss    = torch.mean(-1.*torch.mean(2. * intersection/union,dim=1))



        loss = -1 * target_one_hot * torch.log(inp) * (1-inp)**self.gamma
        loss = torch.mean(torch.mean(loss, dim=2),dim=1).view(bs,-1)
        if wmap is not None: loss = loss*(1+wmap.view(bs,-1))**self.wmap_weight
        loss = torch.mean(loss)

        return loss
