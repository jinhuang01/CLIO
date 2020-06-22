import torch.nn as nn
import torch
import numpy as np
from typing import List, Tuple
import torch.nn.functional as F
from ..utils import box_utils
from collections import namedtuple
#GraphPath = namedtuple("GraphPath", ['s0', 'name', 's1'])  #
from ..nn.mobilenet_v2 import MobileNetV2, InvertedResidual
from .mobilenet_v2_ssd_lite import SeperableConv2d
import copy
from torch.nn import Conv2d, Sequential, ModuleList, ModuleDict, BatchNorm2d
import numpy as np


def get_extras():
    extras = ModuleList([
        InvertedResidual(1280, 512, stride=2, expand_ratio=0.2),
        InvertedResidual(512, 256, stride=2, expand_ratio=0.25),
        InvertedResidual(256, 256, stride=2, expand_ratio=0.5),
        InvertedResidual(256, 64, stride=2, expand_ratio=0.25)
    ])
    extras.apply(_xavier_init_)
    return extras

def get_headers(width_mult, num_classes):
    regression_headers = ModuleList([
        SeperableConv2d(in_channels=round(576 * width_mult), out_channels=6 * 4, kernel_size=3, padding=1, onnx_compatible=False),
        SeperableConv2d(in_channels=1280, out_channels=6 * 4, kernel_size=3, padding=1, onnx_compatible=False),
        SeperableConv2d(in_channels=512, out_channels=6 * 4, kernel_size=3, padding=1, onnx_compatible=False),
        SeperableConv2d(in_channels=256, out_channels=6 * 4, kernel_size=3, padding=1, onnx_compatible=False),
        SeperableConv2d(in_channels=256, out_channels=6 * 4, kernel_size=3, padding=1, onnx_compatible=False),
        Conv2d(in_channels=64, out_channels=6 * 4, kernel_size=1),
    ])
    classification_headers = ModuleList([
        SeperableConv2d(in_channels=round(576 * width_mult), out_channels=6 * num_classes, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=1280, out_channels=6 * num_classes, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=512, out_channels=6 * num_classes, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=256, out_channels=6 * num_classes, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=256, out_channels=6 * num_classes, kernel_size=3, padding=1),
        Conv2d(in_channels=64, out_channels=6 * num_classes, kernel_size=1),
    ])
    regression_headers.apply(_xavier_init_)
    classification_headers.apply(_xavier_init_)
    return regression_headers, classification_headers

def get_simple_shared_layers(pool_pixels=False):
    shared_layers = []
    if pool_pixels:
        shared_layers.append(nn.MaxPool2d(2))
    shared_layers.extend([
        nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
        nn.ReLU(),
        nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=False),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 24, kernel_size=3, padding=1, bias=False),
        nn.ReLU(),
        nn.Conv2d(24, 32, kernel_size=3, padding=1, bias=False),
        nn.ReLU(),
    ])
    if not pool_pixels:
        shared_layers.append(nn.MaxPool2d(2))
    shared_layers = nn.Sequential(*shared_layers)
    return shared_layers

def shared_layers_halving(first_width=32, pool_pixels=False):
    shared_layers = []
    if pool_pixels:
        shared_layers.append(nn.MaxPool2d(2))
    shared_layers.append(nn.Conv2d(1, first_width, kernel_size=3, padding=1, bias=False))
    shared_layers.append(nn.ReLU())
    shared_layers.append(nn.MaxPool2d(2))

    shared_layers.append(nn.Conv2d(first_width, 16, kernel_size=3, padding=1, bias=False))
    shared_layers.append(nn.ReLU())
    shared_layers.append(nn.MaxPool2d(2))

    shared_layers.append(nn.Conv2d(16, 8, kernel_size=3, padding=1, bias=False))
    shared_layers.append(nn.ReLU())
    if not pool_pixels:
        shared_layers.append(nn.MaxPool2d(2))
    
    shared_layers = nn.Sequential(*shared_layers)
    return shared_layers

class MultiHeadedSSD(nn.Module):
    #use_rand_width: in the case were we have multiple possible weights, if true we randomly choose a width to use for each training batch
    #widths: the list of widths we support
    #single_channel_input: True if the input is 1 channel (i.e. grayscale)
    #onnx_compatible: mainly replaces relu with relu6
    #remove_batchnorm: if true we remove the batchnorm layers from the early layers of MobileNetV2
    #use_simple_shared_layers: use the Conv2D/Relu/Maxpool layers rather than the MobileNetV2 layers
    #is_finetune: if true we are fine tuning the classification_headers, so everything else will be frozen
    def __init__(self, num_classes, is_test=False, config=None, device=None,
            width_mult=1.0, use_batch_norm=True, split_point=5, 
            widths=[32], use_rand_width=False, base_net_param_path=None,
            single_channel_input=True, onnx_compatible=True, remove_batchnorm=False,
            shared_layer_conf='simple', is_finetune=False):
        super(MultiHeadedSSD, self).__init__()
        self.num_classes = num_classes
        self.is_test = is_test
        self.config = config
        self.split_point = split_point
        self.widths = widths
        if len(self.widths) == 1:
            self.default_width = self.widths[0]
        else:
            self.default_width = -1 #all possible channels
        self.use_rand_width = use_rand_width
        self.is_finetune = is_finetune
        
        #load the MobileNetV2 backbone
        mb_net = MobileNetV2(
            width_mult=width_mult,
            use_batch_norm=use_batch_norm, 
            onnx_compatible=onnx_compatible
        ).features
        if base_net_param_path is not None:
            mb_net.load_state_dict(torch.load(base_net_param_path, map_location=lambda storage, loc: storage), strict=True)

        
        #if we are assuming a single channel input, we must flatten the weights of the early MobileNetV2 layers
        #this code doesn't matter if we are using the simple shared layers
        if single_channel_input:
            conv_weights = mb_net[0][0].weight.detach()
            new_layer = torch.nn.Conv2d(1, 32, kernel_size=(3,3), stride=(2,2), padding=(1,1), bias=False)
            sd = {'weight': conv_weights.sum(dim=1, keepdims=True)}
            new_layer.load_state_dict(sd)
            mb_net[0][0] = new_layer
        #self.mb_net = mb_net

        if shared_layer_conf != 'mbnetv2':
            self.batchnorms = nn.ModuleDict()
            for w in self.widths:
                self.batchnorms[str(w)] = nn.BatchNorm2d(w)
        else:
            self.batchnorms = None

        if shared_layer_conf == 'simple':
            self.shared_layers = get_simple_shared_layers(False)
        elif shared_layer_conf == 'simple_pool_pixels':
            self.shared_layers = get_simple_shared_layers(True)
        elif shared_layer_conf == 'halving_32_pool_pixels':
            self.shared_layers = shared_layers_halving(32, True)
        elif shared_layer_conf == 'halving_32':
            self.shared_layers = shared_layers_halving(32, False)
        elif shared_layer_conf == 'halving_24_pool_pixels':
            self.shared_layers = shared_layers_halving(32, True)
        elif shared_layer_conf == 'halving_24':
            self.shared_layers = shared_layers_halving(24, False)
        else:
            print('DEFAULTING TO MBNETV2 LAYERS')
            self.shared_layers = mb_net[0:split_point+1]
        
        #copy the later layers of MobileNetV2 
        #create a copy for each size in width
        self.heads = nn.ModuleDict()
        in_channels = mb_net[split_point].conv[6].in_channels
        for w in self.widths:
            base_net = copy.deepcopy(mb_net)
            # base_net[split_point].conv[6] = torch.nn.Conv2d(in_channels, w, kernel_size=1, stride=1, padding=0, bias=False)
            # torch.nn.init.xavier_uniform_(base_net[split_point].conv[6].weight)
            # base_net[split_point].conv[7] = torch.nn.BatchNorm2d(w)
            # base_net[split_point].use_res_connect = False
            out_channels = base_net[split_point + 1].conv[0].out_channels
            base_net[split_point + 1].conv[0] = torch.nn.Conv2d(w, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
            torch.nn.init.xavier_uniform_(base_net[split_point + 1].conv[0].weight)
            base_net[split_point + 1].use_res_connect = False
            extras = get_extras()
            regression_headers, classification_headers = get_headers(width_mult, num_classes)
            base_net = base_net[split_point + 1:] #delete shared layers
            self.heads[str(w)] = ModuleList([base_net, regression_headers, classification_headers, extras])

         
        #remove the batchnorm layers from the early layers of MobileNetV2
        #remove_batchnorm must be false if we are using the simple shared layers
        if remove_batchnorm and shared_layer_conf == 'mbnetv2':
            del self.shared_layers[0][1]
            del self.shared_layers[1].conv[-1]
            del self.shared_layers[1].conv[1]
            for i in range(2, len(self.shared_layers)):
                del self.shared_layers[i].conv[-1]
                del self.shared_layers[i].conv[4]
                del self.shared_layers[i].conv[1]

       
        if device:
            self.device = device
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if is_test:
            self.config = config
            self.priors = config.priors.to(self.device)
   
    def forward(self, x, width=None, freeze_base=False):
        #choose a width to split the output of the shared layers to
        if width is None:
            width = self.default_width
            if self.use_rand_width: #choose randomly (for training)
                width = np.random.choice(self.widths)

        #if we are finetuning, freeze the shared layers
        if self.is_finetune:
            self.shared_layers.eval()
            for param in self.shared_layers.parameters():
                param.requires_grad = False
        shared_feats = self.shared_layers(x)
        shared_feats = shared_feats[:, 0:width, :, :] #slice to the desired width
        return self.forward_from_features(shared_feats, freeze_base)

    def forward_from_features(self, feats, freeze_base=False, add_batch_norm=True):
        x = feats
        _, w, _, _ = x.shape
        if add_batch_norm:
            bn_layer = self.batchnorms[str(w)]
            feats = bn_layer(feats)
            #feats = self.shared_layers[1](feats)

        confidences, locations = [], []
        base_net, regression_headers, classification_headers, extras = self.heads[str(w)]
        
        #freeze_base is used when we only want to freeze the MobileNetV2 backbone
        #this is used when we are training a model with the simple shared layers
        for param in base_net.parameters():
            param.requires_grad = not freeze_base
        
        #if we are fine tuning, freeze everything except the classification_headers
        if self.is_finetune:
            base_net.eval()
            for param in base_net.parameters():
                param.requires_grad = False
            regression_headers.eval()
            for param in regression_headers.parameters():
                param.requires_grad = False
            extras.eval()
            for param in extras.parameters():
                param.requires_grad = False
            self.shared_layers.eval()
            for param in self.shared_layers.parameters():
                param.requires_grad = False
            # classification_headers.eval() 
            # for param in classification_headers.parameters():
                # param.requires_grad = False

        
        offset = self.split_point + 1
        x = base_net[0: 14 - offset](x)
        sub = getattr(base_net[14 - offset], 'conv')
        x = sub[:3](x)
        #orinigal code renames to y here, may not matter but leaving the same to ensure gradient flow isnt effected
        y = x
        x = sub[3:](x)
        confidence, location = self.compute_header(classification_headers[0], regression_headers[0], y)
        confidences.append(confidence)
        locations.append(location)

        x = base_net[15 - offset : 19 - offset](x)
        confidence, location = self.compute_header(classification_headers[1], regression_headers[1], x)
        confidences.append(confidence)
        locations.append(location)

        
        for i, layer in enumerate(extras):
            x = layer(x)
            confidence, location = self.compute_header(classification_headers[i+2], regression_headers[i+2], x)
            confidences.append(confidence)
            locations.append(location)
        
        #final results of the forward pass
        #bbox locations with softmax probs
        confidences = torch.cat(confidences, 1)
        locations = torch.cat(locations, 1)
        
        if self.is_test:
            confidences = F.softmax(confidences, dim=2)
            boxes = box_utils.convert_locations_to_boxes(
                locations, self.priors, self.config.center_variance, self.config.size_variance
            )
            boxes = box_utils.center_form_to_corner_form(boxes)
            return confidences, boxes
        else:
            return confidences, locations

    def compute_header(self, classification_header, regression_header, x):
        confidence = classification_header(x)
        confidence = confidence.permute(0, 2, 3, 1).contiguous()
        confidence = confidence.view(confidence.size(0), -1, self.num_classes)
        location = regression_header(x)
        location = location.permute(0, 2, 3, 1).contiguous()
        location = location.view(location.size(0), -1, 4)
        return confidence, location

    #hold over from oringal ssd code, not used
    def compute_header_by_idx(self, i, x):
        confidence = self.classification_headers[i](x)
        confidence = confidence.permute(0, 2, 3, 1).contiguous()
        confidence = confidence.view(confidence.size(0), -1, self.num_classes)
        location = self.regression_headers[i](x)
        location = location.permute(0, 2, 3, 1).contiguous()
        location = location.view(location.size(0), -1, 4)
        return confidence, location

    def init_from_base_net(self, model):
        pass
        # self.base_net.load_state_dict(torch.load(model, map_location=lambda storage, loc: storage), strict=True)
        # self.source_layer_add_ons.apply(_xavier_init_)
        # self.extras.apply(_xavier_init_)
        # self.classification_headers.apply(_xavier_init_)
        # self.regression_headers.apply(_xavier_init_)

    def init_from_pretrained_ssd(self, model):
        pass
        # state_dict = torch.load(model, map_location=lambda storage, loc: storage)
        # state_dict = {k: v for k, v in state_dict.items() if not (k.startswith("classification_headers") or k.startswith("regression_headers"))}
        # model_dict = self.state_dict()
        # model_dict.update(state_dict)
        # self.load_state_dict(model_dict)
        # self.classification_headers.apply(_xavier_init_)
        # self.regression_headers.apply(_xavier_init_)

    def init(self):
        pass
        # self.base_net.apply(_xavier_init_)
        # self.source_layer_add_ons.apply(_xavier_init_)
        # self.extras.apply(_xavier_init_)
        # self.classification_headers.apply(_xavier_init_)
        # self.regression_headers.apply(_xavier_init_)

    def load(self, model):
        save_dict = torch.load(model, map_location=lambda storage, loc: storage)
        self.load_state_dict(save_dict['state_dict'])

    def save(self, model_path):
        torch.save(self.state_dict(), model_path)


class MatchPrior(object):
    def __init__(self, center_form_priors, center_variance, size_variance, iou_threshold):
        self.center_form_priors = center_form_priors
        self.corner_form_priors = box_utils.center_form_to_corner_form(center_form_priors)
        self.center_variance = center_variance
        self.size_variance = size_variance
        self.iou_threshold = iou_threshold

    def __call__(self, gt_boxes, gt_labels):
        if type(gt_boxes) is np.ndarray:
            gt_boxes = torch.from_numpy(gt_boxes)
        if type(gt_labels) is np.ndarray:
            gt_labels = torch.from_numpy(gt_labels)
        boxes, labels = box_utils.assign_priors(gt_boxes, gt_labels,
                                                self.corner_form_priors, self.iou_threshold)
        boxes = box_utils.corner_form_to_center_form(boxes)
        locations = box_utils.convert_boxes_to_locations(boxes, self.center_form_priors, self.center_variance, self.size_variance)
        return locations, labels


def _xavier_init_(m: nn.Module):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
