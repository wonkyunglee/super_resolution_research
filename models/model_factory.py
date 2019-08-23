import os, sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
sys.path.append('../')
from models.model_block import RelationalLayer
from utils.checkpoint import get_last_checkpoint

device = None


class FSRCNN(nn.Module):
    def __init__(self, scale, n_colors, d=56, s=12, m_1=4, m_2=3, dilation=1):
        super(FSRCNN, self).__init__()


        self.scale = scale
        upscale_factor = scale
        d_padding = dilation -1

        self.feature_extraction = []
        self.feature_extraction.append(nn.Sequential(
            nn.Conv2d(in_channels=n_colors,
                      out_channels=d, kernel_size=3, stride=1, padding=1+d_padding,
                      dilation=dilation),
            nn.PReLU(),
            nn.Conv2d(in_channels=d, out_channels=d,
                      kernel_size=3, stride=1, padding=1+d_padding,
                      dilation=dilation),
            nn.PReLU()))

        self.shrinking = []
        self.shrinking.append(nn.Sequential(
            nn.Conv2d(in_channels=d, out_channels=s,
                      kernel_size=1, stride=1, padding=0),
            nn.PReLU()))

        self.mapping = []
        for _ in range(m_1):
            self.mapping.append(nn.Sequential(
                nn.Conv2d(in_channels=s, out_channels=s,
                          kernel_size=3, stride=1, padding=1+d_padding,
                          dilation=dilation),
                nn.PReLU()))

        self.expanding = []
        self.expanding.append(nn.Sequential(
            nn.Conv2d(in_channels=s, out_channels=d,
                      kernel_size=1, stride=1, padding=0),
            nn.PReLU()))


        self.last_layer = []
        self.last_layer.append(nn.Sequential(
            #nn.Conv2d(d, n_colors, kernel_size=9, stride=1, padding=4))
            nn.Conv2d(d, n_colors, kernel_size=3, stride=1, padding=1))
        )

        self.network = nn.Sequential(
            OrderedDict([
                ('feature_extraction', nn.Sequential(*self.feature_extraction)),
                ('shrinking', nn.Sequential(*self.shrinking)),
                ('mapping', nn.Sequential(*self.mapping)),
                ('expanding', nn.Sequential(*self.expanding)),
                ('last_layer', nn.Sequential(*self.last_layer)),
            ]))


    def forward(self, x):
        return self.network(x)



class ModifiedFSRCNN(nn.Module):
    def __init__(self, scale, n_colors, d=56, s=12, m_1=4, m_2=3, dilation=1):
        super(ModifiedFSRCNN, self).__init__()


        self.scale = scale
        upscale_factor = scale
        d_padding = dilation -1

        self.feature_extraction = []
        self.feature_extraction.append(nn.Sequential(
            nn.Conv2d(in_channels=n_colors,
                      out_channels=d, kernel_size=3, stride=1, padding=1+d_padding,
                      dilation=dilation),
            nn.PReLU(),
            nn.Conv2d(in_channels=d, out_channels=d,
                      kernel_size=3, stride=1, padding=1+d_padding,
                      dilation=dilation),
            nn.PReLU()))

        self.shirinking = []
        self.shirinking.append(nn.Sequential(
            nn.Conv2d(in_channels=d, out_channels=s,
                      kernel_size=1, stride=1, padding=0),
            nn.PReLU()))

        self.mapping = []
        for _ in range(m_1):
            self.mapping.append(nn.Sequential(
                nn.Conv2d(in_channels=s, out_channels=s,
                          kernel_size=3, stride=1, padding=1+d_padding,
                          dilation=dilation),
                nn.PReLU()))

        self.expanding = []
        self.expanding.append(nn.Sequential(
            nn.Conv2d(in_channels=s, out_channels=d,
                      kernel_size=1, stride=1, padding=0),
            nn.PReLU()))


        self.expanding.append(nn.Sequential(
            nn.Conv2d(in_channels=d, out_channels=s,
                      kernel_size=1, stride=1, padding=0),
            nn.PReLU()))

        self.mapping2 = []
        ## last layer has the 4-depth layers
        for _ in range(m_2):
            self.mapping2.append(nn.Sequential(
                nn.Conv2d(in_channels=s, out_channels=s,
                          kernel_size=3, stride=1, padding=1),
                nn.PReLU()))

        self.expanding2 = []
        self.expanding2.append(nn.Sequential(
            nn.Conv2d(in_channels=s, out_channels=d,
                       kernel_size=1, stride=1,
                       padding=0),
            nn.PReLU(),
            nn.Conv2d(d, 1, kernel_size=3, stride=1, padding=1)))

        self.network = nn.Sequential(
            OrderedDict([
                ('feature_extraction', nn.Sequential(*self.feature_extraction)),
                ('shirinking', nn.Sequential(*self.shirinking)),
                ('mapping', nn.Sequential(*self.mapping)),
                ('expanding', nn.Sequential(*self.expanding)),
                ('mapping2', nn.Sequential(*self.mapping2)),
                ('expanding2', nn.Sequential(*self.expanding2))
            ]))


    def forward(self, x):
        return self.network(x)



class RelationalFSRCNN(nn.Module):
    def __init__(self, scale, n_colors, d=56, s=12, m_1=4, m_2=3,
                 dilation=1, relational_kernel_size=3, layer_num=2):
        super(RelationalFSRCNN, self).__init__()

        self.scale = scale
        upscale_factor = scale
        d_padding = dilation -1
        rk = relational_kernel_size
        rp = relational_kernel_size // 2 # padding
        ln = layer_num

        self.feature_extraction = []
        self.feature_extraction.append(nn.Sequential(
            nn.Conv2d(in_channels=n_colors,
                      out_channels=d, kernel_size=3, stride=1, padding=1+d_padding,
                      dilation=dilation),
            nn.PReLU(),
            nn.Conv2d(in_channels=d, out_channels=d,
                      kernel_size=3, stride=1, padding=1+d_padding,
                      dilation=dilation),
            nn.PReLU()))

        self.shrinking = []
        self.shrinking.append(nn.Sequential(
            nn.Conv2d(in_channels=d, out_channels=s,
                      kernel_size=1, stride=1, padding=0),
            nn.PReLU()))

        self.mapping = []
        for _ in range(m_1):
            self.mapping.append(nn.Sequential(
                nn.Conv2d(in_channels=s, out_channels=s,
                          kernel_size=3, stride=1, padding=1+d_padding,
                          dilation=dilation),
                nn.PReLU(),
                RelationalLayer(kernel_size=rk, padding=rp, channel_num=s,
                                layer_num=ln)
            ))

        self.expanding = []
        self.expanding.append(nn.Sequential(
            nn.Conv2d(in_channels=s, out_channels=d,
                      kernel_size=1, stride=1, padding=0),
            nn.PReLU()))


        self.last_layer = []
        self.last_layer.append(nn.Sequential(
            #nn.Conv2d(d, n_colors, kernel_size=9, stride=1, padding=4))
            nn.Conv2d(d, n_colors, kernel_size=3, stride=1, padding=1))
        )

        self.network = nn.Sequential(
            OrderedDict([
                ('feature_extraction', nn.Sequential(*self.feature_extraction)),
                ('shrinking', nn.Sequential(*self.shrinking)),
                ('mapping', nn.Sequential(*self.mapping)),
                ('expanding', nn.Sequential(*self.expanding)),
                ('last_layer', nn.Sequential(*self.last_layer)),
            ]))


    def forward(self, x):
        return self.network(x)


class RelationalLastFSRCNN(nn.Module):
    def __init__(self, scale, n_colors, d=56, s=12, m_1=4, m_2=3,
                 dilation=1, relational_kernel_size=3, layer_num=2):
        super(RelationalLastFSRCNN, self).__init__()

        self.scale = scale
        upscale_factor = scale
        d_padding = dilation -1
        rk = relational_kernel_size
        rp = relational_kernel_size // 2 # padding
        ln = layer_num

        self.feature_extraction = []
        self.feature_extraction.append(nn.Sequential(
            nn.Conv2d(in_channels=n_colors,
                      out_channels=d, kernel_size=3, stride=1, padding=1+d_padding,
                      dilation=dilation),
            nn.PReLU(),
            nn.Conv2d(in_channels=d, out_channels=d,
                      kernel_size=3, stride=1, padding=1+d_padding,
                      dilation=dilation),
            nn.PReLU()))

        self.shrinking = []
        self.shrinking.append(nn.Sequential(
            nn.Conv2d(in_channels=d, out_channels=s,
                      kernel_size=1, stride=1, padding=0),
            nn.PReLU()))

        self.mapping = []
        for _ in range(m_1):
            self.mapping.append(nn.Sequential(
                nn.Conv2d(in_channels=s, out_channels=s,
                          kernel_size=3, stride=1, padding=1+d_padding,
                          dilation=dilation),
                nn.PReLU()
            ))

        self.expanding = []
        self.expanding.append(nn.Sequential(
            nn.Conv2d(in_channels=s, out_channels=d,
                      kernel_size=1, stride=1, padding=0),
            nn.PReLU()))


        self.last_layer = []
        self.last_layer.append(nn.Sequential(
            #nn.Conv2d(d, n_colors, kernel_size=9, stride=1, padding=4))
            nn.Conv2d(d, n_colors, kernel_size=3, stride=1, padding=1),
            RelationalLayer(kernel_size=rk, padding=rp, channel_num=n_colors,
                            layer_num=ln))
        )

        self.network = nn.Sequential(
            OrderedDict([
                ('feature_extraction', nn.Sequential(*self.feature_extraction)),
                ('shrinking', nn.Sequential(*self.shrinking)),
                ('mapping', nn.Sequential(*self.mapping)),
                ('expanding', nn.Sequential(*self.expanding)),
                ('last_layer', nn.Sequential(*self.last_layer)),
            ]))


    def forward(self, x):
        return self.network(x)


class RelationalMiddleFSRCNN(nn.Module):
    def __init__(self, scale, n_colors, d=56, s=12, m_1=4, m_2=3,
                 dilation=1, relational_kernel_size=3, layer_num=2):
        super(RelationalMiddleFSRCNN, self).__init__()

        self.scale = scale
        upscale_factor = scale
        d_padding = dilation -1
        rk = relational_kernel_size
        rp = relational_kernel_size // 2 # padding
        ln = layer_num

        self.feature_extraction = []
        self.feature_extraction.append(nn.Sequential(
            nn.Conv2d(in_channels=n_colors,
                      out_channels=d, kernel_size=3, stride=1, padding=1+d_padding,
                      dilation=dilation),
            nn.PReLU(),
            nn.Conv2d(in_channels=d, out_channels=d,
                      kernel_size=3, stride=1, padding=1+d_padding,
                      dilation=dilation),
            nn.PReLU()))

        self.shrinking = []
        self.shrinking.append(nn.Sequential(
            nn.Conv2d(in_channels=d, out_channels=s,
                      kernel_size=1, stride=1, padding=0),
            nn.PReLU()))

        self.mapping = []
        for _ in range(m_1):
            self.mapping.append(nn.Sequential(
                nn.Conv2d(in_channels=s, out_channels=s,
                          kernel_size=3, stride=1, padding=1+d_padding,
                          dilation=dilation),
                nn.PReLU()
            ))
        self.mapping.append(nn.Sequential(
            RelationalLayer(kernel_size=rk, padding=rp, channel_num=s,
                            layer_num=ln)
        ))

        self.expanding = []
        self.expanding.append(nn.Sequential(
            nn.Conv2d(in_channels=s, out_channels=d,
                      kernel_size=1, stride=1, padding=0),
            nn.PReLU()))


        self.last_layer = []
        self.last_layer.append(nn.Sequential(
            #nn.Conv2d(d, n_colors, kernel_size=9, stride=1, padding=4))
            nn.Conv2d(d, n_colors, kernel_size=3, stride=1, padding=1)
        ))

        self.network = nn.Sequential(
            OrderedDict([
                ('feature_extraction', nn.Sequential(*self.feature_extraction)),
                ('shrinking', nn.Sequential(*self.shrinking)),
                ('mapping', nn.Sequential(*self.mapping)),
                ('expanding', nn.Sequential(*self.expanding)),
                ('last_layer', nn.Sequential(*self.last_layer)),
            ]))


    def forward(self, x):
        return self.network(x)

class BaseNet(nn.Module):

    def __init__(self):
        super(BaseNet, self).__init__()
        self.backbone = None
        self.modules_to_freeze = None
        self.initialize_from = None
        self.modules_to_initialize = None


    def weight_init(self, mean=0.0, std=0.001):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(mean, std)
                if m.bias is not None:
                    m.bias.data.zero_()


    def freeze_modules(self):
        for k, m in self.backbone.network._modules.items():
            if k in self.modules_to_freeze:
                for param in m.parameters():
                    paramrequires_grad = False
                print('freezing layer: %s'%k)


    def load_pretrained_model(self):

        if type(self.initialize_from) != list:
            self.initialize_from = [self.initialize_from]
            self.modules_to_initialize = [self.modules_to_initialize]

        for init_from, modules_to_init in zip(self.initialize_from, self.modules_to_initialize):
            print(init_from)
            checkpoint = get_last_checkpoint(init_from)
            checkpoint = torch.load(checkpoint)
            new_state_dict = self.state_dict()
            for key in checkpoint['state_dict'].keys():
                for k in key.split('.'):
                    if k in modules_to_init:
                        new_state_dict[key] = checkpoint['state_dict'][key]
                        print('pretrain parameters: %s'%k)
            self.load_state_dict(new_state_dict)


class DisentangleTeacherNet(BaseNet):
    def __init__(self, scale, n_colors, modules_to_freeze=None, initialize_from=None,
                 modules_to_initialize=None, dilation=1):
        super(DisentangleTeacherNet, self).__init__()

        self.scale = scale
        d = 56
        s = 12
        m_1 = 4
        m_2 = 3

        self.modules_to_freeze = modules_to_freeze
        self.initialize_from = initialize_from
        self.modules_to_initialize = modules_to_initialize
        self.backbone = ModifiedFSRCNN(scale, n_colors, d, s, m_1, m_2, dilation)
        self.weight_init()
        if initialize_from is not None:
            self.load_pretrained_model()
        if modules_to_freeze is not None:
            self.freeze_modules()


    def forward(self, LR, HR):
        ret_dict = dict()

        x = HR
        layer_names = self.backbone.network._modules.keys()
        for layer_name in layer_names:
            x = self.backbone.network._modules[layer_name](x)
            ret_dict[layer_name] = x

        residual_hr = x
        LR = nn.functional.interpolate(LR, scale_factor=self.scale,
                                        mode='bicubic')
        hr = LR + residual_hr
        ret_dict['hr'] = hr
        ret_dict['residual_hr'] = residual_hr
        return ret_dict


class DisentangleStudentNet(BaseNet):
    def __init__(self, scale, n_colors, modules_to_freeze=None, initialize_from=None,
                 modules_to_initialize=None, dilation=1):
        super(DisentangleStudentNet, self).__init__()

        self.scale = scale
        upscale_factor = scale
        d = 56
        s = 12
        m_1 = 4
        m_2 = 3
        self.initialize_from = initialize_from
        self.modules_to_freeze = modules_to_freeze
        self.modules_to_initialize = modules_to_initialize

        self.backbone = ModifiedFSRCNN(scale, n_colors, d, s, m_1, m_2, dilation)
        self.weight_init()
        if initialize_from is not None:
            self.load_pretrained_model()
        if modules_to_freeze is not None:
            self.freeze_modules(modules_to_freeze)


    def forward(self, LR, teacher_pred_dict=None):
        ret_dict = dict()
        upscaled_lr = nn.functional.interpolate(LR, scale_factor=self.scale, mode='bicubic')
        x = upscaled_lr

        layer_names = self.backbone.network._modules.keys()
        for layer_name in layer_names:
            x = self.backbone.network._modules[layer_name](x)
            ret_dict[layer_name] = x

        residual_hr = x

        hr = upscaled_lr + residual_hr
        ret_dict['hr'] = hr
        ret_dict['residual_hr'] = residual_hr
        return ret_dict



class AttendSimilarityTeacherNet(BaseNet):
    def __init__(self, scale, n_colors,  d=56, s=12, m_1=4, m_2=3,
                 modules_to_freeze=None, initialize_from=None, modules_to_initialize=None,
                 dilation=1):
        super(AttendSimilarityTeacherNet, self).__init__()

        self.scale = scale

        self.initialize_from = initialize_from
        self.modules_to_initialize = modules_to_initialize
        self.modeuls_to_freeze = modules_to_freeze

        self.backbone = ModifiedFSRCNN(scale, n_colors, d, s, m_1, m_2, dilation)
        self.weight_init()
        if initialize_from is not None:
            self.load_pretrained_model()
        if modules_to_freeze is not None:
            self.freeze_modules()


    def forward(self, LR, HR):
        ret_dict = dict()

        x = HR
        layer_names = self.backbone.network._modules.keys()
        for layer_name in layer_names:
            x = self.backbone.network._modules[layer_name](x)
            ret_dict[layer_name] = x

        residual_hr = x
        LR = nn.functional.interpolate(LR, scale_factor=self.scale,
                                        mode='bicubic')
        hr = LR + residual_hr
        ret_dict['hr'] = hr
        ret_dict['residual_hr'] = residual_hr
        return ret_dict


class AttendSimilarityStudentNet(BaseNet):
    def __init__(self, scale, n_colors, d=56, s=12, m_1=4, m_2=3,layers_to_attend=None, modules_to_freeze=None,
                 initialize_from=None, modules_to_initialize=None, dilation=1):
        super(AttendSimilarityStudentNet, self).__init__()

        self.layers_to_attend = layers_to_attend if layers_to_attend is not None else []
        self.scale = scale
        upscale_factor = scale

        self.initialize_from = initialize_from
        self.modules_to_freeze = modules_to_freeze
        self.modules_to_initialize = modules_to_initialize

        self.backbone = ModifiedFSRCNN(scale, n_colors, d, s, m_1, m_2, dilation)
        self.weight_init()
        if initialize_from is not None:
            self.load_pretrained_model()
        if modules_to_freeze is not None:
            self.freeze_modules()


    def forward(self, LR, teacher_pred_dict=None):
        ret_dict = dict()
        upscaled_lr = nn.functional.interpolate(LR, scale_factor=self.scale, mode='bicubic')
        x = upscaled_lr

        layer_names = self.backbone.network._modules.keys()
        for layer_name in layer_names:
            x = self.backbone.network._modules[layer_name](x)
            ret_dict[layer_name] = x

        residual_hr = x
        hr = upscaled_lr + residual_hr
        ret_dict['hr'] = hr
        ret_dict['residual_hr'] = residual_hr

        return ret_dict


class NoisyTeacherNet(BaseNet):
    def __init__(self, scale, n_colors, d=56, s=12, m_1=4, m_2=3,layers_to_attend=None, modules_to_freeze=None,
                 initialize_from=None, modules_to_initialize=None, dilation=1, noise_offset=10, distance='l1'):
        super(NoisyTeacherNet, self).__init__()

        self.layers_to_attend = layers_to_attend if layers_to_attend is not None else []
        self.scale = scale
        upscale_factor = scale

        self.initialize_from = initialize_from
        self.modules_to_freeze = modules_to_freeze
        self.modules_to_initialize = modules_to_initialize

        self.backbone = ModifiedFSRCNN(scale, n_colors, d, s, m_1, m_2, dilation)
        self.weight_init()
        if initialize_from is not None:
            self.load_pretrained_model()
        if modules_to_freeze is not None:
            self.freeze_modules()
        self.noise_offset = noise_offset
        self.distance = distance


    def get_cos_similarity(self, x, y):
        dist = F.cosine_similarity(x, y, dim=1).unsqueeze(1)
        dist = (dist + 1.0) / 2.01
        return dist

    def get_l1_dist(self, x, y):
        dist = torch.abs(x - y)
        return dist


    def forward(self, LR, HR, student_pred_dict=None):
        ret_dict = dict()
        upscaled_lr = nn.functional.interpolate(LR, scale_factor=self.scale, mode='bicubic')
        x = HR

        layer_names = self.backbone.network._modules.keys()
        for layer_name in layer_names:
            if student_pred_dict is not None and layer_name in self.layers_to_attend:
                teacher_x = self.backbone.network._modules[layer_name](x)
                student_x = student_pred_dict[layer_name]
                if self.distance == 'cos':
                    dist = self.get_cos_similarity(student_x, teacher_x).detach()
                    std = (1 - dist) * self.noise_offset
                elif self.distance == 'l1':
                    dist = self.get_l1_dist(student_x, teacher_x).detach()
                    std = dist * self.noise_offset
                print(torch.mean(std))
                x = teacher_x + torch.randn_like(teacher_x).to(device) * std
            else:
                x = self.backbone.network._modules[layer_name](x)
            ret_dict[layer_name] = x

        residual_hr = x
        hr = upscaled_lr + residual_hr
        ret_dict['hr'] = hr
        ret_dict['residual_hr'] = residual_hr

        return ret_dict


class GTNoisyTeacherNet(BaseNet):
    def __init__(self, scale, n_colors, d=56, s=12, m_1=4, m_2=3,layers_to_attend=None, modules_to_freeze=None,
                 initialize_from=None, modules_to_initialize=None, dilation=1, noise_offset=10, distance='l1'):
        super(GTNoisyTeacherNet, self).__init__()

        self.layers_to_attend = layers_to_attend if layers_to_attend is not None else []
        self.scale = scale
        upscale_factor = scale

        self.initialize_from = initialize_from
        self.modules_to_freeze = modules_to_freeze
        self.modules_to_initialize = modules_to_initialize

        self.backbone = FSRCNN(scale, n_colors, d, s, m_1, m_2, dilation)
        self.weight_init()
        if initialize_from is not None:
            self.load_pretrained_model()
        if modules_to_freeze is not None:
            self.freeze_modules()
        self.noise_offset = noise_offset
        self.distance = distance


    def get_cos_similarity(self, x, y):
        dist = F.cosine_similarity(x, y, dim=1).unsqueeze(1)
        dist = (dist + 1.0) / 2.01
        return dist

    def get_l1_dist(self, x, y):
        dist = torch.abs(x - y)
        return dist


    def forward(self, LR, HR, **_):
        ret_dict = dict()
        upscaled_lr = nn.functional.interpolate(LR, scale_factor=self.scale, mode='bicubic')
        diff = self.get_l1_dist(upscaled_lr, HR)
        std = diff * self.noise_offset
        x = HR

        layer_names = self.backbone.network._modules.keys()
        for layer_name in layer_names:
            if layer_name in self.layers_to_attend:
                teacher_x = self.backbone.network._modules[layer_name](x)
                x = teacher_x
                if self.training:
                    noise = torch.randn_like(teacher_x).to(device) * std
                    x += noise
            else:
                x = self.backbone.network._modules[layer_name](x)
            ret_dict[layer_name] = x

        residual_hr = x
        hr = upscaled_lr + residual_hr
        ret_dict['hr'] = hr
        ret_dict['residual_hr'] = residual_hr

        return ret_dict


class FSRCNNTeacherNet(BaseNet):
    def __init__(self, scale, n_colors,  d=56, s=12, m_1=4, m_2=3,
                 modules_to_freeze=None, initialize_from=None, modules_to_initialize=None,
                 dilation=1):
        super(FSRCNNTeacherNet, self).__init__()

        self.scale = scale

        self.initialize_from = initialize_from
        self.modules_to_initialize = modules_to_initialize
        self.modeuls_to_freeze = modules_to_freeze

        self.backbone = FSRCNN(scale, n_colors, d, s, m_1, m_2, dilation)
        self.weight_init()
        if initialize_from is not None:
            self.load_pretrained_model()
        if modules_to_freeze is not None:
            self.freeze_modules()


    def forward(self, LR, HR):
        ret_dict = dict()

        x = HR
        layer_names = self.backbone.network._modules.keys()
        for layer_name in layer_names:
            x = self.backbone.network._modules[layer_name](x)
            ret_dict[layer_name] = x

        residual_hr = x
        LR = nn.functional.interpolate(LR, scale_factor=self.scale,
                                        mode='bicubic')
        hr = LR + residual_hr
        ret_dict['hr'] = hr
        ret_dict['residual_hr'] = residual_hr
        return ret_dict


class FSRCNNStudentNet(BaseNet):
    def __init__(self, scale, n_colors, d=56, s=12, m_1=4, m_2=3,layers_to_attend=None, modules_to_freeze=None,
                 initialize_from=None, modules_to_initialize=None, dilation=1):
        super(FSRCNNStudentNet, self).__init__()

        self.layers_to_attend = layers_to_attend if layers_to_attend is not None else []
        self.scale = scale
        upscale_factor = scale

        self.initialize_from = initialize_from
        self.modules_to_freeze = modules_to_freeze
        self.modules_to_initialize = modules_to_initialize

        self.backbone = FSRCNN(scale, n_colors, d, s, m_1, m_2, dilation)
        self.weight_init()
        if initialize_from is not None:
            self.load_pretrained_model()
        if modules_to_freeze is not None:
            self.freeze_modules()


    def forward(self, LR, teacher_pred_dict=None):
        ret_dict = dict()
        upscaled_lr = nn.functional.interpolate(LR, scale_factor=self.scale, mode='bicubic')
        x = upscaled_lr

        layer_names = self.backbone.network._modules.keys()
        for layer_name in layer_names:
            x = self.backbone.network._modules[layer_name](x)
            ret_dict[layer_name] = x

        residual_hr = x
        hr = upscaled_lr + residual_hr
        ret_dict['hr'] = hr
        ret_dict['residual_hr'] = residual_hr

        return ret_dict


class GTNoisyStudentNet(BaseNet):
    def __init__(self, scale, n_colors, d=56, s=12, m_1=4, m_2=3,layers_to_attend=None, modules_to_freeze=None,
                 initialize_from=None, modules_to_initialize=None, dilation=1, noise_offset=10, distance='l1'):
        super(GTNoisyStudentNet, self).__init__()

        self.layers_to_attend = layers_to_attend if layers_to_attend is not None else []
        self.scale = scale
        upscale_factor = scale

        self.initialize_from = initialize_from
        self.modules_to_freeze = modules_to_freeze
        self.modules_to_initialize = modules_to_initialize

        self.backbone = FSRCNN(scale, n_colors, d, s, m_1, m_2, dilation)
        self.weight_init()
        if initialize_from is not None:
            self.load_pretrained_model()
        if modules_to_freeze is not None:
            self.freeze_modules()
        self.noise_offset = noise_offset
        self.distance = distance


    def get_cos_similarity(self, x, y):
        dist = F.cosine_similarity(x, y, dim=1).unsqueeze(1)
        dist = (dist + 1.0) / 2.01
        return dist

    def get_l1_dist(self, x, y):
        dist = torch.abs(x - y)
        return dist


    def forward(self, LR, HR, **_):
        ret_dict = dict()
        upscaled_lr = nn.functional.interpolate(LR, scale_factor=self.scale, mode='bicubic')
        diff = self.get_l1_dist(upscaled_lr, HR)
        std = diff * self.noise_offset
        x = upscaled_lr

        layer_names = self.backbone.network._modules.keys()
        for layer_name in layer_names:
            if layer_name in self.layers_to_attend:
                student_x = self.backbone.network._modules[layer_name](x)
                x = student_x
                if self.training:
                    noise = torch.randn_like(student_x).to(device) * std
                    x += noise
            else:
                x = self.backbone.network._modules[layer_name](x)
            ret_dict[layer_name] = x

        residual_hr = x
        hr = upscaled_lr + residual_hr
        ret_dict['hr'] = hr
        ret_dict['residual_hr'] = residual_hr

        return ret_dict


class ConstNoisyStudentNet(BaseNet):
    def __init__(self, scale, n_colors, d=56, s=12, m_1=4, m_2=3,layers_to_attend=None, modules_to_freeze=None,
                 initialize_from=None, modules_to_initialize=None, dilation=1, noise_offset=0.01, distance='l1'):
        super(ConstNoisyStudentNet, self).__init__()

        self.layers_to_attend = layers_to_attend if layers_to_attend is not None else []
        self.scale = scale
        upscale_factor = scale

        self.initialize_from = initialize_from
        self.modules_to_freeze = modules_to_freeze
        self.modules_to_initialize = modules_to_initialize

        self.backbone = FSRCNN(scale, n_colors, d, s, m_1, m_2, dilation)
        self.weight_init()
        if initialize_from is not None:
            self.load_pretrained_model()
        if modules_to_freeze is not None:
            self.freeze_modules()
        self.noise_offset = noise_offset
        self.distance = distance


    def get_cos_similarity(self, x, y):
        dist = F.cosine_similarity(x, y, dim=1).unsqueeze(1)
        dist = (dist + 1.0) / 2.01
        return dist

    def get_l1_dist(self, x, y):
        dist = torch.abs(x - y)
        return dist


    def forward(self, LR, **_):
        ret_dict = dict()
        upscaled_lr = nn.functional.interpolate(LR, scale_factor=self.scale, mode='bicubic')
        std = self.noise_offset
        x = upscaled_lr

        layer_names = self.backbone.network._modules.keys()
        for layer_name in layer_names:
            if layer_name in self.layers_to_attend:
                student_x = self.backbone.network._modules[layer_name](x)
                x = student_x
                if self.training:
                    noise = torch.randn_like(student_x).to(device) * std
                    x += noise
            else:
                x = self.backbone.network._modules[layer_name](x)
            ret_dict[layer_name] = x

        residual_hr = x
        hr = upscaled_lr + residual_hr
        ret_dict['hr'] = hr
        ret_dict['residual_hr'] = residual_hr

        return ret_dict


class SelectiveGTNoisyStudentNet(ConstNoisyStudentNet):
    def __init__(self, scale, n_colors, d=56, s=12, m_1=4, m_2=3,layers_to_attend=None, modules_to_freeze=None,
                 initialize_from=None, modules_to_initialize=None, dilation=1, noise_offset=0.01, distance='l1'):
        super(SelectiveGTNoisyStudentNet, self).__init__(scale, n_colors,
                                                            d, s, m_1, m_2,
                                                            layers_to_attend, modules_to_freeze,
                                                            initialize_from, modules_to_initialize,
                                                            dilation, noise_offset, distance)

    def forward(self, LR, HR, **_):
        ret_dict = dict()
        upscaled_lr = nn.functional.interpolate(LR, scale_factor=self.scale, mode='bicubic')
        diff = self.get_l1_dist(upscaled_lr, HR)
        std = diff * self.noise_offset
        x = upscaled_lr

        layer_names = self.backbone.network._modules.keys()
        for layer_name in layer_names:
            if layer_name in self.layers_to_attend:
                student_x = self.backbone.network._modules[layer_name](x)
                noise = torch.randn_like(student_x).to(device) * std
                x = student_x
                if np.random.rand() > 0.5:
                    x += noise
            else:
                x = self.backbone.network._modules[layer_name](x)
            ret_dict[layer_name] = x

        residual_hr = x
        hr = upscaled_lr + residual_hr
        ret_dict['hr'] = hr
        ret_dict['residual_hr'] = residual_hr

        return ret_dict


class SelectiveGTNoisyStudentNet(ConstNoisyStudentNet):
    def __init__(self, scale, n_colors, d=56, s=12, m_1=4, m_2=3,layers_to_attend=None, modules_to_freeze=None,
                 initialize_from=None, modules_to_initialize=None, dilation=1, noise_offset=0.01, distance='l1'):
        super(SelectiveGTNoisyStudentNet, self).__init__(scale, n_colors,
                                                            d, s, m_1, m_2,
                                                            layers_to_attend, modules_to_freeze,
                                                            initialize_from, modules_to_initialize,
                                                            dilation, noise_offset, distance)

    def forward(self, LR, HR, **_):
        ret_dict = dict()
        upscaled_lr = nn.functional.interpolate(LR, scale_factor=self.scale, mode='bicubic')
        diff = self.get_l1_dist(upscaled_lr, HR)
        std = diff * self.noise_offset
        x = upscaled_lr

        layer_names = self.backbone.network._modules.keys()
        for layer_name in layer_names:
            if layer_name in self.layers_to_attend:
                student_x = self.backbone.network._modules[layer_name](x)
                noise = torch.randn_like(student_x).to(device) * std
                x = student_x
                if np.random.rand() > 0.5:
                    x += noise
            else:
                x = self.backbone.network._modules[layer_name](x)
            ret_dict[layer_name] = x

        residual_hr = x
        hr = upscaled_lr + residual_hr
        ret_dict['hr'] = hr
        ret_dict['residual_hr'] = residual_hr

        return ret_dict


class RFSRCNNStudentNet(BaseNet):
    def __init__(self, scale, n_colors, d=56, s=12, m_1=4, m_2=3,layers_to_attend=None, modules_to_freeze=None,
                 initialize_from=None, modules_to_initialize=None, dilation=1, relational_kernel_size=3, layer_num=2):
        super(RFSRCNNStudentNet, self).__init__()

        self.layers_to_attend = layers_to_attend if layers_to_attend is not None else []
        self.scale = scale
        upscale_factor = scale

        self.initialize_from = initialize_from
        self.modules_to_freeze = modules_to_freeze
        self.modules_to_initialize = modules_to_initialize

        self.backbone = RelationalFSRCNN(scale, n_colors, d, s, m_1, m_2,
                                         dilation, relational_kernel_size, layer_num)
        self.weight_init()
        if initialize_from is not None:
            self.load_pretrained_model()
        if modules_to_freeze is not None:
            self.freeze_modules()


    def forward(self, LR, teacher_pred_dict=None):
        ret_dict = dict()
        upscaled_lr = nn.functional.interpolate(LR, scale_factor=self.scale, mode='bicubic')
        x = upscaled_lr

        layer_names = self.backbone.network._modules.keys()
        for layer_name in layer_names:
            x = self.backbone.network._modules[layer_name](x)
            ret_dict[layer_name] = x

        residual_hr = x
        hr = upscaled_lr + residual_hr
        ret_dict['hr'] = hr
        ret_dict['residual_hr'] = residual_hr

        return ret_dict

class RLFSRCNNStudentNet(BaseNet):
    def __init__(self, scale, n_colors, d=56, s=12, m_1=4, m_2=3,layers_to_attend=None, modules_to_freeze=None,
                 initialize_from=None, modules_to_initialize=None, dilation=1, relational_kernel_size=3, layer_num=2):
        super(RLFSRCNNStudentNet, self).__init__()

        self.layers_to_attend = layers_to_attend if layers_to_attend is not None else []
        self.scale = scale
        upscale_factor = scale

        self.initialize_from = initialize_from
        self.modules_to_freeze = modules_to_freeze
        self.modules_to_initialize = modules_to_initialize

        self.backbone = RelationalLastFSRCNN(scale, n_colors, d, s, m_1, m_2,
                                         dilation, relational_kernel_size, layer_num)
        self.weight_init()
        if initialize_from is not None:
            self.load_pretrained_model()
        if modules_to_freeze is not None:
            self.freeze_modules()


    def forward(self, LR, teacher_pred_dict=None):
        ret_dict = dict()
        upscaled_lr = nn.functional.interpolate(LR, scale_factor=self.scale, mode='bicubic')
        x = upscaled_lr

        layer_names = self.backbone.network._modules.keys()
        for layer_name in layer_names:
            x = self.backbone.network._modules[layer_name](x)
            ret_dict[layer_name] = x

        residual_hr = x
        hr = upscaled_lr + residual_hr
        ret_dict['hr'] = hr
        ret_dict['residual_hr'] = residual_hr

        return ret_dict


class RMFSRCNNStudentNet(BaseNet):
    def __init__(self, scale, n_colors, d=56, s=12, m_1=4, m_2=3,layers_to_attend=None, modules_to_freeze=None,
                 initialize_from=None, modules_to_initialize=None, dilation=1, relational_kernel_size=3, layer_num=2):
        super(RMFSRCNNStudentNet, self).__init__()

        self.layers_to_attend = layers_to_attend if layers_to_attend is not None else []
        self.scale = scale
        upscale_factor = scale

        self.initialize_from = initialize_from
        self.modules_to_freeze = modules_to_freeze
        self.modules_to_initialize = modules_to_initialize

        self.backbone = RelationalMiddleFSRCNN(scale, n_colors, d, s, m_1, m_2,
                                         dilation, relational_kernel_size, layer_num)
        self.weight_init()
        if initialize_from is not None:
            self.load_pretrained_model()
        if modules_to_freeze is not None:
            self.freeze_modules()


    def forward(self, LR, teacher_pred_dict=None):
        ret_dict = dict()
        upscaled_lr = nn.functional.interpolate(LR, scale_factor=self.scale, mode='bicubic')
        x = upscaled_lr

        layer_names = self.backbone.network._modules.keys()
        for layer_name in layer_names:
            x = self.backbone.network._modules[layer_name](x)
            ret_dict[layer_name] = x

        residual_hr = x
        hr = upscaled_lr + residual_hr
        ret_dict['hr'] = hr
        ret_dict['residual_hr'] = residual_hr

        return ret_dict



# For Resolution Disentangling Experiments
def get_disentangle_student(scale, n_colors, **kwargs):
    return DisentangleStudentNet(scale, n_colors, **kwargs)


def get_disentangle_teacher(scale, n_colors, **kwargs):
    return DisentangleTeacherNet(scale, n_colors, **kwargs)


def get_attend_similarity_teacher(scale, n_colors, **kwargs):
    return AttendSimilarityTeacherNet(scale, n_colors, **kwargs)


def get_attend_similarity_student(scale, n_colors, **kwargs):
    return AttendSimilarityStudentNet(scale, n_colors, **kwargs)


def get_noisy_teacher(scale, n_colors, **kwargs):
    return NoisyTeacherNet(scale, n_colors, **kwargs)


def get_gt_noisy_teacher(scale, n_colors, **kwargs):
    return GTNoisyTeacherNet(scale, n_colors, **kwargs)


def get_fsrcnn_teacher(scale, n_colors, **kwargs):
    return FSRCNNTeacherNet(scale, n_colors, **kwargs)


def get_fsrcnn_student(scale, n_colors, **kwargs):
    return FSRCNNStudentNet(scale, n_colors, **kwargs)


def get_gt_noisy_student(scale, n_colors, **kwargs):
    return GTNoisyStudentNet(scale, n_colors, **kwargs)


def get_const_noisy_student(scale, n_colors, **kwargs):
    return ConstNoisyStudentNet(scale, n_colors, **kwargs)


def get_selective_gt_noisy_student(scale, n_colors, **kwargs):
    return SelectiveGTNoisyStudentNet(scale, n_colors, **kwargs)


def get_rfsrcnn_student(scale, n_colors, **kwargs):
    return RFSRCNNStudentNet(scale, n_colors, **kwargs)


def get_rlfsrcnn_student(scale, n_colors, **kwargs):
    return RLFSRCNNStudentNet(scale, n_colors, **kwargs)


def get_rmfsrcnn_student(scale, n_colors, **kwargs):
    return RMFSRCNNStudentNet(scale, n_colors, **kwargs)


def get_model(config, model_type):

    global device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print('model name:', config[model_type+'_model'].name)

    f = globals().get('get_' + config[model_type+'_model'].name)
    if config[model_type+'_model'].params is None:
        return f()
    else:
        return f(**config[model_type+'_model'].params)



