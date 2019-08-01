import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import sys
sys.path.append('../')
from utils.checkpoint import get_last_checkpoint

    
class ModifiedFSRCNN(nn.Module):
    def __init__(self, scale, n_colors, d=56, s=12, m_1=4, m_2=3):
        super(ModifiedFSRCNN, self).__init__()

        
        self.scale = scale
        upscale_factor = scale
                
        self.feature_extraction = []
        self.feature_extraction.append(nn.Sequential(
            nn.Conv2d(in_channels=n_colors, 
                      out_channels=d, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(in_channels=d, out_channels=d,
                      kernel_size=3, stride=1, padding=1),
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
                          kernel_size=3, stride=1, padding=1),
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
            nn.PReLU()))
                
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
    
    
class BaseNet(nn.Module):
    
    def __init__(self):
        super(BaseNet, self).__init__()
        self.backbone = None
        
    
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
                    
                    
    def freeze_modules(self, modules_to_freeze):
        for k, m in self.backbone.network._modules.items():
            if k in modules_to_freeze:
                for param in m.parameters():
                    paramrequires_grad = False


    def load_pretrained_model(self, initialize_from, modules_to_initialize):
        
        checkpoint = get_last_checkpoint(initialize_from)
        checkpoint = torch.load(checkpoint)
        new_state_dict = self.state_dict()
        for key in checkpoint['state_dict'].keys():
            for k in key.split('.'):
                if k in modules_to_initialize:
                    new_state_dict[key] = checkpoint['state_dict'][key]
        self.load_state_dict(new_state_dict)
    

class DisentangleTeacherNet(BaseNet):
    def __init__(self, scale, n_colors, modules_to_freeze=None, initialize_from=None, modules_to_initialize=None):
        super(DisentangleTeacherNet, self).__init__()

        self.scale = scale
        d = 56
        s = 12
        m_1 = 4
        m_2 = 3
        
        self.backbone = ModifiedFSRCNN(scale, n_colors, d, s, m_1, m_2)
        self.last_layer = nn.Conv2d(d, 1, kernel_size=3, stride=1, padding=1)
        self.weight_init()
        if initialize_from is not None:
            self.load_pretrained_model(initialize_from, modules_to_initialize)
        if modules_to_freeze is not None:
            self.freeze_modules(modules_to_freeze)
        
        
    def forward(self, LR, HR):
        ret_dict = dict()
        
        x = HR
        layer_names = self.backbone.network._modules.keys()
        for layer_name in layer_names:
            x = self.backbone.network._modules[layer_name](x)
            ret_dict[layer_name] = x
        
        residual_hr = self.last_layer(x)
        LR = nn.functional.interpolate(LR, scale_factor=self.scale,
                                        mode='bicubic')
        hr = LR + residual_hr
        ret_dict['hr'] = hr
        ret_dict['residual_hr'] = residual_hr
        return ret_dict
    
        
class DisentangleStudentNet(BaseNet):
    def __init__(self, scale, n_colors, modules_to_freeze=None, initialize_from=None, modules_to_initialize=None):
        super(DisentangleStudentNet, self).__init__()

        self.scale = scale
        upscale_factor = scale
        d = 56
        s = 12
        m_1 = 4
        m_2 = 3

        self.backbone = ModifiedFSRCNN(scale, n_colors, d, s, m_1, m_2)
        self.last_layer = nn.Conv2d(d, 1, kernel_size=3, stride=1, padding=1)
        self.weight_init()
        if initialize_from is not None:
            self.load_pretrained_model(initialize_from, modules_to_initialize)
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
        
        residual_hr = self.last_layer(x)
        
        hr = upscaled_lr + residual_hr
        ret_dict['hr'] = hr
        ret_dict['residual_hr'] = residual_hr
        return ret_dict

    
    
class AttendSimilarityTeacherNet(BaseNet):
    def __init__(self, scale, n_colors, modules_to_freeze=None, initialize_from=None, modules_to_initialize=None):
        super(AttendSimilarityTeacherNet, self).__init__()

        self.scale = scale
        d = 56
        s = 12
        m_1 = 4
        m_2 = 3
        
        self.backbone = ModifiedFSRCNN(scale, n_colors, d, s, m_1, m_2)
        self.last_layer = nn.Conv2d(d, 1, kernel_size=3, stride=1, padding=1)
        self.weight_init()
        if initialize_from is not None:
            self.load_pretrained_model(initialize_from, modules_to_initialize)
        if modules_to_freeze is not None:
            self.freeze_modules(modules_to_freeze)
        
        
    def forward(self, LR, HR):
        ret_dict = dict()
        
        x = HR
        layer_names = self.backbone.network._modules.keys()
        for layer_name in layer_names:
            x = self.backbone.network._modules[layer_name](x)
            ret_dict[layer_name] = x
        
        residual_hr = self.last_layer(x)
        LR = nn.functional.interpolate(LR, scale_factor=self.scale,
                                        mode='bicubic')
        hr = LR + residual_hr
        ret_dict['hr'] = hr
        ret_dict['residual_hr'] = residual_hr
        return ret_dict
    
    
class AttendSimilarityStudentNet(BaseNet):
    def __init__(self, scale, n_colors, layers_to_attend=None, modules_to_freeze=None, 
                 initialize_from=None, modules_to_initialize=None):
        super(AttendSimilarityStudentNet, self).__init__()

        self.layers_to_attend = layers_to_attend
        self.scale = scale
        upscale_factor = scale
        d = 56
        s = 12
        m_1 = 4
        m_2 = 3
        self.cos_sim = torch.nn.CosineSimilarity(dim=1)
        self.sigmoid = torch.nn.Sigmoid()
        self.backbone = ModifiedFSRCNN(scale, n_colors, d, s, m_1, m_2)
        self.last_layer = nn.Conv2d(d, 1, kernel_size=3, stride=1, padding=1)
        self.weight_init()
        if initialize_from is not None:
            self.load_pretrained_model(initialize_from, modules_to_initialize)
        if modules_to_freeze is not None:
            self.freeze_modules(modules_to_freeze)
        
        
    def forward(self, LR, teacher_pred_dict=None):
        ret_dict = dict()
        upscaled_lr = nn.functional.interpolate(LR, scale_factor=self.scale, mode='bicubic')
        x = upscaled_lr
        
        layer_names = self.backbone.network._modules.keys()
        for layer_name in layer_names:
            x = self.backbone.network._modules[layer_name](x)
            if teacher_pred_dict is not None and layer_name in layers_to_attend:
                teacher_layer = teacher_pred_dict[layer_name]
                x, attention = self.apply_attention(x, teacher_layer)
                ret_dict[layer_name + '_attention'] = attention
            ret_dict[layer_name] = x
        
        residual_hr = self.last_layer(x)
        
        hr = upscaled_lr + residual_hr
        ret_dict['hr'] = hr
        ret_dict['residual_hr'] = residual_hr
            
        return ret_dict
    
    
    def apply_attention(self, x, y):
        attention = self.sigmoid(self.cos_sim(x, y).unsqueeze(1))
        attended_x = x * attention
        return attended_x, attention
    
        
# For Resolution Disentangling Experiments
def get_disentangle_student(scale, n_colors, **kwargs):
    return DisentangleStudentNet(scale, n_colors, **kwargs)


def get_disentangle_teacher(scale, n_colors, **kwargs):
    return DisentangleTeacherNet(scale, n_colors, **kwargs)


def get_attend_similarity_teacher(scale, n_colors, **kwargs):
    return AttendSimilarityTeacherNet(scale, n_colors, **kwargs)


def get_attend_similarity_student(scale, n_colors, **kwargs):
    return AttendSimilarityStudentNet(scale, n_colors, **kwargs)


def get_model(config, model_type):
    print('model name:', config[model_type+'_model'].name)

    f = globals().get('get_' + config[model_type+'_model'].name)
    if config[model_type+'_model'].params is None:
        return f()
    else:
        return f(**config[model_type+'_model'].params)

    
    
