import torch.nn as nn
import torch.nn.functional as F


class StudentNet(nn.Module):
    def __init__(self, scale, n_colors):
        super(StudentNet, self).__init__()

        d = 56
        s = 12
        m_1 = 4
        m_2 = 3
        
        upscale_factor = scale
        
        self.layers = []
        self.layers.append(nn.Sequential(
            nn.Conv2d(in_channels=n_colors, 
                      out_channels=d, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(in_channels=d, out_channels=d,
                      kernel_size=3, stride=1, padding=1),
            nn.PReLU()))
        
        self.first_layer = nn.Sequential(*self.layers)
            
        self.layers = []
        self.layers.append(nn.Sequential(
            nn.Conv2d(in_channels=d, out_channels=s, 
                      kernel_size=1, stride=1, padding=0),
            nn.PReLU()))
        
        for _ in range(m_1):
            self.layers.append(nn.Sequential(
                nn.Conv2d(in_channels=s, out_channels=s, 
                          kernel_size=3, stride=1, padding=1),
                nn.PReLU()))
    
        self.layers.append(nn.Sequential(
            nn.Conv2d(in_channels=s, out_channels=d, 
                      kernel_size=1, stride=1, padding=0),
            nn.PReLU()))
        
        self.mid_part = nn.Sequential(*self.layers)
        
        self.layers = []
        self.layers.append(nn.Sequential(
            nn.Conv2d(in_channels=d, out_channels=s,
                      kernel_size=1, stride=1, padding=0),
            nn.PReLU()))
        
        
        ## last layer has the 4-depth layers
        for _ in range(m_2):
            self.layers.append(nn.Sequential(
                nn.Conv2d(in_channels=s, out_channels=s, 
                          kernel_size=3, stride=1, padding=1),
                nn.PReLU()))

        self.layers.append(nn.Sequential(
            nn.Conv2d(in_channels=s, out_channels=d,
                       kernel_size=1, stride=1, 
                       padding=0),
            nn.PReLU()))
            
        self.layers.append(
            nn.Conv2d(d, upscale_factor ** 2, kernel_size=3, stride=1, padding=1))
        
        self.last_part = nn.Sequential(*self.layers)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        
    
    def forward(self, lr, atten):
        mid_part1 = self.first_layer(lr)
        mid_part2 = self.mid_part(mid_part1)
        out = self.last_part(mid_part2 * atten)
        out = self.pixel_shuffle(out)
        
        return out, mid_part1, mid_part2
    
    
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

                    
class TeacherNet(nn.Module):
    def __init__(self, scale, n_colors):
        super(TeacherNet, self).__init__()

        d = 56
        s = 12
        m_1 = 4
        
        upscale_factor = scale
        
        self.first_layer = nn.Sequential(
            nn.Conv2d(in_channels=n_colors, 
                      out_channels=d, kernel_size=3, stride=1, padding=1),
            nn.PReLU())
        
        self.layers = []
        self.layers.append(nn.Sequential(
            nn.Conv2d(in_channels=d, out_channels=s, 
                      kernel_size=1, stride=1, padding=0),
            nn.PReLU()))
        
        for _ in range(m_1):
            self.layers.append(nn.Sequential(
                nn.Conv2d(in_channels=s, out_channels=s, 
                          kernel_size=3, stride=1, padding=1),
                nn.PReLU()))
    
        self.layers.append(nn.Sequential(
            nn.Conv2d(in_channels=s, out_channels=d, 
                      kernel_size=1, stride=1, padding=0),
            nn.PReLU()))
        
        self.mid_part = nn.Sequential(*self.layers)
        
    def forward(self, x):
        
        mid_part1 = self.first_layer(x)
        mid_part2 = self.mid_part(mid_part1)
        _, _, cha, _ = mid_part2.size()
        atten1 = F.avg_pool2d(mid_part2, kernel_size=2)  # 56 channels
        atten2 = F.avg_pool2d(atten1, kernel_size=2)
        atten3 = F.avg_pool2d(atten2, kernel_size=2)
        
        return F.sigmoid(atten1), F.sigmoid(atten2), F.sigmoid(atten3), F.sigmoid(mid_part2)
    
    
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
                    
                    
class HalluNet(nn.Module):
    # [TODO] refactoring
    def __init__(self, scale, n_colors):
        super(HalluNet, self).__init__()

        d = 56
        s = 12
        m_1 = 4
        
        upscale_factor = scale
        
        self.first_layer = nn.Sequential(
            nn.Conv2d(in_channels=n_colors, out_channels=d, 
                      kernel_size=3, stride=1, padding=1),
            nn.PReLU())
        
        self.layers = []
        self.layers.append(nn.Sequential(
            nn.Conv2d(in_channels=d, out_channels=s, 
                      kernel_size=1, stride=1, padding=0),
            nn.PReLU()))
        
        for _ in range(m_1):
            self.layers.append(nn.Sequential(
                nn.Conv2d(in_channels=s, out_channels=s,
                          kernel_size=3, stride=1, padding=1),
                nn.PReLU()))
    
        self.layers.append(nn.Sequential(
            nn.Conv2d(in_channels=s, out_channels=d, 
                      kernel_size=1, stride=1, padding=0),
            nn.PReLU()))
        
        self.mid_part = nn.Sequential(*self.layers)
        
        
    def forward(self, lr):
        
        mid_part1 = self.first_layer(lr)
        mid_part2 = self.mid_part(mid_part1)
        _, _, cha, _ = mid_part2.size()
        atten1 = F.avg_pool2d(mid_part2, kernel_size=2)  # 56 channels
        atten2 = F.avg_pool2d(atten1, kernel_size=2)
        atten3 = F.avg_pool2d(atten2, kernel_size=2)
        
        return F.sigmoid(atten1), F.sigmoid(atten2), F.sigmoid(atten3), F.sigmoid(mid_part1)
    
    
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


def get_student(scale, n_colors, **kwargs):
    return StudentNet(scale, n_colors, **kwargs)


def get_teacher(scale, n_colors, **kwargs):
    return TeacherNet(scale, n_colors, **kwargs)


def get_hallucination(scale, n_colors, **kwargs):
    return HalluNet(scale, n_colors, **kwargs)


def get_model(config, model_type):
    print('model name:', config[model_type+'_model'].name)

    f = globals().get('get_' + config[model_type+'_model'].name)
    if config[model_type+'_model'].params is None:
        return f()
    else:
        return f(**config[model_type+'_model'].params)

    
