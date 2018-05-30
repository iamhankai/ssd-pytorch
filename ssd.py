import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
import math
from data import voc, coco
from anchor_box import AnchorBox
from inference_layer import InferenceLayer

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}

mbox = [4, 6, 6, 6, 4, 4]  # number of boxes per feature map location

class VGG(nn.Module):
    def __init__(self, base, phase='train', num_classes=1000):
        super(VGG, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.cfg = (coco, voc)[num_classes==21]
        self.size = self.cfg['min_dim']
        self.priorbox = AnchorBox(self.cfg)
        self.priors = Variable(self.priorbox.forward(), volatile=True)
        self.l2_norm_scale = L2NormScale(512, 20)
        # 38*38
        self.features = nn.ModuleList(base)
        self.loc0 = nn.Conv2d(512, 4*4, kernel_size=3, padding=1)
        self.class0 = nn.Conv2d(512, 4*num_classes, kernel_size=3, padding=1)
        # 19*19
        self.extra1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),
            nn.ReLU(inplace=True), 
            nn.Conv2d(1024, 1024, kernel_size=1),
            nn.ReLU(inplace=True), 
        )
        self.loc1 = nn.Conv2d(1024, 6*4, kernel_size=3, padding=1)
        self.class1 = nn.Conv2d(1024, 6*num_classes, kernel_size=3, padding=1)
        # 10*10
        self.extra2 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1),
            nn.ReLU(inplace=True), 
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True), 
        )
        self.loc2 = nn.Conv2d(512, 6*4, kernel_size=3, padding=1)
        self.class2 = nn.Conv2d(512, 6*num_classes, kernel_size=3, padding=1)
        # 5*5
        self.extra3 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1),
            nn.ReLU(inplace=True), 
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True), 
        )
        self.loc3 = nn.Conv2d(256, 6*4, kernel_size=3, padding=1)
        self.class3 = nn.Conv2d(256, 6*num_classes, kernel_size=3, padding=1)
        # 3*3
        self.extra4 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1),
            nn.ReLU(inplace=True), 
            nn.Conv2d(128, 256, kernel_size=3),
            nn.ReLU(inplace=True), 
        )
        self.loc4 = nn.Conv2d(256, 4*4, kernel_size=3, padding=1)
        self.class4 = nn.Conv2d(256, 4*num_classes, kernel_size=3, padding=1)
        # 1*1
        self.extra5 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1),
            nn.ReLU(inplace=True), 
            nn.Conv2d(128, 256, kernel_size=3),
            nn.ReLU(inplace=True), 
        )
        self.loc5 = nn.Conv2d(256, 4*4, kernel_size=3, padding=1)
        self.class5 = nn.Conv2d(256, 4*num_classes, kernel_size=3, padding=1)
        #self._initialize_weights()
        if phase == 'test':
            self.softmax = nn.Softmax()  #dim=-1
            self.inference_layer = InferenceLayer(top_k=200, conf_thresh=0.01, nms_thresh=0.45)

    def forward(self, x):
        loc_list = []
        class_list = []
        # apply vgg up to conv4_3 relu
        for k in range(23):
            x = self.features[k](x)
        s = self.l2_norm_scale(x)
        loc_list.append(self.loc0(s))
        class_list.append(self.class0(s))
        for k in range(23, len(self.features)):
            x = self.features[k](x)
        #x = self.extra1(x)
        loc_list.append(self.loc1(x))
        class_list.append(self.class1(x))
        x = self.extra2(x)
        loc_list.append(self.loc2(x))
        class_list.append(self.class2(x))
        x = self.extra3(x)
        loc_list.append(self.loc3(x))
        class_list.append(self.class3(x))
        x = self.extra4(x)
        loc_list.append(self.loc4(x))
        class_list.append(self.class4(x))
        x = self.extra5(x)
        loc_list.append(self.loc5(x))
        class_list.append(self.class5(x))
        
        # apply multibox head to source layers
        loc = []
        conf = []
        for (l, c) in zip(loc_list, class_list):
            loc.append(l.permute(0, 2, 3, 1).contiguous())
            conf.append(c.permute(0, 2, 3, 1).contiguous())
            
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if self.phase=='test':
            conf = conf.view(conf.size(0), -1,
                             self.num_classes)
            for idx in range(conf.size(0)):
                conf[idx] = self.softmax(conf[idx])
            output = self.inference_layer(
                loc.view(loc.size(0), -1, 4),      # loc preds
                conf,
                self.priors.type(type(x.data))     # default boxes
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )  
        return output

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
                
    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                 map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')

# https://github.com/weiliu89/caffe/issues/241
class L2NormScale(nn.Module):
    def __init__(self, n_channels, init_scale):
        super(L2NormScale, self).__init__()
        self.n_channels = n_channels
        self.init_scale = init_scale
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        init.constant(self.weight, self.init_scale)
        
    def forward(self, x):
        x = F.normalize(x, dim=1)
        x = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return x

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels=3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers


cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M', 512, 512, 512]


def vgg16(pretrained=False, **kwargs):
    """VGG 16-layer model

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg), **kwargs)
    if pretrained:
        pretrained_dict = model_zoo.load_url(model_urls['vgg16'])
        model_dict = model.state_dict()
        print(pretrained_dict.keys())
        common_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        print(common_dict.keys())
        model_dict.update(common_dict) 
        model.load_state_dict(model_dict)
    return model

