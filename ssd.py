import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd import Variable
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
import math
from math import sqrt as sqrt
from data import voc, coco
from data import voc as voc_cfg

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

# generate default anchor boxes
class AnchorBox(object):
    def __init__(self, cfg):
        super(AnchorBox, self).__init__()
        self.feature_maps = cfg['feature_maps']
        self.aspect_ratios = cfg['aspect_ratios']
        self.clip = cfg['clip']
        self.m = len(self.feature_maps)
        self.s_min = 0.1
        self.s_max = 0.9
        
    def forward(self):
        anchor_box_list = []
        for k,f_k in enumerate(self.feature_maps):
            s_k = self.s_min + (self.s_max-self.s_min) * k/(self.m-1)
            s_k_next = self.s_min + (self.s_max-self.s_min) * (k+1)/(self.m-1)
            s_k_2 = sqrt(s_k * s_k_next)
            for i in range(f_k):
                for j in range(f_k):
                    # center
                    cx = (j+0.5)/f_k
                    cy = (i+0.5)/f_k
                    # aspect_ratio: 1
                    anchor_box_list += [cx, cy, s_k, s_k]
                    anchor_box_list += [cx, cy, s_k_2, s_k_2]
                    # other aspect_ratio
                    for r in self.aspect_ratios[k]:
                        anchor_box_list += [cx, cy, s_k*sqrt(r), s_k/sqrt(r)]
                        anchor_box_list += [cx, cy, s_k/sqrt(r), s_k*sqrt(r)]
        output = torch.Tensor(anchor_box_list).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output


# Loss function, used in training stage
# Convert prior_boxes (cx, cy, w, h) to (xmin, ymin, xmax, ymax)
def point_form(box):
    return torch.cat([box[:, :2] - box[:, 2:]/2, box[:, :2] + box[:, 2:]/2], 1)


def iou_jaccard(box1, box2):
    # box: xmin, ymin, xmax, ymax
    # box1: num_obj*4, box2: num_priori
    # return num_obj*num_priori
    len1 = box1.size(0)
    len2 = box2.size(0)
    left = torch.max(box1[:, 0].unsqueeze(1).expand(len1, len2),
                     box2[:, 0].unsqueeze(0).expand(len1, len2))
    top = torch.max(box1[:, 1].unsqueeze(1).expand(len1, len2),
                    box2[:, 1].unsqueeze(0).expand(len1, len2))
    right = torch.min(box1[:, 2].unsqueeze(1).expand(len1, len2),
                      box2[:, 2].unsqueeze(0).expand(len1, len2))
    bottom = torch.min(box1[:, 3].unsqueeze(1).expand(len1, len2),
                       box2[:, 3].unsqueeze(0).expand(len1, len2))
    intersect = torch.clamp(right-left, min=0)*torch.clamp(bottom-top, min=0)
    area1 = (box1[:, 2]-box1[:, 0])*(box1[:, 3]-box1[:, 1])
    area1 = area1.unsqueeze(1).expand_as(intersect)
    area2 = (box2[:, 2]-box2[:, 0])*(box2[:, 3]-box2[:, 1])
    area2 = area2.unsqueeze(0).expand_as(intersect)
    union = area1+area2-intersect
    return intersect/union


def encode(loc_t, priors, variance):
    # error of cx, cy, w, h
    gt_cxcy = (loc_t[:, :2]+loc_t[:, 2:])/2 - priors[:, :2]
    gt_cxcy /= (priors[:, 2:]*variance[0])
    gt_wh = (loc_t[:, 2:]-loc_t[:, :2]) / priors[:, 2:]
    gt_wh = torch.log(gt_wh) / variance[1]
    return torch.cat([gt_cxcy, gt_wh], 1)


def assign(threshold, variance, gt_box, gt_class, priors, loc_t, class_t, idx):
    # assign the gt_box with highest iou to each prior box
    # gt_box: xmin, ymin, xmax, ymax
    # priors: (cx, cy, w, h)
    iou = iou_jaccard(gt_box, point_form(priors))
    best_iou, best_gt_idx = iou.max(0, keepdim=False)
    _, best_prior_idx = iou.max(1, keepdim=False)
    best_iou.index_fill_(0, best_prior_idx, 2.0)  # ensure each obj with >=1 prior
    loc_t[idx] = encode(gt_box[best_gt_idx], priors, variance)
    class_t_idx = gt_class[best_gt_idx]+1
    class_t_idx[best_iou < threshold] = 0  # label as background
    class_t[idx] = class_t_idx


class MultiBoxLoss(nn.Module):
    def __init__(self, cfg, use_gpu, negpos_ratio):
        super(MultiBoxLoss, self).__init__()
        self.variance = cfg['variance']
        self.num_classes = cfg['num_classes']
        self.use_gpu = use_gpu
        self.negpos_ratio = negpos_ratio

    def forward(self, prediction, gt):
        # loc_p: batch_size*num_prior*4
        # class_p: batch_size*num_priori*num_class
        loc_p, class_p, priors = prediction
        batch_size = loc_p.size(0)
        priors = priors[:loc_p.size(1), :]  # due to multi-gpu
        num_priors = priors.size(0)
        # assign target for each prior anchor
        loc_t = torch.FloatTensor(batch_size, num_priors, 4)
        class_t = torch.LongTensor(batch_size, num_priors)
        for idx in range(batch_size):
            gt_box = gt[idx][:, :-1].data  # num_obj*4
            gt_class = gt[idx][:, -1].data  # num_obj
            assign(0.5, self.variance, gt_box, gt_class, priors.data, loc_t, class_t, idx)
        # wrap targets
        if self.use_gpu:
            loc_t = loc_t.cuda()
            class_t = class_t.cuda()
        loc_t = Variable(loc_t, requires_grad=False)
        class_t = Variable(class_t, requires_grad=False)
        # Localization Loss (Smooth L1)
        pos_mask = (class_t > 0)
        num_pos_each_sample = pos_mask.long().sum(1, keepdim=True)
        pos_mask_loc = pos_mask.unsqueeze(2).expand_as(loc_t)
        loc_p = loc_p[pos_mask_loc].view(-1, 4)
        loc_t = loc_t[pos_mask_loc].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)
        # hard negative mining
        loss_gt = -torch.log(F.softmax(class_p.view(-1, self.num_classes)))  # -lg(softmax(x))
        loss_gt = loss_gt.gather(dim=1, index=class_t.view(-1, 1))
        loss_gt[pos_mask] = 0  # filter out pos
        loss_gt = loss_gt.view(batch_size, -1)
        _, loss_sort_idx = loss_gt.sort(1, descending=True)
        _, idx_rank = loss_sort_idx.sort(1)
        num_neg_each_sample = torch.clamp(self.negpos_ratio*num_pos_each_sample,
                                          max=pos_mask.size(1)-1)
        neg_mask = idx_rank < num_neg_each_sample.expand_as(idx_rank)
        # class loss
        pos_mask_class = pos_mask.unsqueeze(2).expand_as(class_p)
        neg_mask_class = neg_mask.unsqueeze(2).expand_as(class_p)
        class_p = class_p[(pos_mask_class + neg_mask_class) > 0].view(-1, self.num_classes)
        class_t = class_t[(pos_mask + neg_mask) > 0].view(-1)
        loss_c = F.cross_entropy(class_p, class_t, size_average=False)
        # loss sum
        N = num_pos_each_sample.data.sum()
        loss_l /= N
        loss_c /= N
        return loss_l, loss_c
    
# Used in inference stage
class InferenceLayer(Function):
    # used in test stage, the final layer
    def __init__(self, top_k, conf_thresh, nms_thresh):
        self.top_k = top_k
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.num_classes = voc_cfg['num_classes']
        self.variance = voc_cfg['variance']
        
    def forward(self, loc_p, class_p, priors):
        # loc_p: batch_size*num_priors*4
        # class_p: batch_size*num_priors*num_class
        # priors: num_priors*4
        batch_size = loc_p.size(0)
        num_priors = priors.size(0)
        output = torch.zeros(batch_size, self.num_classes, self.top_k, 5)
        class_p = class_p.transpose(1,2)
        
        for idx in range(batch_size):
            decoded_boxes = decode(loc_p[idx], priors, self.variance)
            # perform nms for each class
            for c in range(1, self.num_classes):
                # select box with confidence > thresh
                c_mask = (class_p[idx][c]>self.conf_thresh)
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                scores = class_p[idx][c][c_mask]
                boxes = decoded_boxes[l_mask].view(-1, 4)
                if scores.dim()==0:
                    continue
                ids, count = nms(boxes, scores, self.nms_thresh, self.top_k)
                output[idx, c, :count] = torch.cat((scores[ids[:count]].unsqueeze(1),
                                                   boxes[ids[:count]]), 1)
        return output

def decode(loc_p, priors, variance):
    cxcy_p = loc_p[:,:2]*(priors[:,2:]*variance[0]) + priors[:,:2]
    wh_p = torch.exp(loc_p[:,2:]*variance[1]) * priors[:,2:]
    return torch.cat([cxcy_p-wh_p/2,cxcy_p+wh_p/2], 1)
    
# Original author: Francisco Massa:
# https://github.com/fmassa/object-detection.torch
# Ported to PyTorch by Max deGroot (https://github.com/amdegroot/ssd.pytorch)
def nms(boxes, scores, overlap=0.5, top_k=200):
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location preds for the img, Shape: [num_priors,4].
        scores: (tensor) The class predscores for the img, Shape:[num_priors].
        overlap: (float) The overlap thresh for suppressing unnecessary boxes.
        top_k: (int) The Maximum number of box preds to consider.
    Return:
        The indices of the kept boxes with respect to num_priors.
    """

    keep = scores.new(scores.size(0)).zero_().long()
    if boxes.numel() == 0:
        return keep
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)  # sort in ascending order
    # I = I[v >= 0.01]
    idx = idx[-top_k:]  # indices of the top-k largest vals
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()

    # keep = torch.Tensor()
    count = 0
    while idx.numel() > 0:
        i = idx[-1]  # index of current largest val
        # keep.append(i)
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]  # remove kept element from view
        # load bboxes of next highest vals
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)
        # store element-wise max with next highest score
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        # check sizes of xx1 and xx2.. after each iteration
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w*h
        # IoU = i / (area(a) + area(b) - i)
        rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
        union = (rem_areas - inter) + area[i]
        IoU = inter/union  # store result in iou
        # keep only elements with an IoU <= overlap
        idx = idx[IoU.le(overlap)]
    return keep, count
    
    