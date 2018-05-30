import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def point_form(box):
    # Convert prior_boxes (cx, cy, w, h) to (xmin, ymin, xmax, ymax)
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
