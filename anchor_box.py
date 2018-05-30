import torch
from math import sqrt as sqrt


mbox = [4, 6, 6, 6, 4, 4]  # number of boxes per feature map location
num_classes = 21

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


