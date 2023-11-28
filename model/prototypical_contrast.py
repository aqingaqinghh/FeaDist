from abc import ABC

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
def Weighted_GAP(supp_feat, mask):
    supp_feat = supp_feat * mask
    feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
    area = F.avg_pool2d(mask.float(), (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005
    supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area  
    return supp_feat
  

class PrototypeContrastLoss(nn.Module, ABC):
    def __init__(self):
        super(PrototypeContrastLoss, self).__init__()

        self.temperature = 1
        self.m = 1000
        self.n = 2000


    def _contrastive(self, base_pro, pos_pro, cross_prototype, cross_p, neg_pro_1, neg_pro_2):

        loss = torch.zeros(1).cuda()
        bs = base_pro.shape[0]
        pos_pro_1 = torch.mean(pos_pro, dim=0)
        pos_pro_2 = torch.mean(cross_prototype, dim=0)
        pos_pro_3 = torch.mean(cross_p, dim=0)
        #pos_pro = pos_pro.permute(1,2,0)

        for base, pos_1, pos_2, pos_3, neg_1, neg_2 in zip(base_pro, pos_pro_1, pos_pro_2, pos_pro_3, neg_pro_1, neg_pro_2):
            neg_logits = 0

            #positive_dot_contrast_1 = torch.div(F.cosine_similarity(base.unsqueeze(-1), pos,0),
                                        #self.temperature)
            #print("positive_dot_contrast_1: ", positive_dot_contrast_1)
            positive_dot_contrast_1 = torch.div(F.cosine_similarity(base, pos_1,0),
                                        self.temperature)
            #positive_dot_contrast_2 = torch.div(F.cosine_similarity(base, pos_2,0),
                                        #self.temperature)
            positive_dot_contrast_3 = torch.div(F.cosine_similarity(base, pos_3,0),
                                        self.temperature)

            #negative_samples = neg_dict[clss].transpose(1, 0)
            #negative_samples = neg_dict[clss][]

            negative_dot_contrast_1 = torch.div(F.cosine_similarity(base, neg_1, 0),
                                        self.temperature)

            negative_dot_contrast_2 = torch.div(F.cosine_similarity(base, neg_2, 0),
                                        self.temperature)

            #pos_logits  = torch.exp(positive_dot_contrast_1).sum()*0.3 + torch.exp(positive_dot_contrast_2)*0.7
            pos_logits  = torch.exp(positive_dot_contrast_1) + torch.exp(positive_dot_contrast_3)

            neg_logits = torch.exp(negative_dot_contrast_1) + torch.exp(negative_dot_contrast_2) + pos_logits
            mean_log_prob_pos = - torch.log((pos_logits/(neg_logits))+1e-8)

            loss = loss + mean_log_prob_pos
            #print("loss: ", loss, loss/bs)

        return loss/bs

    def forward(self, base_pro, pos_pro, cross_prototype, cross_p, neg_pro_1, neg_pro_2):     # feats:4,256,260,260 ; labels:4,520,520; predict:4,260,260
        base_pro = base_pro.clone()
        pos_pro = pos_pro.clone()
        cross_prototype = cross_prototype.clone()
        cross_p = cross_p.clone()

        loss = self._contrastive(base_pro, pos_pro, cross_prototype, cross_p, neg_pro_1, neg_pro_2)
        return loss