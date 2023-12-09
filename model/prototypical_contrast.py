from abc import ABC

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class PrototypeContrastLoss(nn.Module, ABC):
    def __init__(self):
        super(PrototypeContrastLoss, self).__init__()
        self.temperature = 1

    def _contrastive(self, base_pro, pos_pro, cross_p, neg_pro_1):

        loss = torch.zeros(1).cuda()
        bs = base_pro.shape[0]
        pos_pro_1 = torch.mean(pos_pro, dim=0)
        pos_pro_2 = torch.mean(cross_p, dim=0)

        for base, pos_1, pos_2, neg in zip(base_pro, pos_pro_1, pos_pro_2, neg_pro_1):
            neg_logits = 0

            positive_dot_contrast_1 = torch.div(F.cosine_similarity(base, pos_1,0),
                                        self.temperature)

            positive_dot_contrast_2 = torch.div(F.cosine_similarity(base, pos_2,0),
                                        self.temperature)

            negative_dot_contrast = torch.div(F.cosine_similarity(base, neg, 0),
                                        self.temperature)

            pos_logits  = torch.exp(positive_dot_contrast_1) + torch.exp(positive_dot_contrast_2)

            neg_logits = torch.exp(negative_dot_contrast) + pos_logits
            mean_log_prob_pos = - torch.log((pos_logits/(neg_logits))+1e-8)

            loss = loss + mean_log_prob_pos

        return loss/bs

    def forward(self, base_pro, pos_pro, cross_p, neg_pro_1):
        base_pro = base_pro.clone()
        pos_pro = pos_pro.clone()
        cross_p = cross_p.clone()

        loss = self._contrastive(base_pro, pos_pro, cross_p, neg_pro_1)
        return loss
