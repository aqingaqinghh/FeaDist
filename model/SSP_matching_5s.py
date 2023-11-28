#import model.resnet_sei as resnet
#import model.resnet_bam as resnet
import model.resnet as resnet
#import model.resnet_SimCLR as resnet2

import torch
from torch import nn
import torch.nn.functional as F
import pdb
from model.seed_init import place_seed_points
from model.prototypical_contrast import PrototypeContrastLoss
import numpy as np

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)  # 7,3     3,1
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

def Weighted_GAP(supp_feat, mask):
    supp_feat = supp_feat * mask
    feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
    area = F.avg_pool2d(mask, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005
    supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area
    return supp_feat

def get_gram_matrix(fea):
    b, c, h, w = fea.shape        
    fea = fea.reshape(b, c, h*w)    # C*N
    fea_T = fea.permute(0, 2, 1)    # N*C
    fea_norm = fea.norm(2, 2, True)
    fea_T_norm = fea_T.norm(2, 1, True)
    gram = torch.bmm(fea, fea_T)/(torch.bmm(fea_norm, fea_T_norm) + 1e-7)    # C*C
    return gram

class Attention(nn.Module):
    """
    Guided Attention Module (GAM).

    Args:
        in_channels: interval channel depth for both input and output
            feature map.
        drop_rate: dropout rate.
    """

    def __init__(self, in_channels, drop_rate=0.1):
        super().__init__()
        self.DEPTH = in_channels
        self.DROP_RATE = drop_rate
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels=self.DEPTH, out_channels=self.DEPTH,
                      kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.DEPTH, out_channels=self.DEPTH,
                      kernel_size=1),
            nn.Dropout(p=drop_rate),
            nn.Sigmoid())

    @staticmethod
    def mask(embedding, mask):
        h, w = embedding.size()[-2:]

        mask = F.interpolate(mask.unsqueeze(1), size=(h, w), mode='nearest')
        mask=mask
        return mask * embedding

    def forward(self, *x):
        if len(x) == 2:
            Fs, Ys = x
            att = F.adaptive_avg_pool2d(self.mask(Fs, Ys), output_size=(1, 1))
        else:
            Fs = x[0]
            att = F.adaptive_avg_pool2d(Fs, output_size=(1, 1))
        g = self.gate(att)
        Fs = g * Fs
        return Fs

class SSP_MatchingNet(nn.Module):
    def __init__(self, backbone_name, shot):
        super(SSP_MatchingNet, self).__init__()

        self.backbone = resnet.__dict__[backbone_name](pretrained=True)

        self.layer0 = nn.Sequential(self.backbone.conv1, self.backbone.bn1, self.backbone.relu, self.backbone.maxpool)
        self.layer1, self.layer2, self.layer3 = self.backbone.layer1, self.backbone.layer2, self.backbone.layer3

        self.contrast_loss = PrototypeContrastLoss()
        self.shot = 5

        k_size = 9
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

        self.gam = Attention(in_channels=1024)

    def forward(self, img_s_list, mask_s_list, img_q, mask_q):
        #img_s_list = torch.cat(img_s_list, dim=0).squeeze(1).unsqueeze(0)
        #mask_s_list = torch.cat(mask_s_list, dim=0).squeeze(1).unsqueeze(0)
        #print(img_s_list.shape)
        img_s_list = img_s_list.permute(1, 0, 2, 3, 4)
        mask_s_list = mask_s_list.permute(1, 0, 2, 3)


        # feature maps of support images and query image
        feature_s_list = []
        supp_feat_list = []
        for k in range(len(img_s_list)):
            with torch.no_grad():
                x = self.layer2(self.layer1(self.layer0(img_s_list[k])))
                #supp_feat_list.append(x)
            s = self.layer3(x)
            s = self.ECA(s)
            feature_s_list.append(s)

        with torch.no_grad():
            y = self.layer2(self.layer1(self.layer0(img_q)))
        feature_q = self.layer3(y)
        feature_q = self.ECA(feature_q)
        bs = feature_q.shape[0]
  
        if self.shot == 5:
            que_gram = get_gram_matrix(feature_q)
            norm_max = torch.ones_like(que_gram).norm(dim=(1, 2))
            est_val_list = []
            for supp_item in feature_s_list:
                #supp_item = feature_s_list[k]
                supp_gram = get_gram_matrix(supp_item)
                gram_diff = que_gram - supp_gram
                est_val_list.append((gram_diff.norm(dim=(1, 2)) / norm_max).reshape(bs, 1, 1, 1))  # norm2
            est_val_total = torch.cat(est_val_list, 1).squeeze(-1).squeeze(-1)  # [bs, shot, 1, 1]
            est_mean = torch.mean(est_val_total, dim=1)
            est_list_final = []
            for bs_ in range(bs):
                est_list = []
                for k in range(len(img_s_list)):
                    est = (1 / (est_val_total[bs_, k] / est_mean[bs_] + 1e-5)) / 5
                    est_list.append(torch.tensor([est]).unsqueeze(0))
                est_list_final.append(torch.cat(est_list, dim=1))
            est_final = torch.cat(est_list_final, dim=0).cuda()
            val1, idx1 = est_val_total.sort(dim=1)
            v, i = est_val_total.sort(dim=1, descending=True)
            val2, idx2 = idx1.sort(1)
            b = torch.gather(v, 1, idx2)
            weight_su = est_final.permute(1, 0).unsqueeze(-1)

        h, w = img_q.shape[-2:]
        bs = feature_q.shape[0]

        # Gets multiple supported prototypes (SGC)
        SFP = self.SGC(img_s_list, mask_s_list, feature_s_list, bs)

        # foreground(target class) and background prototypes pooled from K support features
        feature_fg_list = []
        feature_bg_list = []
        cross_fg_list = []
        cross_prototype_list = []
        supp_out_ls = []
        for k in range(len(img_s_list)):
            feature_fg = self.masked_average_pooling(feature_s_list[k],
                                                               (mask_s_list[k] == 1).float())[None, :]
            cross_fg, num_fg, loss_q = self.cross_prototype(feature_q, feature_s_list[k], mask_s_list[k])
            feature_bg = self.masked_average_pooling(feature_s_list[k],
                                                               (mask_s_list[k] == 0).float())[None, :]
            cross_prototype_list.append(cross_fg.unsqueeze(0))
            feature_fg_list.append(feature_fg)
            cross_fg_list.append(feature_fg * 0.5 + cross_fg.unsqueeze(0) * 0.5)
            feature_bg_list.append(feature_bg)

        # average K foreground prototypes and K background prototypes
        support_prototype = torch.cat(feature_fg_list, dim=0) #[5,16,1024]
        #new_support = support_prototype * weight_su
        cross_prototype = torch.cat(cross_fg_list, dim=0)
        cross_prototype = cross_prototype * weight_su
        cross_p = torch.cat(cross_prototype_list, dim=0)

        FP_0 = torch.sum(cross_prototype, dim=0).unsqueeze(-1).unsqueeze(-1)
        FP_cross = torch.mean(cross_p, dim=0).unsqueeze(-1).unsqueeze(-1)
        BP_0 = torch.mean(torch.cat(feature_bg_list, dim=0), dim=0).unsqueeze(-1).unsqueeze(-1)
               
        # measure the similarity of query features to fg/bg prototypes
        #out_0 = self.similarity_func(feature_q, FP_0, BP_0)
        out_0 = self.similarity_func_0(feature_q, SFP, FP_0, BP_0)
        pred_0 = torch.argmax(out_0, dim=1)
        feature_q0 = self.gam(feature_q, pred_0)

        ##################### Self-Support Prototype (SSP) #####################
        SSFP_1, SSBP_1, ASFP_1, ASBP_1 = self.SSP_func(feature_q0, out_0, feature_s_list[0], mask_s_list[0])

        FP_1 = FP_0.cuda() * 0.9 + SSFP_1.cuda() * 0.1
        BP_1 = SSBP_1 * 0.3 + ASBP_1 * 0.7

        out_1 = self.similarity_func(feature_q.cuda(), FP_1.cuda(), BP_1.cuda())
        pred_1 = torch.argmax(out_1, dim=1)
        feature_q1 = self.gam(feature_q, pred_1)

        ##################### SSP Refinement #####################
        SSFP_2, SSBP_2, ASFP_2, ASBP_2 = self.SSP_func(feature_q1, out_1, feature_s_list[0], mask_s_list[0])

        BP_2 = SSBP_2 * 0.3 + ASBP_2 * 0.7

        FP_2 = FP_cross * 0.2 + SSFP_1 * 0.3 + SSFP_2 * 0.5
        BP_2 = BP_1 * 0.4 + BP_2 * 0.6

        #FP_2 = FP_0.cuda() * 0.5 + SSFP_2.cuda() * 0.5
        #BP_2 = SSBP_2 * 0.3 + ASBP_2 * 0.7

        #FP_2 = FP_0.cuda() * 0.3 + FP_1.cuda() * 0.3 + FP_2.cuda() * 0.4
        #BP_2 = BP_1 * 0.4 + BP_2 * 0.6


        out_2 = self.similarity_func(feature_q.cuda(), FP_2.cuda(), BP_2.cuda())

        #pred_2 = torch.argmax(out_2, dim=1)

        #SFP = self.SGC_q(img_s_list, [pred_2], [feature_q], bs)
        #FP_sgc = torch.mean(SFP, dim=2).unsqueeze(-1)
        #FP_sgc = FP_2 * 0.5 + FP_sgc * 0.5
        #out_2 = self.similarity_func_0(feature_q.cuda(), SFP, FP_sgc, BP_2)


        
        out_1 = F.interpolate(out_1, size=(h, w), mode="bilinear", align_corners=True)

        out_2 = F.interpolate(out_2, size=(h, w), mode="bilinear", align_corners=True)
        #print(out_2.shape)
        out_ls = [out_2, out_1]

        if self.training:
            #clses = clses.clone()
            base_sample = self.masked_average_pooling(feature_q,
                                                               (mask_q == 1).float())
            #[16,1024]
            neg_sample_1 = self.masked_average_pooling(feature_q,
                                                               (mask_q == 0).float())
            neg_sample_2 = neg_sample_1
            
            prototype_contrast_loss = self.contrast_loss(base_sample, support_prototype, cross_prototype, cross_p, neg_sample_1, neg_sample_2)
            out_ls.append(prototype_contrast_loss)
        return out_ls

    def SSP_func(self, feature_q, out, feature_s, mask):
        bs = feature_q.shape[0]
        mask = F.interpolate(mask.unsqueeze(1).float(), size=feature_s.shape[-2:], mode='bilinear', align_corners=True)
        pred_1 = out.softmax(1)
        pred_1 = pred_1.view(bs, 2, -1)
        pred_fg = pred_1[:, 1]
        pred_bg = pred_1[:, 0]
        fg_ls = []
        bg_ls = []
        fg_local_ls = []
        bg_local_ls = []
        for epi in range(bs):
            fg_thres = 0.7 #0.9 #0.6
            bg_thres = 0.6 #0.6
            cur_feat_s = feature_s[epi].view(1024,-1)
            cur_feat_s_norm = cur_feat_s / torch.norm(cur_feat_s, 2, 0, True) # 1024, 3600
            cur_feat_s_mask = (mask[epi].view(-1) == 1).cuda()
            cur_feat_s_mask_bg = (mask[epi].view(-1) != 1).cuda()
            cur_feat = feature_q[epi].view(1024, -1).cuda()
            f_h, f_w = feature_q[epi].shape[-2:]
            if (pred_fg[epi] > fg_thres).sum() > 0:
                fg_feat = cur_feat[:, (pred_fg[epi]>fg_thres)] #.mean(-1) 1024,Nq
                cur_feat_q_norm = fg_feat / torch.norm(fg_feat, 2, 0, True) # 1024, Nq
                sim_fg = torch.matmul(cur_feat_s_norm.t().cuda(), cur_feat_q_norm.cuda()) * 2.0 # 3600, Nq
                index_s = sim_fg.max(0)[1]  #Nq
                index_judge = cur_feat_s_mask[index_s]  #
                fg_feat_cross = fg_feat[:, index_judge]
                if fg_feat_cross.shape[1] != 0:
                    fg_feat = fg_feat_cross
            else:
                fg_feat = cur_feat[:, torch.topk(pred_fg[epi], 12).indices] #.mean(-1)
            if (pred_bg[epi] > bg_thres).sum() > 0:
                bg_feat = cur_feat[:, (pred_bg[epi]>bg_thres)] #.mean(-1)
                cur_feat_q_norm_bg = bg_feat / torch.norm(bg_feat, 2, 0, True) # 1024, Nq
                sim_bg = torch.matmul(cur_feat_s_norm.t().cuda(), cur_feat_q_norm_bg.cuda()) * 2.0 # 3600, Nq
                index_s_bg = sim_bg.max(0)[1]  #Nq
                index_judge_bg = cur_feat_s_mask_bg[index_s_bg]  #
                bg_feat_cross = bg_feat[:, index_judge_bg]
                if bg_feat_cross.shape[1] != 0:
                    bg_feat = bg_feat_cross
            else:
                bg_feat = cur_feat[:, torch.topk(pred_bg[epi], 12).indices] #.mean(-1)
            # global proto
            fg_proto = fg_feat.mean(-1)
            bg_proto = bg_feat.mean(-1)
            fg_ls.append(fg_proto.unsqueeze(0))
            bg_ls.append(bg_proto.unsqueeze(0))

            # local proto
            fg_feat_norm = fg_feat / torch.norm(fg_feat, 2, 0, True) # 1024, N1
            bg_feat_norm = bg_feat / torch.norm(bg_feat, 2, 0, True) # 1024, N2
            cur_feat_norm = cur_feat / torch.norm(cur_feat, 2, 0, True) # 1024, N3

            cur_feat_norm_t = cur_feat_norm.t() # N3, 1024
            fg_sim = torch.matmul(cur_feat_norm_t, fg_feat_norm) * 2.0 # N3, N1
            bg_sim = torch.matmul(cur_feat_norm_t, bg_feat_norm) * 2.0 # N3, N2

            fg_sim = fg_sim.softmax(-1)
            bg_sim = bg_sim.softmax(-1)

            fg_proto_local = torch.matmul(fg_sim, fg_feat.t()) # N3, 1024
            bg_proto_local = torch.matmul(bg_sim, bg_feat.t()) # N3, 1024

            fg_proto_local = fg_proto_local.t().view(1024, f_h, f_w).unsqueeze(0) # 1024, N3
            bg_proto_local = bg_proto_local.t().view(1024, f_h, f_w).unsqueeze(0) # 1024, N3

            fg_local_ls.append(fg_proto_local)
            bg_local_ls.append(bg_proto_local)

        # global proto
        new_fg = torch.cat(fg_ls, 0).unsqueeze(-1).unsqueeze(-1)
        new_bg = torch.cat(bg_ls, 0).unsqueeze(-1).unsqueeze(-1)

        # local proto
        new_fg_local = torch.cat(fg_local_ls, 0).unsqueeze(-1).unsqueeze(-1)
        new_bg_local = torch.cat(bg_local_ls, 0)

        return new_fg, new_bg, new_fg_local, new_bg_local

    def similarity_func(self, feature_q, fg_proto, bg_proto):
        similarity_fg = F.cosine_similarity(feature_q, fg_proto, dim=1)
        similarity_bg = F.cosine_similarity(feature_q, bg_proto, dim=1)

        out = torch.cat((similarity_bg[:, None, ...], similarity_fg[:, None, ...]), dim=1) * 10.0
        return out

    def masked_average_pooling(self, feature, mask):
        mask = F.interpolate(mask.unsqueeze(1), size=feature.shape[-2:], mode='bilinear', align_corners=True)
        masked_feature = torch.sum(feature * mask, dim=(2, 3)) \
                         / (mask.sum(dim=(2, 3)) + 1e-5)
        return masked_feature

    def SGC(self, img_s_list, mask_s_list, feature_s_list, bs):
        s_seed = torch.zeros(bs, len(img_s_list), 5, 2)

        for bs_ in range(bs):
            for k in range(len(img_s_list)):
                mas = (mask_s_list[k][bs_, :, :] == 1).float()
                res = place_seed_points(mas, 8, 5, 100)
                s_seed[bs_, k, :, :] = res

        bs, _, max_num_sp, _ = s_seed.size()  # bs x shot x max_num_sp x 2

        FP = torch.zeros(bs, feature_s_list[0].shape[1], 25).cuda()

        # prototypes
        for bs_ in range(bs):
            sp_center_list = []
            for k in range(len(img_s_list)):
                with torch.no_grad():
                    supp_feat_ = feature_s_list[k][bs_, :, :, :].type(torch.float)  # c x h x w
                    supp_mask = mask_s_list[k][bs_, :, :].type(torch.float)  # 1 x h x w
                    supp_mask_ = F.interpolate(supp_mask.unsqueeze(0).unsqueeze(0), size=supp_feat_.shape[-2:],
                                               mode='bilinear', align_corners=True).squeeze(0)
                    s_seed_ = s_seed[bs_, k, :, :]  # max_num_sp x 2
                    num_sp = max(len(torch.nonzero(s_seed_[:, 0])), len(torch.nonzero(s_seed_[:, 1])))

                    if (num_sp == 0) or (num_sp == 1):
                        supp_proto = Weighted_GAP(supp_feat_.unsqueeze(0).type(torch.float),
                                                  supp_mask_.unsqueeze(0).type(torch.float))  # 1 x c x 1 x 1

                        sp_center_list.append(supp_proto.squeeze().unsqueeze(-1))  # c x 1
                        continue

                    s_seed_ = s_seed_[:num_sp, :].type(torch.long)  # num_sp x 2
                    sp_init_center = supp_feat_[:, s_seed_[:, 1], s_seed_[:, 0]].type(
                        torch.long)  # c x num_sp (sp_seed)

                    # sp_init_center = torch.cat([sp_init_center.cuda(), s_seed_.transpose(1, 0).float().cuda()], dim=0)  # (c + xy) x num_sp
                    sp_init_center = torch.cat([sp_init_center.cuda(), s_seed_.transpose(1, 0).cuda()],
                                               dim=0)  # (c + xy) x num_sp

                    if self.training:
                        sp_center = self.sp_center_iter(supp_feat_, supp_mask_, sp_init_center, n_iter=20)
                        sp_center_list.append(sp_center)
                    else:
                        sp_center = self.sp_center_iter(supp_feat_, supp_mask_, sp_init_center, n_iter=20)
                        sp_center_list.append(sp_center)

            sp_center = torch.cat(sp_center_list, dim=1)  # c x num_sp_all (collected from all shots)
            FP[bs_, :, :sp_center.shape[1]] = sp_center
        FP = FP.unsqueeze(-1)
        return FP

    def SGC_q(self, img_s_list, mask_s_list, feature_s_list, bs):
        s_seed = torch.zeros(bs, len(img_s_list), 10, 2)

        for bs_ in range(bs):
            for k in range(1):
                mas = (mask_s_list[k][bs_, :, :] == 1).float()
                res = place_seed_points(mas, 8, 10, 100)
                s_seed[bs_, k, :, :] = res

        bs, _, max_num_sp, _ = s_seed.size()  # bs x shot x max_num_sp x 2

        FP = torch.zeros(bs, feature_s_list[0].shape[1], 10).cuda()

        # prototypes
        for bs_ in range(bs):
            sp_center_list = []
            for k in range(1):
                with torch.no_grad():
                    supp_feat_ = feature_s_list[k][bs_, :, :, :].type(torch.float)  # c x h x w
                    supp_mask = mask_s_list[k][bs_, :, :].type(torch.float)  # 1 x h x w
                    supp_mask_ = F.interpolate(supp_mask.unsqueeze(0).unsqueeze(0), size=supp_feat_.shape[-2:],
                                               mode='bilinear', align_corners=True).squeeze(0)
                    s_seed_ = s_seed[bs_, k, :, :]  # max_num_sp x 2
                    num_sp = max(len(torch.nonzero(s_seed_[:, 0])), len(torch.nonzero(s_seed_[:, 1])))

                    if (num_sp == 0) or (num_sp == 1):
                        supp_proto = Weighted_GAP(supp_feat_.unsqueeze(0).type(torch.float),
                                                  supp_mask_.unsqueeze(0).type(torch.float))  # 1 x c x 1 x 1

                        sp_center_list.append(supp_proto.squeeze().unsqueeze(-1))  # c x 1
                        continue

                    s_seed_ = s_seed_[:num_sp, :].type(torch.long)  # num_sp x 2
                    sp_init_center = supp_feat_[:, s_seed_[:, 1], s_seed_[:, 0]].type(
                        torch.long)  # c x num_sp (sp_seed)

                    # sp_init_center = torch.cat([sp_init_center.cuda(), s_seed_.transpose(1, 0).float().cuda()], dim=0)  # (c + xy) x num_sp
                    sp_init_center = torch.cat([sp_init_center.cuda(), s_seed_.transpose(1, 0).cuda()],
                                               dim=0)  # (c + xy) x num_sp

                    if self.training:
                        sp_center = self.sp_center_iter(supp_feat_, supp_mask_, sp_init_center, n_iter=20)
                        sp_center_list.append(sp_center)
                    else:
                        sp_center = self.sp_center_iter(supp_feat_, supp_mask_, sp_init_center, n_iter=20)
                        sp_center_list.append(sp_center)

            sp_center = torch.cat(sp_center_list, dim=1)  # c x num_sp_all (collected from all shots)
            FP[bs_, :, :sp_center.shape[1]] = sp_center
        FP = FP.unsqueeze(-1)
        return FP

    def sp_center_iter(self, supp_feat, supp_mask, sp_init_center, n_iter):
        '''
        :param supp_feat: A Tensor of support feature, (C, H, W)
        :param supp_mask: A Tensor of support mask, (1, H, W)
        :param sp_init_center: A Tensor of initial sp center, (C + xy, num_sp)
        :param n_iter: The number of iterations
        :return: sp_center: The centroid of superpixels (prototypes)
        '''

        c_xy, num_sp = sp_init_center.size()
        _, h, w = supp_feat.size()
        h_coords = torch.arange(h).view(h, 1).contiguous().repeat(1, w).unsqueeze(0).float().cuda()
        w_coords = torch.arange(w).repeat(h, 1).unsqueeze(0).float().cuda()
        supp_feat = torch.cat([supp_feat.cuda(), h_coords, w_coords], 0).cuda()
        supp_feat_roi = supp_feat[:, (supp_mask == 1).squeeze()]  # (C + xy) x num_roi

        num_roi = supp_feat_roi.size(1)
        supp_feat_roi_rep = supp_feat_roi.unsqueeze(-1).repeat(1, 1, num_sp)
        sp_center = torch.zeros_like(sp_init_center).cuda()  # (C + xy) x num_sp

        for i in range(n_iter):
            # Compute association between each pixel in RoI and superpixel
            if i == 0:
                sp_center_rep = sp_init_center.unsqueeze(1).repeat(1, num_roi, 1)
            else:
                sp_center_rep = sp_center.unsqueeze(1).repeat(1, num_roi, 1)
            assert supp_feat_roi_rep.shape == sp_center_rep.shape  # (C + xy) x num_roi x num_sp
            dist = torch.pow(supp_feat_roi_rep - sp_center_rep, 2.0)
            feat_dist = dist[:-2, :, :].sum(0)
            spat_dist = dist[-2:, :, :].sum(0)
            total_dist = torch.pow(feat_dist + spat_dist / 100, 0.5)
            p2sp_assoc = torch.neg(total_dist).exp()
            p2sp_assoc = p2sp_assoc / (p2sp_assoc.sum(0, keepdim=True))  # num_roi x num_sp

            sp_center = supp_feat_roi_rep * p2sp_assoc.unsqueeze(0)  # (C + xy) x num_roi x num_sp
            sp_center = sp_center.sum(1)

        return sp_center[:-2, :]

    def similarity_func_0(self, feature_q, fg_proto, FP_0, bg_proto):
        similarity_fg_list = []
        for x in range(fg_proto.shape[2]):
            if not torch.equal(fg_proto[:, :, x, :].cuda(),
                               torch.zeros(fg_proto.shape[0], fg_proto.shape[1], 1, 1).cuda()):
                similarity_fg_ = F.cosine_similarity(feature_q.cuda(), fg_proto[:, :, x, :].unsqueeze(2).cuda(),
                                                     dim=1).unsqueeze(-1)
                similarity_fg_list.append(similarity_fg_)

        similarity_fg = torch.cat(similarity_fg_list, dim=-1)
        similarity_fg = torch.max(similarity_fg, dim=-1).values

        similarity_Fg = F.cosine_similarity(feature_q.cuda(), FP_0.cuda(), dim=1)
        total = similarity_fg + similarity_Fg
        similarity_fg = similarity_fg * (similarity_fg/total) + similarity_Fg * (similarity_Fg/total)

        similarity_bg = F.cosine_similarity(feature_q.cuda(), bg_proto.cuda(), dim=1)

        out = torch.cat((similarity_bg[:, None, ...], similarity_fg[:, None, ...]), dim=1) * 10.0
        return out

    def cross_prototype(self, feature_q, feature_s, mask):
        bs = feature_q.shape[0]
        mask = F.interpolate(mask.unsqueeze(1).float(), size=feature_s.shape[-2:], mode='bilinear', align_corners=True)
        prototype_list = []
        prototype_num_list = []
        select_loss = 0.0
        for epi in range(bs):
            index_list = []
            cur_feat_q = feature_q[epi].view(1024,-1)
            cur_feat_q_norm = cur_feat_q / torch.norm(cur_feat_q, 2, 0, True) # 1024, 3600
            cur_feat_s = feature_s[epi].view(1024,-1)
            cur_feat_s_norm = cur_feat_s / torch.norm(cur_feat_s, 2, 0, True) # 1024, 3600
            cur_feat_s_mask = (mask[epi].view(-1) == 1)
            cur_feat_s_fg = cur_feat_s[:, cur_feat_s_mask] # 1024, Ns
            mask_num = cur_feat_s_fg.shape[1]

            cur_feat_q_norm_t = cur_feat_q_norm.t()
            sim_1 = torch.matmul(cur_feat_q_norm_t, cur_feat_s_fg) * 2.0 # 3600, Ns
            index_q = sim_1.max(0)[1]
            cur_feat_q_fg = cur_feat_q[:, index_q] # 1024, Nq
            sim_2 = torch.matmul(cur_feat_s_norm.t(), cur_feat_q_fg) * 2.0 # 3600, Nq        
            index_s = sim_2.max(0)[1]
            index_judge = cur_feat_s_mask[index_s]
            index_cross = index_q[index_judge]
            if len(index_cross) == 0:

                cross_prototype = cur_feat_s_fg

            else:
                cross_prototype = cur_feat_q[:, index_cross]
            num = cross_prototype.shape[1]
            select_loss += (mask_num -num )/(mask_num+1)
            prototype_num_list.append(torch.tensor([num]).unsqueeze(0))
            prototype_list.append(torch.mean(cross_prototype, dim=-1).unsqueeze(0))

        return torch.cat(prototype_list, dim=0), torch.cat(prototype_num_list, dim=0), select_loss/bs

    def ECA(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

    def train_mode(self):
        self.train()
        self.backbone.eval()

    def SE(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

    def CBAM(self, x):
        out = x * self.sa(x)
        result = out * self.ca(out)
        #out = x * self.ca(x)
        #result = out * self.sa(out)
        return result






