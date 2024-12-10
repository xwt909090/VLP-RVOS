import torch.nn as nn
import torch
import torch.nn.functional as F
# import time
from util.FrozenBatchNorm2d import FrozenBatchNorm2d


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1,
         freeze_bn=False):
    if freeze_bn:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True),
            FrozenBatchNorm2d(out_planes),
            nn.ReLU(inplace=False))
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=False))


class Corner_Predictor(nn.Module):
    """ Corner Predictor module"""

    def __init__(self, inplanes=64, channel=256, stride=16, freeze_bn=False):
        super(Corner_Predictor, self).__init__()
        self.stride = stride
        '''top-left corner'''
        self.conv1_tl = conv(inplanes, channel, freeze_bn=freeze_bn)
        self.conv2_tl = conv(channel, channel // 2, freeze_bn=freeze_bn)
        self.conv3_tl = conv(channel // 2, channel // 4, freeze_bn=freeze_bn)
        self.conv4_tl = conv(channel // 4, channel // 8, freeze_bn=freeze_bn)
        self.conv5_tl = nn.Conv2d(channel // 8, 1, kernel_size=1)

        '''bottom-right corner'''
        self.conv1_br = conv(inplanes, channel, freeze_bn=freeze_bn)
        self.conv2_br = conv(channel, channel // 2, freeze_bn=freeze_bn)
        self.conv3_br = conv(channel // 2, channel // 4, freeze_bn=freeze_bn)
        self.conv4_br = conv(channel // 4, channel // 8, freeze_bn=freeze_bn)
        self.conv5_br = nn.Conv2d(channel // 8, 1, kernel_size=1)

    def forward(self, x, feat_sz, return_dist=False, softmax=True):
        """ Forward pass with input x. """
        h, w = feat_sz
        img_h = h * self.stride
        img_w = w * self.stride
        score_map_tl, score_map_br = self.get_score_map(x)
        if return_dist:
            coorx_tl, coory_tl, prob_vec_tl = self.soft_argmax(score_map_tl, feat_sz, return_dist=True, softmax=softmax)
            coorx_br, coory_br, prob_vec_br = self.soft_argmax(score_map_br, feat_sz, return_dist=True, softmax=softmax)
            bbox_ori = torch.stack((coorx_tl, coory_tl, coorx_br, coory_br), dim=1)
            coorx_tl = coorx_tl / img_w
            coory_tl = coory_tl / img_h
            coorx_br = coorx_br / img_w
            coory_br = coory_br / img_h
            return bbox_ori, torch.stack((coorx_tl, coory_tl, coorx_br, coory_br), dim=1), prob_vec_tl, prob_vec_br
        else:
            coorx_tl, coory_tl = self.soft_argmax(score_map_tl, feat_sz)
            coorx_br, coory_br = self.soft_argmax(score_map_br, feat_sz)
            bbox_ori = torch.stack((coorx_tl, coory_tl, coorx_br, coory_br), dim=1)
            coorx_tl = coorx_tl / img_w
            coory_tl = coory_tl / img_h
            coorx_br = coorx_br / img_w
            coory_br = coory_br / img_h
            return bbox_ori, torch.stack((coorx_tl, coory_tl, coorx_br, coory_br), dim=1)

    def get_score_map(self, x):
        # top-left branch
        x_tl1 = self.conv1_tl(x)
        x_tl2 = self.conv2_tl(x_tl1)
        x_tl3 = self.conv3_tl(x_tl2)
        x_tl4 = self.conv4_tl(x_tl3)
        score_map_tl = self.conv5_tl(x_tl4)

        # bottom-right branch
        x_br1 = self.conv1_br(x)
        x_br2 = self.conv2_br(x_br1)
        x_br3 = self.conv3_br(x_br2)
        x_br4 = self.conv4_br(x_br3)
        score_map_br = self.conv5_br(x_br4)
        return score_map_tl, score_map_br

    def soft_argmax(self, score_map, feat_sz, return_dist=False, softmax=True):
        """ get soft-argmax coordinate for a given heatmap """
        h, w = feat_sz
        score_vec = score_map.view((-1, h * w))  # (batch, feat_sz * feat_sz)
        prob_vec = nn.functional.softmax(score_vec, dim=1)

        '''about coordinates and indexs'''
        indice_h = torch.arange(0, h).view(-1, 1) * self.stride
        indice_w = torch.arange(0, w).view(-1, 1) * self.stride
        # generate mesh-grid
        coord_x = indice_w.repeat((h, 1)) \
            .view((h * w,)).float().cuda()
        coord_y = indice_h.repeat((1, w)) \
            .view((h * w,)).float().cuda()

        exp_x = torch.sum((coord_x * prob_vec), dim=1)
        exp_y = torch.sum((coord_y * prob_vec), dim=1)
        if return_dist:
            if softmax:
                return exp_x, exp_y, prob_vec
            else:
                return exp_x, exp_y, score_vec
        else:
            return exp_x, exp_y


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, BN=False):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        if BN:
            self.layers = nn.ModuleList(nn.Sequential(nn.Linear(n, k), nn.BatchNorm1d(k))
                                        for n, k in zip([input_dim] + h, h + [output_dim]))
        else:
            self.layers = nn.ModuleList(nn.Linear(n, k)
                                        for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build_box_head(cfg):
    if cfg.MODEL.HEAD_TYPE == "MLP":
        hidden_dim = cfg.MODEL.BOX_HEAD_HIDDEN_DIM
        mlp_head = MLP(hidden_dim, hidden_dim, 4, 3)  # dim_in, dim_hidden, dim_out, 3 layers
        return mlp_head
    elif cfg.MODEL.HEAD_TYPE == "CORNER":
        stride = 16
        feat_sz = int(cfg.DATA.SEARCH.SIZE / stride)
        channel = getattr(cfg.MODEL, "HEAD_DIM", 256)
        # print("head channel: %d" % channel)
        if cfg.MODEL.HEAD_TYPE == "CORNER":
            corner_head = Corner_Predictor(inplanes=cfg.MODEL.VL.HIDDEN_DIM, channel=channel,
                                           feat_sz=feat_sz, stride=stride)
        else:
            raise ValueError()
        return corner_head
    else:
        raise ValueError("HEAD TYPE %s is not supported." % cfg.MODEL.BOX_HEAD.HEAD_TYPE)


