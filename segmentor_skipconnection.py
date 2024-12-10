from logging import raiseExceptions
from numpy import short
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvN(nn.Module):
    def __init__(self, indim, outdim, kernel_size, norm='gn', gn_groups=8):
        super().__init__()
        self.conv = nn.Conv2d(indim, outdim, kernel_size, padding=kernel_size//2)
        if norm == 'gn':
            self.norm = nn.GroupNorm(gn_groups, outdim)
        elif norm == 'bn':
            self.norm = nn.BatchNorm2d(outdim)
        else:
            print('Use bn or gn in decoder.')
            raise
    
    def forward(self, x):
        return self.norm(self.conv(x))


class Segmentor(nn.Module):
    def __init__(self, in_dim,  # 从Transformer输入特征的维度
                       out_dim, # 输出mask的维度
                       hidden_dim=256,
                       shortcut_dims=[320, 640, 1280],   # Backbone特征的维度
                       align_corners=True,
                       upsample_logits=True,
                       scale_factor = 4,
                       norm='gn'):  # bn or gn
        super().__init__()
        self.K = 12
        self.upsample_logits = upsample_logits
        self.scale_factor = scale_factor
        self.align_corners = align_corners

        self.conv_in = ConvN(in_dim, hidden_dim, 1, norm=norm)
        self.conv_16x = ConvN(hidden_dim, hidden_dim, 3, norm=norm)
        self.conv_8x = ConvN(hidden_dim, hidden_dim//2, 3, norm=norm)
        self.conv_4x = ConvN(hidden_dim//2, hidden_dim//2, 3, norm=norm)

        self.adapter_16x = nn.Conv2d(shortcut_dims[-1], hidden_dim, 1)
        self.adapter_8x = nn.Conv2d(shortcut_dims[-2], hidden_dim, 1)
        self.adapter_4x = nn.Conv2d(shortcut_dims[-3], hidden_dim//2, 1)

        self.conv_out = nn.Conv2d(hidden_dim//2, out_dim, 1)
        self._init_weight()

    def forward(self, input, shortcuts, valid=None, multi_object_fusion=True):
        '''
        args:
            input: [B*no, C, h, w], input feature from transformer layers
            shortcuts: list of features, from backbone layer1, layer2, layer3
        return:
            [B*no, out_dim, H, W], output mask of each object
        '''
        x = F.relu_(self.conv_in(input))
        x = F.interpolate(x, size=shortcuts[-1].size()[-2:], mode='bilinear', align_corners=self.align_corners)
        x = F.relu_(self.conv_16x(self.adapter_16x(shortcuts[-1]) + x)) # 1/16 hidden_dim

        x = F.interpolate(x, size=shortcuts[-2].size()[-2:], mode='bilinear', align_corners=self.align_corners)
        x = F.relu_(self.conv_8x(self.adapter_8x(shortcuts[-2]) + x))   # 1/8 hidden_dim/2

        x = F.interpolate(x, size=shortcuts[-3].size()[-2:], mode='bilinear', align_corners=self.align_corners)
        x = F.relu_(self.conv_4x(self.adapter_4x(shortcuts[-3]) + x))   # 1/4 hidden_dim/2

        mask_logit = self.conv_out(x)

        if self.upsample_logits:
            # 上采样到原图的二分类logits
            mask_logit = F.interpolate(mask_logit, scale_factor=self.scale_factor, mode='bilinear', align_corners=self.align_corners)

        return mask_logit

    def _init_weight(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)




def build_segmentor(cfg):
    segmentor = Segmentor(cfg.MODEL.HIDDEN_DIM,
                          cfg.MODEL.DECODER.OUT_DIM,
                          hidden_dim=cfg.MODEL.HIDDEN_DIM,
                          shortcut_dims=cfg.MODEL.DECODER.SHORTCUT_DIMS,
                          align_corners=cfg.MODEL.DECODER.ALIGN_CORNER,
                          upsample_logits=cfg.MODEL.DECODER.UPSAMPLE_LOGITS,
                          norm=cfg.MODEL.DECODER.NORM,
                          pred_edge=cfg.MODEL.DECODER.PRED_EDGE)
    
    return segmentor


if __name__ =='__main__':
    segmentor = Segmentor(256, 2,
                        hidden_dim=256,
                        shortcut_dims=[256, 512, 1024],
                        align_corners=False,
                        norm='gn')

    input = torch.randn((5, 256, 24, 24))
    shortcuts = [
        torch.randn((5,256,96,96)),
        torch.randn((5,512,48,48)),
        torch.randn((5,1024,24,24)),
    ]

    out = segmentor(input, shortcuts)
    print(out.shape)