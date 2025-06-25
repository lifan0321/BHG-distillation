import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial
from models.resnet_cls_1 import BasicBlock
from models.resnet_cls_1 import ResNet_1
from einops import rearrange
import numpy as np
from torch import einsum

class AftNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.norm(self.fn(x, **kwargs))

class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head=64, dropout=0.5): #dropout=0.5
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context):
        b, n, _ = x.shape
        h = self.heads

        qkv = (self.to_q(x), *self.to_kv(context).chunk(2, dim=-1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = self.attend(dots)    
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                AftNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                AftNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x, context=None):
        for attn, ff in self.layers:
            x = attn(x, context=context) + x
            x = ff(x) + x
        return self.norm(x)


class AllInOne(nn.Module):

    def __init__(self, args, embed_dim=64, num_classes=2):

        super(AllInOne, self).__init__()
        self.CNN_brain = ResNet_1(BasicBlock, [3, 4, 6, 3], args.input_W, args.input_H, args.input_D, args.resnet_shortcut)
        self.CNN_brain = self.CNN_brain.cuda() 
        print(args.pretrain_path_brain, flush=True)
        pretrain = torch.load(args.pretrain_path_brain)
        self.CNN_brain.load_state_dict(pretrain)

        self.CNN_heart = ResNet_1(BasicBlock, [3, 4, 6, 3], args.input_W, args.input_H, args.input_D, args.resnet_shortcut)
        self.CNN_heart = self.CNN_heart.cuda() 
        print(args.pretrain_path_heart, flush=True)
        pretrain = torch.load(args.pretrain_path_heart)
        self.CNN_heart.load_state_dict(pretrain)

        self.CNN_gut = ResNet_1(BasicBlock, [3, 4, 6, 3], args.input_W, args.input_H, args.input_D, args.resnet_shortcut)
        self.CNN_gut = self.CNN_gut.cuda() 
        print(args.pretrain_path_gut, flush=True)
        pretrain = torch.load(args.pretrain_path_gut)
        self.CNN_gut.load_state_dict(pretrain)

        self.convbrain1 = nn.Conv3d(in_channels=512, out_channels=256, kernel_size=(3, 5, 3), stride=(1, 1, 1), padding=(0, 0, 0))
        self.convbrain2 = nn.Conv3d(in_channels=256, out_channels=128, kernel_size=(3, 5, 3), stride=(1, 1, 1), padding=(0, 0, 0))
        self.brainbn = nn.BatchNorm3d(128)

        self.convheart = nn.Conv3d(in_channels=512, out_channels=128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
        self.heartbn = nn.BatchNorm3d(128)

        self.convgut = nn.Conv3d(in_channels=512, out_channels=128, kernel_size=(3, 5, 5), stride=(2, 1, 3), padding=(0, 0, 0))
        self.gutbn = nn.BatchNorm3d(128)

        self.bbn = nn.BatchNorm1d(4096)
        self.hbn = nn.BatchNorm1d(4096)
        self.gbn = nn.BatchNorm1d(4096)

        self.chbn = nn.BatchNorm1d(64)
        self.cgbn = nn.BatchNorm1d(64)

        head_dim = 4
        self.self_attentionb = Transformer(embed_dim, 1, 4, 64, 64)
        self.cross_attentionh = Transformer(embed_dim, 1, 4, 64, 64)
        self.cross_attentiong = Transformer(embed_dim, 1, 4, 64, 64)


        self.out1 = nn.Linear(64, num_classes)
        self.out2 = nn.Linear(64, num_classes)
        self.out3 = nn.Linear(64, num_classes)
        self.outall = nn.Linear(64, num_classes)

        self.convbrain1 = self.convbrain1.cuda() 
        self.convbrain2 = self.convbrain2.cuda() 
        self.brainbn = self.brainbn.cuda() 
        self.self_attentionb = self.self_attentionb.cuda()

        self.convheart = self.convheart.cuda() 
        self.heartbn = self.heartbn.cuda()
        self.cross_attentionh = self.cross_attentionh.cuda()

        self.convgut = self.convgut.cuda() 
        self.gutbn = self.gutbn.cuda()
        self.cross_attentiong = self.cross_attentiong.cuda()

        self.bbn = self.bbn.cuda()
        self.hbn = self.hbn.cuda()
        self.gbn = self.gbn.cuda()
        self.chbn = self.chbn.cuda()
        self.cgbn = self.cgbn.cuda()

        self.relu = nn.ReLU(inplace=False)
        self.out1 = self.out1.cuda()
        self.out2 = self.out2.cuda()
        self.out3 = self.out3.cuda()
        self.outall = self.outall.cuda()

    def forward(self, brain, heart, gut):
        fb = self.CNN_brain(brain)
        fh = self.CNN_heart(heart)
        fg = self.CNN_gut(gut)

        fb1 = self.convbrain1(fb)
        fb1 = self.convbrain2(fb1)
        fb2 = self.brainbn(fb1)

        fh1 = self.convheart(fh)
        fh1 = self.heartbn(fh1)

        fg1 = self.convgut(fg)
        fg2 = self.gutbn(fg1)

        avg_fb = torch.mean(fb2, dim=1)
        avg_fh = torch.mean(fh1, dim=1)
        avg_fg = torch.mean(fg2, dim=1)

        avg_fb_flatten = avg_fb.view(avg_fb.size(0), -1)
        avg_fh_flatten = avg_fh.view(avg_fh.size(0), -1)
        avg_fg_flatten = avg_fg.view(avg_fg.size(0), -1)

        avg_fb_detach = avg_fb_flatten.detach()
        avg_fb_detach = self.bbn(avg_fb_detach)
        avg_fh_flatten = self.hbn(avg_fh_flatten)
        avg_fg_flatten = self.gbn(avg_fg_flatten)

        h_b = avg_fh_flatten @ avg_fb_detach.T
        g_b = avg_fg_flatten @ avg_fb_detach.T

        patch_size = 4
        b, h, w, d = avg_fb.size()
        fb2_patches = avg_fb.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        fb2_final = fb2_patches.contiguous().view(b, -1, patch_size**3)

        fh1_patches = avg_fh.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        fh1_final = fh1_patches.contiguous().view(b, -1, patch_size**3)

        fg2_patches = avg_fg.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        fg2_final = fg2_patches.contiguous().view(b, -1, patch_size**3)

        cross_b = self.self_attentionb(fb2_final, fb2_final)
        cross_h = self.cross_attentionh(fb2_final, fh1_final)
        cross_g = self.cross_attentiong(fb2_final, fg2_final)

        avg_cross_b = torch.mean(cross_b, dim=1)
        avg_cross_h = torch.mean(cross_h, dim=1)
        avg_cross_g = torch.mean(cross_g, dim=1)

        fusion = torch.stack((avg_cross_b, avg_cross_h, avg_cross_g))
        fusion = torch.mean(fusion, dim=0)
        out_global = fusion.view(fusion.shape[0], -1)

        avg_cross_h_flatten = self.chbn(avg_cross_h)
        avg_cross_g_flatten = self.cgbn(avg_cross_g)

        fg_ca_detach = avg_cross_h_flatten.clone().detach()
        fh_ca_detach = avg_cross_g_flatten.clone().detach()
        per_fh_ca = avg_cross_g_flatten @ fg_ca_detach.T
        per_fg_ca = avg_cross_h_flatten @ fh_ca_detach.T

        log_x1 = self.out1(self.relu(avg_cross_b))
        log_x2 = self.out2(self.relu(avg_cross_h))
        log_x3 = self.out3(self.relu(avg_cross_g))
        log_x4 = self.outall(self.relu(out_global))

        return log_x4, out_global




