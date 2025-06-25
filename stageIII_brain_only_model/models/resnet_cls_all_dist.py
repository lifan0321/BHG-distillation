import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial
from models.resnet_cls import BasicBlock
from models.resnet_cls import ResNet_1
from models.resnet_cls_all import AllInOne
from einops import rearrange
import numpy as np
from torch import einsum

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
        # print("x : size : ", x.size())
        # print("weights size : ", weights.size())
        attn = self.attend(dots)    
        # print("attn size : ", attn.size())

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class AllInOne_Dist(nn.Module):

    def __init__(self, args):

        super(AllInOne_Dist, self).__init__()
        self.CNN_brain_dist = ResNet_1(BasicBlock, [3, 4, 6, 3], args.input_W, args.input_H, args.input_D, args.resnet_shortcut)
        self.CNN_brain_dist = self.CNN_brain_dist.cuda() 
        print(args.pretrain_path_brain, flush=True)
        pretrain = torch.load(args.pretrain_path_brain)
        self.CNN_brain_dist.load_state_dict(pretrain)

        self.AllInOne = AllInOne(args)
        self.AllInOne = self.AllInOne.cuda() 
        print(args.pretrain_path_allone, flush=True)
        pretrain_allone = torch.load(args.pretrain_path_allone)
        self.AllInOne.load_state_dict(pretrain_allone)

        self.convbrain1 = nn.Conv3d(in_channels=512, out_channels=256, kernel_size=(3, 5, 3), stride=(1, 1, 1), padding=(0, 0, 0))
        self.convbrain2 = nn.Conv3d(in_channels=256, out_channels=128, kernel_size=(3, 5, 3), stride=(1, 1, 1), padding=(0, 0, 0))
        self.brainbn = nn.BatchNorm3d(128)

        gradcam_dim = 4
        self.self_attentionb = Attention(64, heads=gradcam_dim)
        self.out1 = nn.Linear(64, 2)

        self.convbrain1 = self.convbrain1.cuda() 
        self.convbrain2 = self.convbrain2.cuda() 
        self.brainbn = self.brainbn.cuda() 
        self.self_attentionb = self.self_attentionb.cuda()

        self.relu = nn.ReLU(inplace=False)
        self.out1 = self.out1.cuda()

    def forward(self, brain, heart, gut):
        fb = self.CNN_brain_dist(brain)
        fb1 = self.convbrain1(fb)
        fb1 = self.convbrain2(fb1)
        fb2 = self.brainbn(fb1)

        avg_fb = torch.mean(fb2, dim=1)
        patch_size = 4
        b, h, w, d = avg_fb.size()
        fb2_patches = avg_fb.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        fb2_final = fb2_patches.contiguous().view(b, -1, patch_size**3)
        cross_b = self.self_attentionb(fb2_final, fb2_final)
        avg_cross_b = torch.mean(cross_b, dim=1)
        log_x1 = self.out1(self.relu(avg_cross_b))

        log_x4, out_global = self.AllInOne(brain, heart, gut)

        return log_x1, log_x4, avg_cross_b, out_global




