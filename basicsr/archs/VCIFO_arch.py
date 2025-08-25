import math
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
import torchvision
from einops import rearrange
from basicsr.utils.registry import ARCH_REGISTRY
class VCIFO(nn.Module):

    def __init__(self, net_en,net_ir,depth=[1,1, 1, 1]):
        super(VCIFO, self).__init__()
        base_channel = 32
        # self.conv_first_vi = BasicConv(3, base_channel, 3, 1)
        # self.conv_first_ir = BasicConv(1, base_channel, 3, 1)
        self.Encoder_en = net_en.encoder
        self.Encoder_ir=net_ir.encoder
        # self.Encoder_en = nn.ModuleList([
        #     BasicConv(base_channel, base_channel, 3, 1),
        #     nn.Sequential(*[TransformerBlock(dim=base_channel, num_heads=8, ffn_expansion_factor=2,
        #                                      bias=False, LayerNorm_type='WithBias') for _ in range(1)]),
        #     Down_scale(base_channel),
        #     nn.Sequential(*[TransformerBlock(dim=base_channel*2, num_heads=8, ffn_expansion_factor=2,
        #                                      bias=False, LayerNorm_type='WithBias') for _ in range(1)]),
        #     Down_scale(base_channel * 2),
        #     nn.Sequential(*[TransformerBlock(dim=base_channel * 4, num_heads=8, ffn_expansion_factor=2,
        #                                      bias=False, LayerNorm_type='WithBias') for _ in range(1)]),
        #     Down_scale(base_channel * 4),
        # ])
        # self.Encoder_ir = nn.ModuleList([
        #     BasicConv(base_channel, base_channel, 3, 1),
        #     nn.Sequential(*[TransformerBlock(dim=base_channel, num_heads=8, ffn_expansion_factor=2,
        #                                      bias=False, LayerNorm_type='WithBias') for _ in range(1)]),
        #     Down_scale(base_channel),
        #     nn.Sequential(*[TransformerBlock(dim=base_channel*2, num_heads=8, ffn_expansion_factor=2,
        #                                      bias=False, LayerNorm_type='WithBias') for _ in range(1)]),
        #     Down_scale(base_channel * 2),
        #     nn.Sequential(*[TransformerBlock(dim=base_channel * 4, num_heads=8, ffn_expansion_factor=2,
        #                                      bias=False, LayerNorm_type='WithBias') for _ in range(1)]),
        #     Down_scale(base_channel * 4),
        # ])
        # Middle
        self.middle = nn.Sequential(*[TransformerBlock(dim=base_channel*8, num_heads=8, ffn_expansion_factor=2,
                                             bias=False, LayerNorm_type='WithBias') for _ in range(depth[1])])
        self.baseatt=nn.ModuleList(
            [
                BaseAttenBlock(base_channel),
                BaseAttenBlock(base_channel*2),
                BaseAttenBlock(base_channel * 4),
                BaseAttenBlock(base_channel * 8)
            ]
        )
        # self.Attconv=ConvLayer(1, 1, 3, 1)
        # self.se_main=ASELayer(base_channel*8)
        # self.skff_list=nn.ModuleList([
        #     ASKFF(base_channel),
        #     ASKFF(base_channel*2),
        #     ASKFF(base_channel*4),
        #     ASKFF(base_channel*8)
        # ])
        self.se_main = Corss_Attention(dim=base_channel * 8,num_heads=8,bias=False)
        #
        # self.conv_main=BasicConv(base_channel * 16, base_channel * 8, 3, 1)
        self.se_sub=nn.ModuleList([
            Corss_Attention(dim=base_channel , num_heads=8, bias=False),
            Corss_Attention(dim=base_channel * 2, num_heads=8, bias=False),
            Corss_Attention(dim=base_channel * 4, num_heads=8, bias=False)
            # ASELayer(base_channel),
            # ASELayer(base_channel * 2),
            # ASELayer(base_channel * 4)
        ])
        # self.conv_sub =nn.ModuleList([
        #     BasicConv(base_channel * 2, base_channel * 1, 3, 1),
        #     BasicConv(base_channel * 4, base_channel * 2, 3, 1),
        #     BasicConv(base_channel * 8, base_channel * 4, 3, 1)
        # ])
        # decoder
        self.Decoder = nn.ModuleList([
            Up_scale_van(base_channel * 8, 1),
            BasicConv(base_channel * 8, base_channel * 4, 3, 1),
            nn.Sequential(*[TransformerBlock(dim=base_channel * 4, num_heads=8, ffn_expansion_factor=2,
                                             bias=False, LayerNorm_type='WithBias') for _ in range(depth[2])]),
            Up_scale_van(base_channel * 4, 1),
            BasicConv(base_channel * 4, base_channel * 2, 3, 1),
            nn.Sequential(*[TransformerBlock(dim=base_channel * 2, num_heads=8, ffn_expansion_factor=2,
                                             bias=False, LayerNorm_type='WithBias') for _ in range(depth[2])]),
            Up_scale_van(base_channel * 2, 1),
            BasicConv(base_channel * 2, base_channel, 3, 1),
            nn.Sequential(*[TransformerBlock(dim=base_channel, num_heads=8, ffn_expansion_factor=2,
                                             bias=False, LayerNorm_type='WithBias') for _ in range(depth[1])]),
        ])
        self.conv_last = nn.Conv2d(base_channel, 3, 3, 1, 1)
    def fusion_sub(self,en_vis_out_list,en_ir_out__list):
        dim_list=[32,64,128]
        output_list=[]
        channel_atten_list=[]
        spatial_atten_list=[]
        for i,dim in enumerate(dim_list):
            en_vis_out=en_vis_out_list[i]
            en_ir_out = en_ir_out__list[i]
            output_1=self.se_sub[i](en_vis_out,en_ir_out)
            # output=torch.concat([en_vis_out,en_ir_out],dim=1)
            # output=self.conv_sub[i](output)
            # output_1=self.se_sub[i](output)
            output_2=self.baseatt[i](en_ir_out,en_vis_out)
            output=output_1+output_2
            # output=output+en_vis_out+en_ir_out
            # output =self.skff_list[i]([output_1,output_2])
            # output,v_base,v_detail,ir_base,ir_detail=self.askff_sub[i](en_vis_out,en_ir_out)
            output_list.append(output)
            channel_atten_list.append(output_1)
            spatial_atten_list.append(output_2)
            # vis_base_list.append(v_base)
            # vis_detail_list.append(v_detail)
            # ir_base_list.append(ir_base)
            # ir_detail_list.append(ir_detail)
        # return output_list,vis_base_list,vis_detail_list,ir_base_list,ir_detail_list
        return output_list,channel_atten_list,spatial_atten_list
        # return output_list
    def fusion_main(self,en_vis_out,en_ir_out):
        output_1=self.se_main(en_vis_out,en_ir_out)
        # output = torch.concat([en_vis_out, en_ir_out], dim=1)
        # output = self.conv_main(output)
        # output_1 = self.se_main(output)
        output_2 = self.baseatt[3](en_ir_out, en_vis_out)
        # output=self.skff_list[3]([output_1,output_2])
        output=output_1+output_2
        return output
    def Cem(self,input_f):
        input_m=torch.mean(input_f,dim=[2,3],keepdim=True)
        contrast=torch.sqrt(torch.mean((input_f - input_m)**2, dim=[2, 3], keepdim=True))
        output_f=torch.mean(contrast, dim=[2, 3], keepdim=True)
        return torch.multiply(input_f,output_f)

    # def Encoder_ir(self, x):
    #     # if self.input_type=="visible":
    #     #     x=torch.concat([x,x_he],dim=1)
    #     x = self.conv_first(x)
    #     shortcuts = []
    #     for i in range(len(self.Encoder)):
    #         x = self.Encoder[i](x)
    #         if (i + 2) % 3 == 0:
    #             shortcuts.append(x)
    #     return x, shortcuts
    def decoder(self, x, shortcuts):
        for i in range(len(self.Decoder)):
            if (i + 2) % 3 == 0:
                index = len(shortcuts) - (i // 3 + 1)
                size = x.shape[2:]
                size_s = shortcuts[index].shape[2:]
                if size != size_s:
                    x = F.interpolate(shortcuts[index], size=shortcuts[index].shape[2:], mode='bilinear',
                                      align_corners=True)
                x = torch.cat([x, shortcuts[index]], 1)
            x = self.Decoder[i](x)
        return x

    def forward(self, ir,vis):
        x1, shortcuts1 = self.Encoder_en(vis)
        x2, shortcuts2 = self.Encoder_ir(ir)
        # x1=self.grad_en(x1,x2)
        # x =  self.middle(x)
        # x,v_base,v_detail,ir_base,ir_detail = self.fusion_main(x1, x2)
        # shortcuts,vis_base_list,vis_detail_list,ir_base_list,ir_detail_list=self.fusion_sub(shortcuts1,shortcuts2)
        x_main=self.fusion_main(x1, x2)
        shortcuts,_,_=self.fusion_sub(shortcuts1,shortcuts2)
        # shortcuts= self.fusion_sub(shortcuts1, shortcuts2)
        # shortcuts_base = [self.baseatt(sc2,sc1) for sc1,sc2 in zip(shortcuts1,shortcuts2)]
        # x_base=self.baseatt(x2,x1)
        # vis_base_list.append(v_base)
        # vis_detail_list.append(v_detail)
        # ir_base_list.append(ir_base)
        # ir_detail_list.append(ir_detail)
        x = self.middle(x_main)
        # x=self.Cem(x)
        x = self.decoder(x, shortcuts)
        x = self.conv_last(x)
        gray = (torch.tanh(x) + 1) / 2
        # return gray,vis_base_list,vis_detail_list,ir_base_list,ir_detail_list
        # return gray,channel_atten,spatial_atten,shortcuts,shortcuts1,shortcuts2
        return gray
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)
class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight
class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias
class Corss_Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Corss_Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv_A = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv_A = nn.Conv2d(
            dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.qkv_B = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv_B = nn.Conv2d(
            dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out_A = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.project_out_B = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.concat_conv=nn.Conv2d(dim*2, dim, kernel_size=1, bias=bias)
    def forward(self, x,y):
        b, c, h, w = x.shape

        qkv_A = self.qkv_dwconv_A(self.qkv_A(x))
        qkv_B = self.qkv_dwconv_B(self.qkv_B(y))
        q_a, k_a, v_a = qkv_A.chunk(3, dim=1)
        q_b, k_b, v_b = qkv_B.chunk(3, dim=1)
        q_a = rearrange(q_a, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k_a = rearrange(k_a, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v_a = rearrange(v_a, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        q_b = rearrange(q_b, 'b (head c) h w -> b head c (h w)',
                        head=self.num_heads)
        k_b = rearrange(k_b, 'b (head c) h w -> b head c (h w)',
                        head=self.num_heads)
        v_b = rearrange(v_b, 'b (head c) h w -> b head c (h w)',
                        head=self.num_heads)
        q_a = torch.nn.functional.normalize(q_a, dim=-1)
        k_a = torch.nn.functional.normalize(k_a, dim=-1)
        q_b = torch.nn.functional.normalize(q_b, dim=-1)
        k_b = torch.nn.functional.normalize(k_b, dim=-1)
        attn_a = (q_b @ k_a.transpose(-2, -1)) * self.temperature
        attn_a = attn_a.softmax(dim=-1)

        out_a = (attn_a @ v_a)

        out_a = rearrange(out_a, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        out_a = self.project_out_A(out_a)
        out_a=out_a+x
        attn_b = -(q_a @ k_b.transpose(-2, -1)) * self.temperature
        attn_b = attn_b.softmax(dim=-1)

        out_b = (attn_b @ v_b)

        out_b = rearrange(out_b, 'b head c (h w) -> b (head c) h w',
                          head=self.num_heads, h=h, w=w)

        out_b = self.project_out_B(out_b)
        out_b = out_b + y
        out=torch.concat([out_a,out_b],dim=1)
        out=self.concat_conv(out)
        return out
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out
class AttentionBase(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,):
        super(AttentionBase, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv1 = nn.Conv2d(dim, dim*3, kernel_size=1, bias=qkv_bias)
        self.qkv2 = nn.Conv2d(dim*3, dim*3, kernel_size=3, padding=1, bias=qkv_bias)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        b, c, h, w = x.shape
        qkv = self.qkv2(self.qkv1(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        out = self.proj(out)
        return out
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(
            dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(
            hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x
class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=True, activation=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 -1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.InstanceNorm2d(out_channel))
        if activation:
            layers.append(nn.GELU())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)
class Down_scale(nn.Module):
    def __init__(self, in_channel):
        super(Down_scale, self).__init__()
        self.main = BasicConv(in_channel, in_channel*2, 3, 2)

    def forward(self, x):
        return self.main(x)
class Up_scale_van(nn.Module):
    def __init__(self, in_channel,kernel_size):
        super(Up_scale_van, self).__init__()
        # self.main = BasicConv(in_channel, in_channel//2, kernel_size=4, activation=True, stride=2, transpose=True)
        self.main=BasicConv(in_channel, in_channel//2,kernel_size,1)
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x=self.main(x)
        return x
class BaseAttenBlock(nn.Module):
    def __init__(self,channels):
        super(BaseAttenBlock, self).__init__()
        self.c=channels
        self.IN=nn.InstanceNorm2d(channels)
        self.alpha=nn.Parameter(torch.ones(1))
        self.beta=nn.Parameter(torch.ones(1))
    def forward(self,ir,vi):
        channels = vi.shape[1]
        ir_f=self.IN(ir)
        vi_f = self.IN(vi)
        sim_mat_l1 = torch.abs(ir_f - vi_f)  # <0  (b,c,h,w)
        # sim_mat_l1 = torch.mean(sim_mat_l1, dim=1, keepdim=True)  # (b,1,h,w)
        sim_mat_l1 = torch.tanh(sim_mat_l1/self.alpha)  # (0, 0.5) (b,1,h,w)
        sim_mat_l1 = torch.tanh(sim_mat_l1)
        # sim_mat_l1 = sim_mat_l1.repeat(1, channels, 1, 1)
        sim_mat_l1 = sim_mat_l1  # (0, 1)
        # cos distance
        sim_mat_cos = ir_f * vi_f # >0 (b,c,h,w)
        sim_mat_cos = torch.sum(sim_mat_cos, dim=1, keepdim=True)  # (b,1,h,w)
        mat_ir=torch.sqrt(torch.sum(ir_f**2,dim=1, keepdim=True))
        mat_vi = torch.sqrt(torch.sum(vi_f ** 2, dim=1, keepdim=True))
        sim_mat_cos=sim_mat_cos/(mat_ir*mat_vi)
        # sim_mat_cos=torch.exp(-sim_mat_cos)
        sim_mat_cos = -sim_mat_cos/self.beta
        # a, _ = torch.max(sim_mat_cos, dim=-1)
        # c = torch.mean(sim_mat_cos, dim=-1)
        # b, _ = torch.min(sim_mat_cos, dim=-1)
        # sim_mat_cos=sim_mat_cos/self.c
        sim_mat_cos = torch.sigmoid(sim_mat_cos)  # (0, 1) (b,1,h,w)

        sim_mat_cos = sim_mat_cos.repeat(1, channels, 1, 1)  # (0, 1)

        # similarity matrix
        sim_mat = sim_mat_l1 * sim_mat_cos  # (0, 1)
        # sim_mat_min=torch.min(sim_mat)
        # sim_mat_max = torch.max(sim_mat)
        #embeding
        f_emb = (ir + vi) * sim_mat
        return f_emb
class ASKFF(nn.Module):
    def __init__(self, in_channels, height=2, reduction=8, bias=False):
        super(ASKFF, self).__init__()

        self.height = height
        d = max(int(in_channels / reduction), 4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(nn.Conv2d(in_channels, d, 1, padding=0, bias=bias), nn.LeakyReLU(0.2))

        self.fcs = nn.ModuleList([])
        for i in range(self.height):
            self.fcs.append(nn.Conv2d(d, in_channels, kernel_size=1, stride=1, bias=bias))

        self.softmax = nn.Softmax(dim=1)

    def forward(self, inp_feats):
        batch_size = inp_feats[0].shape[0]
        n_feats = inp_feats[0].shape[1]

        inp_feats = torch.cat(inp_feats, dim=1)
        inp_feats = inp_feats.view(batch_size, self.height, n_feats, inp_feats.shape[2], inp_feats.shape[3])

        feats_U = torch.sum(inp_feats, dim=1)
        feats_S = self.avg_pool(feats_U)
        feats_Z = self.conv_du(feats_S)

        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.height, n_feats, 1, 1)
        # stx()
        attention_vectors = self.softmax(attention_vectors)
        # print(attention_vectors[:,1,1,:,:])
        # print(attention_vectors[:, 0, 1, :, :])
        feats_V = torch.sum(inp_feats * attention_vectors, dim=1)

        return feats_V