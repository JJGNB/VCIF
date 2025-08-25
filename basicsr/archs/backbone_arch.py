from torch import nn as nn
from torch.nn import functional as F
from einops import rearrange
import torch
import numbers
from basicsr.utils.registry import ARCH_REGISTRY
@ARCH_REGISTRY.register()
class backbone(nn.Module):

    def __init__(self,source="vi", base_channel=32, depth=[1, 1, 1, 1],de_type="LLIE"):
        super(backbone, self).__init__()
        if source=="vi":
            self.dim=3
        else:
            self.dim=1
        base_channel = 32
        # self.Encoder_color = net_c.encoder
        # self.pce = pce()
        # encoder
        self.Encoder = nn.ModuleList([
            BasicConv(base_channel, base_channel, 3, 1),
            nn.Sequential(*[TransformerBlock(dim=base_channel, num_heads=8, ffn_expansion_factor=2,
                                             bias=False, LayerNorm_type='WithBias',de_type=de_type) for _ in range(depth[0])]),
            Down_scale(base_channel),
            nn.Sequential(*[TransformerBlock(dim=base_channel*2, num_heads=8, ffn_expansion_factor=2,
                                             bias=False, LayerNorm_type='WithBias',de_type=de_type) for _ in range(depth[1])]),
            Down_scale(base_channel * 2),
            nn.Sequential(*[TransformerBlock(dim=base_channel * 4, num_heads=8, ffn_expansion_factor=2,
                                             bias=False, LayerNorm_type='WithBias',de_type=de_type) for _ in range(depth[2])]),
            Down_scale(base_channel * 4),
        ])
        # for k in [0,2,4,6]:
        #     froze_weight(self.Encoder[k])

        # Middle
        self.middle = nn.Sequential(*[TransformerBlock(dim=base_channel*8, num_heads=8, ffn_expansion_factor=2,
                                             bias=False, LayerNorm_type='WithBias',de_type=de_type) for _ in range(depth[3])])
        # decoder
        self.Decoder = nn.ModuleList([
            Up_scale_van(base_channel * 8, 1),
            BasicConv(base_channel * 8, base_channel * 4, 3, 1),
            nn.Sequential(*[TransformerBlock(dim=base_channel * 4, num_heads=8, ffn_expansion_factor=2,
                                             bias=False, LayerNorm_type='WithBias',de_type=de_type) for _ in range(depth[2])]),
            Up_scale_van(base_channel * 4, 1),
            BasicConv(base_channel*4, base_channel*2, 3, 1),
            nn.Sequential(*[TransformerBlock(dim=base_channel * 2, num_heads=8, ffn_expansion_factor=2,
                             bias=False, LayerNorm_type='WithBias',de_type=de_type) for _ in range(depth[1])]),
            Up_scale_van(base_channel * 2, 1),
            BasicConv(base_channel * 2, base_channel, 3, 1),
            nn.Sequential(*[TransformerBlock(dim=base_channel, num_heads=8, ffn_expansion_factor=2,
                                             bias=False, LayerNorm_type='WithBias',de_type=de_type) for _ in range(depth[0])]),
        ])
        for k in [0,1,3,4,6,7]:
            froze_weight(self.Decoder[k])
        # conv
        self.conv_first =BasicConv(self.dim, base_channel, 3, 1)
        # froze_weight(self.conv_first)
        self.conv_last = nn.Conv2d(base_channel, self.dim, 3, 1, 1)
        # froze_weight(self.conv_last)
    def encoder(self, x):
        # if self.input_type=="visible":
        #     x=torch.concat([x,x_he],dim=1)
        # x_color, _ = self.Encoder_color(x)
        x = self.conv_first(x)
        # if self.input_type=="visible":
        #     x=self.se(x)
        shortcuts = []
        for i in range(len(self.Encoder)):
            x = self.Encoder[i](x)

            if i==1 or i==3 or i==5:
                shortcuts.append(x)
        # x=self.FPA_main(x)
        # sr1 = self.FPA_sub[0](shortcuts[0])
        # sr2 = self.FPA_sub[1](shortcuts[1])
        # sr3 = self.FPA_sub[2](shortcuts[2])
        # shortcuts = [sr1, sr2, sr3]
        # shortcuts = self.pce(x_color, shortcuts)
        return x, shortcuts

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

    def forward(self, x):
        x, shortcuts = self.encoder(x)
        x = self.middle(x)
        # x=gaussian_blur2d(x,(5,5),(2,2))
        # shortcuts=[gaussian_blur2d(sr,(5,5),(2,2)) for sr in shortcuts]
        x = self.decoder(x, shortcuts)
        x = self.conv_last(x)
        gray = (torch.tanh(x) + 1) / 2
        return gray
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
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type,de_type="LLIE"):
        super(TransformerBlock, self).__init__()
        self.de_type=de_type
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        # self.norm1.requires_grad_(requires_grad=False)
        froze_weight(self.norm1)
        froze_weight(self.attn)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        froze_weight(self.norm2)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
        # self.ffn_Dehaze=FeedForward(dim, ffn_expansion_factor, bias)
        # self.ffn_Derain=FeedForward(dim, ffn_expansion_factor, bias)
        # if de_type=="LLIE":
        #     froze_weight(self.ffn_Dehaze)
        #     froze_weight(self.ffn_Derain)
        # elif de_type=="Dehaze":
        #     # for param_name, param in self.ffn.named_parameters():
        #     #     self.ffn_Dehaze._parameters[param_name] = param.detach().clone()
        #     froze_weight(self.ffn)
        #     froze_weight(self.ffn_Derain)
        # elif de_type=="Derain":
        #     # for param_name, param in self.ffn.named_parameters():
        #     #     self.ffn_Derain._parameters[param_name] = param.detach().clone()
        #     froze_weight(self.ffn_Dehaze)
        #     froze_weight(self.ffn)
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        # if self.de_type=="LLIE":
        x = x + self.ffn(self.norm2(x))
        # elif self.de_type=="Dehaze":
        #     x = x + self.ffn_Dehaze(self.norm2(x))
        # elif self.de_type=="Derain":
        #     x = x + self.ffn_Derain(self.norm2(x))
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
def froze_weight(model):
    # for param in model.parameters():
    #     param.requires_grad = False
    pass