from torch import nn as nn
from torch.nn import functional as F
from einops import rearrange
import torch
import numbers
import functools
import math
from basicsr.utils.registry import ARCH_REGISTRY

# @ARCH_REGISTRY.register()
class ED_noskip(nn.Module):

    def __init__(self, in_ch=12,ch=48,out_ch=12,ch_mult=[1, 2, 4],embed_dim=48):
        super(ED_noskip, self).__init__()
        self.encoder=Encode_noskip(in_ch=in_ch,ch=ch,ch_mult=ch_mult,embed_dim=embed_dim)
        self.decoder=Decode_noskip(ch=ch,out_ch=out_ch,ch_mult=ch_mult,embed_dim=embed_dim)
    def forward(self,x):
        x1=self.encoder(x)
        x=self.decoder(x1)
        return x

class Encode_noskip(nn.Module):
    def __init__(self, in_ch=12,ch=12, ch_mult=[1, 2, 4, 4], embed_dim=4):
        super().__init__()
        self.depth = len(ch_mult)

        block_class = functools.partial(ResBlock, conv=default_conv, act=NonLinearity())

        self.init_conv1 = default_conv(in_ch, ch, 3)
        # layers
        self.encoder1= nn.ModuleList([])

        ch_mult = [1] + ch_mult
        for i in range(self.depth):
            dim_in = ch * ch_mult[i]
            dim_out = ch * ch_mult[i+1]
            self.encoder1.append(nn.ModuleList([
                block_class(dim_in=dim_in, dim_out=dim_in),
                block_class(dim_in=dim_in, dim_out=dim_in),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))) if i == (self.depth-1) else Identity(),
                Downsample(dim_in, dim_out) if i != (self.depth-1) else default_conv(dim_in, dim_out)
            ]))

        mid_dim = ch * ch_mult[-1]

        self.latent_conv1 =block_class(dim_in=mid_dim, dim_out=embed_dim)

    def check_image_size(self, x, h, w):
        s = int(math.pow(2, self.depth))
        mod_pad_h = (s - h % s) % s
        mod_pad_w = (s - w % s) % s
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward(self, x1):
        self.H, self.W = x1.shape[2:]
        x1 = self.check_image_size(x1, self.H, self.W)
        x1= self.init_conv1(x1)
        h1 = [x1]
        h=[]
        for b1, b2, attn, downsample in self.encoder1:
            x1 = b1(x1)
            h1.append(x1)
            x1 = b2(x1)
            x1 = attn(x1)
            h1.append(x1)
            x1 = downsample(x1)

        x1 = self.latent_conv1(x1)
        return x1



class Decode_noskip(nn.Module):
    def __init__(self, out_ch=12, ch=64, ch_mult=[1, 2, 4, 4], embed_dim=4):
        super().__init__()
        self.depth = len(ch_mult)

        block_class = functools.partial(ResBlock, conv=default_conv, act=NonLinearity())


        self.decoder = nn.ModuleList([])
        ch_mult = [1] + ch_mult
        for i in range(self.depth):
            dim_in = ch * ch_mult[i]
            dim_out = ch * ch_mult[i + 1]
            self.decoder.insert(0, nn.ModuleList([
                block_class(dim_in=dim_out, dim_out=dim_out),
                block_class(dim_in=dim_out, dim_out=dim_out),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))) if i == (self.depth - 1) else Identity(),
                Upsample(dim_out, dim_in) if i != 0 else default_conv(dim_out, dim_in)
            ]))

        mid_dim = ch * ch_mult[-1]

        self.post_latent_conv =block_class(dim_in=embed_dim, dim_out=mid_dim)

        self.final_conv = nn.Conv2d(ch, out_ch, 3, 1, 1)


    def check_image_size(self, x, h, w):
        s = int(math.pow(2, self.depth))
        mod_pad_h = (s - h % s) % s
        mod_pad_w = (s - w % s) % s
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward(self, x):
        x = self.post_latent_conv(x)
        for i, (b1, b2, attn, upsample) in enumerate(self.decoder):
            x = b1(x)
            x = b2(x)
            x = attn(x)
            x = upsample(x)
        x = self.final_conv(x)
        x=torch.tanh(x)
        return x


def default_conv(dim_in, dim_out, kernel_size=3, bias=False):
    return nn.Conv2d(dim_in, dim_out, kernel_size, padding=(kernel_size//2), bias=bias)
def exists(x):
    return x is not None
def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d
def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1)
    )
def NonLinearity(inplace=False):
    return nn.SiLU(inplace)

def Downsample(dim, dim_out=None):
    return nn.Conv2d(dim, default(dim_out, dim), 4, 2, 1)
class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            LayerNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(
            t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        v = v / (h * w)

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y',
                        h=self.heads, x=h, y=w)
        return self.to_out(out)
class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x
class Block(nn.Module):
    def __init__(self, conv, dim_in, dim_out, act=NonLinearity()):
        super().__init__()
        self.proj = conv(dim_in, dim_out)
        self.act = act

    def forward(self, x, scale_shift=None):
        x = self.proj(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x
class ResBlock(nn.Module):
    def __init__(self, conv, dim_in, dim_out, time_emb_dim=None, act=NonLinearity()):
        super(ResBlock, self).__init__()
        self.mlp = nn.Sequential(
            act, nn.Linear(time_emb_dim, dim_out * 2)
        ) if time_emb_dim else None

        self.block1 = Block(conv, dim_in, dim_out, act)
        self.block2 = Block(conv, dim_out, dim_out, act)
        self.res_conv = conv(dim_in, dim_out, 1) if dim_in != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)

        return h + self.res_conv(x)
class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x
if __name__ == '__main__':

    test_sample1 = torch.randn(1, 12, 256, 256).cuda(0)
    test_sample2 = torch.randn(1, 12, 256, 256).cuda(0)


    LDM_Enc = Encode_noskip(in_ch=12,ch=48,ch_mult=[1, 2, 4],embed_dim=48).cuda(0).eval()
    LDM_Dec = Decode_noskip(ch=48,out_ch=12,ch_mult=[1, 2, 4],embed_dim=48).cuda(0).eval()

    x1= LDM_Enc(test_sample1)

    x= LDM_Dec(x1)

    print(x.shape)
    total = sum([param.nelement() for param in LDM_Enc.parameters()])
    print("Number of parameters: %.2fM" % (total / 1e6))
