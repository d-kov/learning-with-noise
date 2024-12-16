# coded with reference to 
# https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/simple_vit.py

import operator

import torch
import torch.nn as nn

from functools import reduce

from einops import rearrange
from einops.layers.torch import Rearrange


def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype = torch.float32):    
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    assert dim > 4, "dim must be greater than 4"

    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    
    return pe.type(dtype)


class Attention(nn.Module):
    def __init__(self, dim, no_heads):
        super(Attention, self).__init__()

        self.no_heads = no_heads
        
        self.to_qkv = nn.Linear(dim, 3*dim*no_heads, bias=False)
        self.to_out = nn.Linear(dim * no_heads, dim, bias=False)

        self.softmax = nn.Softmax(dim=-1)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x)

        q, k, v = rearrange(qkv, 'b n (c h d) -> c b h n d', c=3, h=self.no_heads)
        
        k = k.transpose(-2, -1)

        dot_prod = torch.matmul(q, k)

        attention = self.softmax(dot_prod)

        out = torch.matmul(attention, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)


class Forward(nn.Module):
    def __init__(self, dim, hidden_dim, out_dim):
        super(Forward, self).__init__()

        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim)
        )
    
    def forward(self, x):
        x = self.net(x)
        return x


class CustomTransformer(nn.Module):
    def __init__(self, dim, depth, no_heads, mlp_hidden_dim):
        super(CustomTransformer, self).__init__()

        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, no_heads),
                Forward(dim, mlp_hidden_dim, dim)
            ]))

    def forward(self, x):
        for attn, forward in self.layers:
            x = attn(x) + x
            x = self.norm(x)

            x = forward(x) + x
            x = self.norm(x)

        return x


class CustomViT(nn.Module):
    def __init__(self, chw, no_h_patches, no_w_patches, no_classes, dim, depth,
                 no_heads, mlp_hidden_dim):
        super(CustomViT, self).__init__()

        if (chw[1] % no_h_patches != 0
            or chw[2] % no_w_patches != 0):
            assert False, "Image can't be split into specified patch size"

        self.patch_dim = reduce(operator.mul, chw, 1) // no_w_patches // no_h_patches

        self.to_patches = nn.Sequential(
            Rearrange("b c (nh hp) (nw hw) -> b (nh nw) (c hp hw)", nh=no_h_patches, nw=no_w_patches),
            nn.LayerNorm(self.patch_dim),
            nn.Linear(self.patch_dim, dim),
            nn.LayerNorm(dim),
        )

        # print(no_h_patches, no_w_patches, dim)

        self.pos_embedding = posemb_sincos_2d(
            h = no_h_patches,
            w = no_w_patches,
            dim = dim,
            temperature=100
        ) 
        # print(self.pos_embedding.dtype)
        # print(self.pos_embedding)

        self.transformer = CustomTransformer(dim, depth, no_heads, mlp_hidden_dim)
        self.to_class = nn.Linear(dim, no_classes)

 
    def forward(self, x):
        device = x.device

        x = self.to_patches(x)
        pos_emb = self.pos_embedding.to(device, dtype=x.dtype)
        x += pos_emb

        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.to_class(x)
        return x


if __name__ == "__main__":
    m = CustomViT(
        chw=(1, 8, 8), 
        no_h_patches=1, 
        no_w_patches=1, 
        no_classes=2, 
        dim = 8, 
        depth = 1, 
        no_heads = 2, 
        mlp_hidden_dim = 5
    )
    
    input = torch.Tensor([[[[1, 2, 3, 4, 1, 2, 3, 4], [6, 7, 8, 800, 1, 2, 3, 4], [1, 2, 3, 4, 1, 2, 3, 4], [6, 7, 8, 800, 1, 2, 3, 4], [1, 2, 3, 4, 1, 2, 3, 4], [6, 7, 8, 800, 1, 2, 3, 4], [1, 2, 3, 4, 1, 2, 3, 4], [6, 7, 8, 800, 1, 2, 3, 4]]]])
    print(input.shape)
    
    m(input)