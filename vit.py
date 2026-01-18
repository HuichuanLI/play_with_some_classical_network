#coding:utf8
import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

## 预标准化方法
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

## 前向计算模块定义，包括两个全连接层
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
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

## 注意力模块定义
class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads ## 8*64=512
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5 ## 1/sqrt(64)=1/8

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False) ## 默认dim=1024，inner_dim * 3 = 512*3

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1) ## 从输入x，生成q,k,v，每一个维度 = inner_dim = dim_head * heads = 512
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale ## q*k/sqrt(d)
        attn = self.attend(dots) ## 得到softmax(q*k/sqrt(d))
        attn = self.dropout(attn)

        out = torch.matmul(attn, v) ## 得到softmax(q*k/sqrt(d))*v
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

## Transformer模型定义
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x ## 自注意力模块
            x = ff(x) + x ## feedforward模块
        return x

## ViT模型定义
class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size) ## 图像尺寸
        patch_height, patch_width = pair(patch_size) ## 裁剪子图尺寸

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width) ## 子图数量
        patch_dim = channels * patch_height * patch_width ## 展平后子图维度
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        ## 把图片切分为patch，然后拉成序列，假设输入图片大小是256x256（b,3,256,256），打算分成64个patch，每个patch是32x32像素，则rearrange操作是先变成(b,3,8x32,8x32)，最后变成(b,8x8,32x32x3)即(b,64,3072)
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim), ## 把子图维度映射到特定维度dim，比如32*32*3 -> 1024
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim)) ## num_patches=64，dim=1024,+1是因为多了一个cls开启解码标志
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim)) ## 额外的分类token
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        ## 在编码器后接fc分类器head即可
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        print('x shape', x.shape)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1) ## 在输入token数量维度进行拼接,## 额外追加token，变成b,65,1024
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0] ## 取第0个token的特征，或者所有特征的平均值

        x = self.to_latent(x)
        return self.mlp_head(x)


model = ViT(
    image_size = 256,
    patch_size = 32,
    num_classes = 1000,
    dim = 1024, ## token维度
    depth = 6, ## 模块数量
    heads = 16, ## 头的数量
    mlp_dim = 2048 ## mlp隐藏层维度
)

if __name__ == '__main__':
    img = torch.randn(1, 3, 256, 256)
    preds = model(img)
    print(preds.shape)

    torch.onnx.export(model, img, 'ViT.onnx')

