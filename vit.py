import torch
from torch import nn
import numpy as np
from einops import rearrange, repeat
from einops.layers.torch import Rearrange 
        
class mlp( nn.Module ):
    def __init__(self, dim, hidden_dim, dropout=0 ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear( dim , hidden_dim ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear( hidden_dim, dim ),
            nn.Dropout(dropout)
        )
    
    def forward(self , x):
        return self.net(x)

class patch_embedding( nn.Module ):
    def __init__(self, dim, patch_num, channels:int = 3, patch_size:int = 16, dropout=0 ):
        super().__init__()

        self.embed_dim = patch_size * patch_size * 3
        self.embed_num = patch_num
        self.dim       = dim
        self.position  = nn.Parameter( torch.randn(1, self.embed_num +1 , self.dim ) )
        self.cls_token = nn.Parameter( torch.randn(1, 1, self.dim) )
        self.dp        = nn.Dropout(dropout) 
        self.net       = nn.Sequential(
            # nn.Conv2d( channels , self.embed_dim, kernel_size=patch_size , stride=patch_size ),
            # Rearrange('b e (h) (w) -> b (h w) e'),
            # nn.Linear(self.embed_dim, dim),
            # nn.LayerNorm(self.dim)
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, dim),
            nn.LayerNorm(dim),
        )
    def forward(self , image):

        x = self.net( image )
        b, n, _ = x.shape
        cls_token = repeat( self.cls_token , ' 1 1 e -> b 1 e ' , b=b )
        x = torch.cat([cls_token, x], dim=1)
        x += self.position[:, :(n + 1)]
        x = self.dp(x)
        return x

class multiAttention( nn.Module ):
    def __init__(self, dim, heads=16, head_dim=64, dropout=0):
        super().__init__()
        
        self.heads   = heads
        self.scale   = head_dim ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dp      = nn.Dropout(dropout)
        self.get_qkv = nn.Linear( dim , heads * head_dim * 3, bias=False  )
        self.projection = nn.Sequential(
            nn.Linear(heads * head_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self , x):

        qkv = self.get_qkv( x ).chunk(3,dim=-1) # qkv 為3個 [ 16 , 50 , 1024 ]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul( q , k.transpose(-1,-2) ) * self.scale
        x = self.softmax(dots)
        x = self.dp(x)
        out = torch.matmul( x , v )
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.projection(out)

        return out
    
class Attention(nn.Module):
    def __init__(self, dim, heads = 16, dim_head = 64, dropout = 0.):
        super().__init__()
        self.inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, self.inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(self.inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
  

class TransformerEncoder( nn.Module ):
    def __init__(self, dim, block_depth, heads, mlp_dim , head_dim=64, dropout=0):
        super().__init__()

        self.layers = nn.ModuleList([])
        for _ in range( block_depth ):
            self.layers.append(nn.ModuleList([
                nn.LayerNorm(dim),
                # Attention( dim=dim, heads=heads, dim_head=head_dim, dropout=dropout ),
                multiAttention(dim=dim, heads=heads, head_dim=head_dim, dropout=dropout),
                nn.LayerNorm(dim),
                mlp( dim=dim , hidden_dim=mlp_dim, dropout=dropout  )
            ]))

    def forward(self, x):
        for ln, msa, ln, mlp in self.layers:
            x = ln(x)
            x = msa(x) + x
            x = ln(x)
            x = mlp(x) + x
        return x
  
class vit(nn.Module):
    def __init__(self, img_shape, patch_size, classes, dim, block_depth, heads, head_dim, mlp_dim, channels=0, dropout=0, emb_dropout=0 ):
        super().__init__()

        num_patch = (img_shape // patch_size) ** 2
        self.patch_emb = patch_embedding( dim=dim, patch_num=num_patch, channels=channels, patch_size=patch_size, dropout=0 )
        self.transformer = TransformerEncoder( dim=dim, block_depth=block_depth, heads=heads, head_dim=head_dim, mlp_dim=mlp_dim, dropout=dropout  )
        self.identity = nn.Identity()
        self.classifier = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, classes)
        )
    def forward(self, img):

        x = self.patch_emb(img)
        x = self.transformer(x)
        x = x[:,0]
        x = self.classifier(x)
        return x


