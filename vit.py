import torch
from torch import nn
import numpy as np
from einops import rearrange, repeat
from einops.layers.torch import Rearrange 
import time
        
class mlp( nn.Module ):
    def __init__(self, dim, hidden_dim, dropout=0 ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear( dim ,hidden_dim ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear( hidden_dim, dim ),
            nn.Dropout(dropout)
        )
    
    def forward(self , x):
        return self.net(x)

class patch_pos_embed( nn.Module ):
    def __init__(self, embed_dim, dim, patch_num, channels:int = 3, patch_size:int = 16, dropout=0 , bias=True):
        super().__init__()
        self.dim        = dim
        self.embed_dim  = embed_dim
        self.embed_num  = patch_num
        self.dp         = nn.Dropout(dropout) 
        self.position   = nn.Parameter( torch.randn(1, self.embed_num +1 , self.dim ) )
        self.cls_token  = nn.Parameter( torch.randn(1, 1, self.dim) )
        self.projection = nn.Sequential(
            # Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Conv2d( channels, self.embed_dim ,kernel_size=patch_size , stride=patch_size , bias=bias ),
            Rearrange('b e (h) (w) -> b (h w) e'),
            nn.LayerNorm( self.embed_dim ),
            nn.Linear( self.embed_dim , dim ),
            nn.LayerNorm(dim)
        )
         
    def forward(self , image):

        x = self.projection(image)
        b, n, _ = x.shape
        cls_token = repeat( self.cls_token , ' 1 1 d -> b 1 d ' , b=b )
        
        x  = torch.cat([cls_token, x], dim=1)
        x += self.position[:, :(n + 1)]
        x  = self.dp(x)
        return x

class multiAttention( nn.Module ):
    def __init__(self, dim, heads = 16, head_dim = 64, dropout = 0.):
        super().__init__()

        self.heads = heads
        self.scale = head_dim ** -0.5
        self.inner_dim = head_dim * heads
        project_out = not (heads == 1 and head_dim == dim)

        self.attend  = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv  = nn.Linear(dim, self.inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )if project_out else nn.Identity()
        
    def forward(self, x):
        
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots =  q @ k.transpose(-1, -2) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = attn @ v
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out , attn

class TransformerEncoder( nn.Module ):
    def __init__(self, dim, block_depth, heads, mlp_dim, head_dim=64, dropout=0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range( block_depth ):
            self.layers.append(nn.ModuleList([
                nn.LayerNorm(dim),
                multiAttention(dim=dim, heads=heads, head_dim=head_dim, dropout=dropout),
                nn.LayerNorm(dim),
                mlp( dim , hidden_dim=mlp_dim, dropout=dropout )
            ]))

    def forward(self, x):
        atts = []
        for ln, msa, ln, mlp in self.layers:
            
            ln_out        = ln(x)
            
            msa_out ,attn = msa(ln_out)
            msa_out       = msa_out + x
            
            ln_out        = ln(msa_out)
            
            mlp_out       = mlp(ln_out) + msa_out
            atts.append(attn)
        return mlp_out , atts
  
class vit(nn.Module):
    def __init__(self, img_shape, patch_size, dim, classes, block_depth, heads, mlp_dim, channels=3, dropout=0.5, emb_dropout=0.6 ):
        super().__init__()

        num_patch = (img_shape // patch_size) ** 2
        embed_dim = patch_size ** 2 * channels
        head_dim  = 64
        
        self.patch_pos_embed   = patch_pos_embed( embed_dim=embed_dim, dim=dim, patch_num=num_patch, channels=channels, patch_size=patch_size, dropout=emb_dropout )
        self.transformer       = TransformerEncoder( dim=dim, block_depth=block_depth, heads=heads, mlp_dim=mlp_dim, head_dim=head_dim, dropout=dropout  )
        self.identity          = nn.Identity()
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, classes)
        )
    def forward(self, img):

        x        = self.patch_pos_embed(img)
        x , atts = self.transformer(x)
        x        = x[:,0]
        x        = self.classifier(x)
        
        return x , atts


