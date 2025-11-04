# src/models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math

# ---- simple patch embedding (2D) ----
class PatchEmbed(nn.Module):
    def __init__(self, in_chans, emb_dim, patch_size=16, img_size=256):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, emb_dim, kernel_size=patch_size, stride=patch_size)
        self.num_patches = (img_size // patch_size) ** 2

    def forward(self, x):
        # x: [B, C, H, W]
        x = self.proj(x)  # [B, emb_dim, H/ps, W/ps]
        x = x.flatten(2).transpose(1,2)  # [B, N, emb_dim]
        return x

# ---- MLP block ----
class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim=None, drop=0.):
        super().__init__()
        out_dim = out_dim or in_dim
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop)
    def forward(self,x):
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x

# ---- basic Transformer encoder block ----
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim*mlp_ratio), drop=drop)
    def forward(self, x):
        # x: [B, N, dim]
        x2 = self.norm1(x)
        attn_out, _ = self.attn(x2, x2, x2)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x

# ---- Cross-attention: query from A, key/value from B ----
class CrossAttention(nn.Module):
    def __init__(self, dim_q, dim_kv, num_heads, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim_q
        self.q_lin = nn.Linear(dim_q, dim_q)
        self.k_lin = nn.Linear(dim_kv, dim_q)
        self.v_lin = nn.Linear(dim_kv, dim_q)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim_q, dim_q)
        self.proj_drop = nn.Dropout(proj_drop)
        self.scale = (dim_q // num_heads) ** -0.5

    def forward(self, q_x, kv_x):
        # q_x: [B, Nq, Dq], kv_x: [B, Nk, Dkv]
        Q = self.q_lin(q_x)  # [B,Nq,Dq]
        K = self.k_lin(kv_x)
        V = self.v_lin(kv_x)
        B, Nq, D = Q.shape
        # reshape for heads
        h = self.num_heads
        Q = Q.view(B, Nq, h, D//h).transpose(1,2)  # [B,h,Nq,dh]
        K = K.view(B, -1, h, D//h).transpose(1,2)  # [B,h,Nk,dh]
        V = V.view(B, -1, h, D//h).transpose(1,2)
        attn = (Q @ K.transpose(-2,-1)) * self.scale  # [B,h,Nq,Nk]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ V)  # [B,h,Nq,dh]
        out = out.transpose(1,2).reshape(B, Nq, D)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out, attn  # return attn for explainability

# ---- ViT encoder for MRI ----
class ViTEncoder(nn.Module):
    def __init__(self, in_chans=1, emb_dim=256, img_size=256, patch_size=16, depth=6, heads=8):
        super().__init__()
        self.patch = PatchEmbed(in_chans, emb_dim, patch_size, img_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch.num_patches, emb_dim))
        self.blocks = nn.ModuleList([TransformerBlock(emb_dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(emb_dim)
    def forward(self, x):
        # x: [B,C,H,W]
        x = self.patch(x) + self.pos_embed
        attns = []
        for b in self.blocks:
            x = b(x)
        x = self.norm(x)
        return x  # [B, N, D]

# ---- LightViT for HSI: we treat spectral channels as additional embedding dimension ----
class LightViTEncoder(nn.Module):
    def __init__(self, in_channels_hsi, emb_dim=192, img_size=256, patch_size=16, depth=4, heads=6):
        super().__init__()
        # first reduce spectral dimension to emb_dim via 1x1 conv
        self.spec_proj = nn.Conv2d(in_channels_hsi, emb_dim, kernel_size=1)
        self.patch = PatchEmbed(emb_dim, emb_dim, patch_size, img_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch.num_patches, emb_dim))
        self.blocks = nn.ModuleList([TransformerBlock(emb_dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(emb_dim)
    def forward(self, x):
        # x: [B, C_hsi, H, W]
        x = self.spec_proj(x)  # [B, emb_dim, H, W]
        x = self.patch(x) + self.pos_embed
        for b in self.blocks:
            x = b(x)
        x = self.norm(x)
        return x  # [B, N, D]

# ---- Fusion decoder (simple) ----
class SimpleDecoder(nn.Module):
    def __init__(self, patch_size, emb_dim_mri, emb_dim_hsi, img_size=256, out_channels=1):
        super().__init__()
        self.patch_size = patch_size
        num_patches = (img_size // patch_size) ** 2
        # fuse dims
        fused_dim = emb_dim_mri + emb_dim_hsi
        self.fc = nn.Linear(fused_dim, fused_dim)
        self.conv_up = nn.Sequential(
            nn.ConvTranspose2d(fused_dim, fused_dim//2, kernel_size=patch_size, stride=patch_size),
            nn.ReLU(),
            nn.Conv2d(fused_dim//2, out_channels, kernel_size=1)
        )
    def forward(self, fused_tokens, B, H, W):
        # fused_tokens: [B, N, D]
        x = fused_tokens.transpose(1,2).view(B, -1, H//self.patch_size, W//self.patch_size)
        x = self.conv_up(x)
        return x  # [B, 1, H, W]

# ---- Full dual-modality model with bi-directional cross-attention ----
class DualModalityNet(nn.Module):
    def __init__(self, in_chans_mri=1, in_chans_hsi=30, img_size=256, patch_size=16):
        super().__init__()
        self.mri_enc = ViTEncoder(in_chans=in_chans_mri, emb_dim=256, img_size=img_size, patch_size=patch_size, depth=6, heads=8)
        self.hsi_enc = LightViTEncoder(in_channels_hsi=in_chans_hsi, emb_dim=192, img_size=img_size, patch_size=patch_size, depth=4, heads=6)
        self.cross1 = CrossAttention(dim_q=256, dim_kv=192, num_heads=8)
        self.cross2 = CrossAttention(dim_q=192, dim_kv=256, num_heads=6)
        self.norm_m = nn.LayerNorm(256)
        self.norm_h = nn.LayerNorm(192)
        # fuse
        self.fuse_linear = nn.Linear(256+192, 256)
        self.decoder = SimpleDecoder(patch_size=patch_size, emb_dim_mri=256, emb_dim_hsi=192, img_size=img_size, out_channels=1)

    def forward(self, mri, hsi):
        # mri: [B, C, H, W], hsi: [B, C_hsi, H, W]
        B, _, H, W = mri.shape
        m_tokens = self.mri_enc(mri)  # [B, N, 256]
        h_tokens = self.hsi_enc(hsi)  # [B, N, 192]

        # cross attention: MRI queries HSI (MRI gets spectral guidance)
        m2h, attn_m2h = self.cross1(m_tokens, h_tokens)  # [B,N,256]
        m_tokens = m_tokens + m2h
        m_tokens = self.norm_m(m_tokens)

        # cross attention: HSI queries MRI (HSI gets spatial guidance)
        h2m, attn_h2m = self.cross2(h_tokens, m_tokens)  # [B,N,192]
        h_tokens = h_tokens + h2m
        h_tokens = self.norm_h(h_tokens)

        # fuse per-token
        fused = torch.cat([m_tokens, h_tokens], dim=-1)  # [B,N,256+192]
        fused = self.fuse_linear(fused)  # [B,N,256]
        out = self.decoder(fused, B, H, W)  # [B,1,H,W]
        return out.sigmoid(), {'m2h': attn_m2h, 'h2m': attn_h2m}
