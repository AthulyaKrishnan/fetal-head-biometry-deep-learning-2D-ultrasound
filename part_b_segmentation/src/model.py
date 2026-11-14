"""2D U-Net (Ronneberger et al., 2015) with optional self-attention and cross-attention.

The convolutional U-Net is IDENTICAL to the baseline repository (same DoubleConv / Down / Up
blocks, same channel counts, same output head). Attention is added in two places only:

  1. SELF-attention  : after selected encoder stages (default: bottleneck and the stage above it).
  2. CROSS-attention : on selected skip connections; decoder features are the queries,
                       the encoder skip features are the keys/values (default: the two deepest).

Both are controlled by constructor arguments. With  self_attn_levels=() and cross_attn_levels=()
the network is exactly the baseline U-Net (and its weights load into the attention model).

Level numbering: level 0 = full resolution (first DoubleConv), level k = after k poolings,
level `depth` = bottleneck. Self-attention can sit at levels 1..depth, cross-attention on the
skip connection at levels 0..depth-1.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import CrossAttentionBlock2d, SelfAttentionBlock2d, auto_reduction


class DoubleConv(nn.Module):
    def __init__(self, cin, cout, dropout=0.0):
        super().__init__()
        layers = [nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                  nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    def __init__(self, cin, cout, dropout=0.0):
        super().__init__()
        self.pool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(cin, cout, dropout))

    def forward(self, x):
        return self.pool_conv(x)


class Up(nn.Module):
    def __init__(self, cin, cskip, cout, bilinear=False, dropout=0.0, cross=None):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            cup = cin
        else:
            self.up = nn.ConvTranspose2d(cin, cin // 2, 2, stride=2)
            cup = cin // 2
        self.cup, self.cskip = cup, cskip
        self.cross = cross                                  # CrossAttentionBlock2d or None
        self.conv = DoubleConv(cup + cskip, cout, dropout)

    def forward(self, x, skip):
        x = self.up(x)
        dy, dx = skip.size(2) - x.size(2), skip.size(3) - x.size(3)
        x = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        if self.cross is not None:                          # decoder asks the skip what it needs
            skip = self.cross(x, skip)
        return self.conv(torch.cat([skip, x], dim=1))


class UNet(nn.Module):
    """Returns raw logits of shape (B, n_classes, H, W)."""

    def __init__(self, in_channels=1, n_classes=1, base_channels=32, depth=4,
                 bilinear=False, dropout=0.0,
                 self_attn_levels=(), cross_attn_levels=(), num_heads=4, attn_dropout=0.0,
                 mlp_ratio=2.0, max_kv_tokens=1024, image_size=256):
        super().__init__()
        self_attn_levels, cross_attn_levels = tuple(self_attn_levels), tuple(cross_attn_levels)
        if any(not 1 <= l <= depth for l in self_attn_levels):
            raise ValueError(f"self_attn_levels must be within 1..{depth}, got {self_attn_levels}")
        if any(not 0 <= l <= depth - 1 for l in cross_attn_levels):
            raise ValueError(f"cross_attn_levels must be within 0..{depth - 1}, got {cross_attn_levels}")
        self.depth = depth
        ch = [base_channels * 2 ** i for i in range(depth + 1)]

        self.inc = DoubleConv(in_channels, ch[0])
        self.downs = nn.ModuleList(
            Down(ch[i], ch[i + 1], dropout if i == depth - 1 else 0.0) for i in range(depth))

        # --- self-attention after the chosen encoder stages (ModuleDict keys must be strings)
        self.self_attn = nn.ModuleDict({
            str(l): SelfAttentionBlock2d(ch[l], num_heads, auto_reduction(image_size, l, max_kv_tokens),
                                         mlp_ratio, attn_dropout)
            for l in self_attn_levels})

        # --- decoder; Up for level i has a cross-attention block if i is in cross_attn_levels
        ups = []
        for i in reversed(range(depth)):
            cin = ch[i + 1]
            cup = cin if bilinear else cin // 2
            cross = (CrossAttentionBlock2d(cup, ch[i], num_heads, auto_reduction(image_size, i, max_kv_tokens),
                                           attn_dropout) if i in cross_attn_levels else None)
            ups.append(Up(cin, ch[i], ch[i], bilinear, cross=cross))
        self.ups = nn.ModuleList(ups)
        self.outc = nn.Conv2d(ch[0], n_classes, 1)

    def forward(self, x):
        skips = [self.inc(x)]                               # level 0
        for lvl, down in enumerate(self.downs, start=1):
            x = down(skips[-1])
            key = str(lvl)
            if key in self.self_attn:                       # global context at this level
                x = self.self_attn[key](x)
            skips.append(x)
        x = skips.pop()                                     # bottleneck
        for up in self.ups:
            x = up(x, skips.pop())
        return self.outc(x)

    def attention_modules(self):
        """All (name, MultiHeadAttention) pairs - used by the visualisation tool."""
        from .attention import MultiHeadAttention
        return [(n, m) for n, m in self.named_modules() if isinstance(m, MultiHeadAttention)]
