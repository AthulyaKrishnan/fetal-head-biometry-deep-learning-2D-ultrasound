"""Self-attention and cross-attention blocks for 2D feature maps.

Terminology (see the documentation for a full beginner's explanation):
  * token      : one spatial position of a feature map, described by its C channel values.
                 A (B, C, H, W) feature map is treated as H*W tokens of size C.
  * query (Q)  : "what am I looking for?"   (one per position that is being updated)
  * key   (K)  : "what do I contain?"       (one per position that can be looked at)
  * value (V)  : "what do I hand over if I am chosen?"
  * attention  : softmax(Q K^T / sqrt(d)) V  - every query takes a weighted average of the values,
                 weights = how well its query matches each key.

Self-attention : Q, K and V all come from the SAME feature map (every position looks at every other).
Cross-attention: Q comes from one feature map (decoder), K and V from another (encoder skip).
"""
import math
from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- positional encoding
@lru_cache(maxsize=64)
def _sincos_2d_cached(h: int, w: int, dim: int, device_str: str):
    """Fixed 2D sine/cosine position code, shape (1, h*w, dim). Half of the channels encode the
    row index, half the column index, each as sin/cos pairs at geometrically spaced frequencies."""
    if dim % 4 != 0:
        raise ValueError(f"2D positional encoding needs channels divisible by 4, got {dim}")
    half = dim // 2                                   # channels for rows, same for columns
    n_freq = half // 2
    omega = 1.0 / (10000 ** (torch.arange(n_freq, dtype=torch.float32) / n_freq))   # (n_freq,)
    ys = torch.arange(h, dtype=torch.float32)[:, None] * omega[None]               # (h, n_freq)
    xs = torch.arange(w, dtype=torch.float32)[:, None] * omega[None]               # (w, n_freq)
    y_emb = torch.cat([ys.sin(), ys.cos()], dim=1)                                  # (h, half)
    x_emb = torch.cat([xs.sin(), xs.cos()], dim=1)                                  # (w, half)
    pe = torch.cat([y_emb[:, None, :].expand(h, w, half),
                    x_emb[None, :, :].expand(h, w, half)], dim=-1)                  # (h, w, dim)
    return pe.reshape(1, h * w, dim).to(device_str)


def sincos_2d(h, w, dim, device, dtype):
    return _sincos_2d_cached(int(h), int(w), int(dim), str(device)).to(dtype)


def auto_reduction(image_size: int, level: int, max_kv_tokens: int) -> int:
    """Smallest power-of-two stride r such that the keys/values have <= max_kv_tokens positions.
    At pyramid level `level` the feature map has (image_size / 2**level)**2 positions."""
    tokens = max(image_size // (2 ** level), 1) ** 2
    r = 1
    while tokens / (r * r) > max_kv_tokens:
        r *= 2
    return r


# --------------------------------------------------------------------------- core attention
def _sdpa(q, k, v, dropout_p=0.0):
    """softmax(q k^T / sqrt(d)) v   with q,k,v of shape (B, heads, N, d)."""
    if hasattr(F, "scaled_dot_product_attention"):        # fused / memory-efficient kernel (torch >= 2)
        return F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
    attn = (q @ k.transpose(-2, -1)) / math.sqrt(q.size(-1))
    attn = F.dropout(attn.softmax(dim=-1), p=dropout_p)
    return attn @ v


class MultiHeadAttention(nn.Module):
    """Learned projections + multi-head scaled dot-product attention.

    q_in : (B, Nq, q_dim)    k_in, v_in : (B, Nk, kv_dim)   ->   output (B, Nq, out_dim)
    Splitting the embedding into `num_heads` groups lets each head learn its own notion of
    'which positions are relevant' (e.g. one head may track edges, another the head centre).
    """

    def __init__(self, q_dim, kv_dim, embed_dim, out_dim, num_heads=4, dropout=0.0):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim {embed_dim} must be divisible by num_heads {num_heads}")
        self.h, self.d, self.dropout = num_heads, embed_dim // num_heads, dropout
        self.to_q = nn.Linear(q_dim, embed_dim)
        self.to_k = nn.Linear(kv_dim, embed_dim)
        self.to_v = nn.Linear(kv_dim, embed_dim)
        self.proj = nn.Linear(embed_dim, out_dim)
        self.store_attention = False          # set True to keep the attention weights (slow path)
        self.last_attn = None                 # (B, Nq, Nk), averaged over heads, when stored

    def _split(self, t):                      # (B, N, E) -> (B, heads, N, d)
        B, N, _ = t.shape
        return t.view(B, N, self.h, self.d).transpose(1, 2)

    def forward(self, q_in, k_in, v_in):
        q, k, v = self._split(self.to_q(q_in)), self._split(self.to_k(k_in)), self._split(self.to_v(v_in))
        if self.store_attention:              # explicit softmax so we can look at the weights
            attn = (q @ k.transpose(-2, -1) / math.sqrt(self.d)).softmax(dim=-1)
            self.last_attn = attn.mean(dim=1).detach()
            out = attn @ v
        else:
            out = _sdpa(q, k, v, self.dropout if self.training else 0.0)
        B, _, N, _ = out.shape
        return self.proj(out.transpose(1, 2).reshape(B, N, self.h * self.d))


class _KVReducer(nn.Module):
    """Optional spatial down-sampling of the keys/values (Pyramid-Vision-Transformer style).
    A strided convolution shrinks the (B, C, H, W) map by `r`, cutting the number of keys by r*r
    and therefore the attention cost from N*N to N*N/r^2."""

    def __init__(self, dim, r):
        super().__init__()
        self.r = r
        if r > 1:
            self.conv = nn.Conv2d(dim, dim, kernel_size=r, stride=r)
            self.norm = nn.LayerNorm(dim)

    def forward(self, tokens, H, W):          # tokens (B, N, C) -> (tokens', H', W')
        if self.r == 1:
            return tokens, H, W
        B, N, C = tokens.shape
        m = self.conv(tokens.transpose(1, 2).reshape(B, C, H, W))
        Hk, Wk = m.shape[-2:]
        return self.norm(m.flatten(2).transpose(1, 2)), Hk, Wk


# --------------------------------------------------------------------------- self-attention block
class SelfAttentionBlock2d(nn.Module):
    """Transformer-encoder layer on a feature map:   x = x + g1 * MHSA(LN(x));  x = x + g2 * FFN(LN(x)).

    * Pre-LayerNorm (LN before each sub-layer) for stable training.
    * 2D sine/cosine position codes are added to queries and keys (not values), so attention can
      use *where* things are, which plain convolution features do not tell it explicitly.
    * g1, g2 are learnable scalar gates initialised to 0, so at the start of training the block is
      the identity and the network behaves exactly like the plain U-Net; attention is switched on
      gradually only if it helps.
    """

    def __init__(self, dim, num_heads=4, reduction=1, mlp_ratio=2.0, dropout=0.0):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.reducer = _KVReducer(dim, reduction)
        self.attn = MultiHeadAttention(dim, dim, dim, dim, num_heads, dropout)
        hidden = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, dim))
        self.gate_attn = nn.Parameter(torch.zeros(1))
        self.gate_ffn = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape
        t = x.flatten(2).transpose(1, 2)                                  # (B, N, C) tokens
        n = self.norm1(t)
        q_in = n + sincos_2d(H, W, C, x.device, n.dtype)
        kv, Hk, Wk = self.reducer(n, H, W)
        k_in = kv + sincos_2d(Hk, Wk, C, x.device, n.dtype) if self.reducer.r > 1 else q_in
        v_in = kv if self.reducer.r > 1 else n
        t = t + self.gate_attn * self.attn(q_in, k_in, v_in)
        t = t + self.gate_ffn * self.ffn(self.norm2(t))
        return t.transpose(1, 2).reshape(B, C, H, W)


# --------------------------------------------------------------------------- cross-attention block
class CrossAttentionBlock2d(nn.Module):
    """Decoder-queries-encoder cross-attention used on a skip connection.

        decoder features  (after up-sampling)  -> QUERIES   "what context do I need here?"
        encoder skip features                   -> KEYS/VALUES  "here is the fine detail I stored"

    Every decoder position takes a learned weighted average of encoder-skip features from *all*
    positions (instead of just the one at the same spot, as plain concatenation does). The result,
    expressed in the skip's own channels, is added to the skip with a zero-initialised gate:

        skip_refined = skip + gate * CrossAttn(Q = decoder, K,V = skip)

    so the following concatenation / DoubleConv are unchanged (same shapes, same channel counts).
    """

    def __init__(self, dim_q, dim_kv, num_heads=4, reduction=1, dropout=0.0):
        super().__init__()
        self.norm_q, self.norm_kv = nn.LayerNorm(dim_q), nn.LayerNorm(dim_kv)
        self.reducer = _KVReducer(dim_kv, reduction)
        self.attn = MultiHeadAttention(dim_q, dim_kv, dim_q, dim_kv, num_heads, dropout)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, dec, skip):
        B, Cq, H, W = dec.shape
        Ck = skip.shape[1]
        assert skip.shape[-2:] == (H, W), "decoder and skip maps must have the same spatial size"
        q = self.norm_q(dec.flatten(2).transpose(1, 2))                   # (B, N, Cq)
        s = self.norm_kv(skip.flatten(2).transpose(1, 2))                 # (B, N, Ck)
        s, Hk, Wk = self.reducer(s, H, W)                                 # fewer keys if reduction > 1
        q_in = q + sincos_2d(H, W, Cq, dec.device, q.dtype)
        k_in = s + sincos_2d(Hk, Wk, Ck, dec.device, s.dtype)
        out = self.attn(q_in, k_in, s)                                    # (B, N, Ck)
        out = out.transpose(1, 2).reshape(B, Ck, H, W)
        return skip + self.gate * out
