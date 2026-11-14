#!/usr/bin/env python
"""Self-test of the attention U-Net. Run from the repository root:   python tools/check_model.py

Verifies (1) shapes, (2) that the baseline U-Net weights load into the attention model and that
the attention model then reproduces the baseline output EXACTLY at initialisation (zero gates),
(3) that attention really switches on when a gate is non-zero, (4) gradient flow, (5) the stored
attention maps are proper probability distributions, (6) other input sizes / configurations, and
prints a parameter and token table.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

from src.attention import auto_reduction
from src.model import UNet


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def gates(model):
    return [p for n, p in model.named_parameters() if n.endswith("gate") or ".gate_" in n]


def main():
    torch.manual_seed(0)
    x = torch.randn(2, 1, 256, 256)
    base = UNet(base_channels=32, depth=4).eval()
    cfg = dict(base_channels=32, depth=4, self_attn_levels=(4, 3), cross_attn_levels=(3, 2), image_size=256)
    attn = UNet(**cfg).eval()

    # 1. shape
    y = attn(x)
    assert y.shape == (2, 1, 256, 256), y.shape
    print("1  output shape OK:", tuple(y.shape))

    # 2/3. baseline weights load; identical output at init
    res = attn.load_state_dict(base.state_dict(), strict=False)
    assert not res.unexpected_keys
    assert all(k.startswith("self_attn.") or ".cross." in k for k in res.missing_keys), res.missing_keys
    with torch.no_grad():
        assert torch.allclose(base(x), attn(x), atol=1e-6)
    print(f"2  baseline weights load into attention model ({len(res.missing_keys)} new attention tensors)")
    print("3  attention model == baseline U-Net at initialisation (zero gates): exact match")

    # 4. attention switches on when gates open
    with torch.no_grad():
        for g in gates(attn):
            g.fill_(0.5)
        ref_out = base(x)
        diff = (attn(x) - ref_out).abs().max().item()
        assert diff > 1e-3 * ref_out.abs().max().item(), diff      # untrained logits are tiny, so compare relatively
    print(f"4  opening the {len(gates(attn))} gates changes the output: attention is wired in")

    # 5. gradient flow: at gate=0 only the gates get gradient; once open, attention weights learn
    for g in gates(attn):
        g.data.zero_()
    attn.train()
    attn(x).mean().backward()
    assert all(g.grad is not None and g.grad.abs().item() > 0 for g in gates(attn)), "gate without gradient"
    w = attn.self_attn["4"].attn.to_q.weight
    assert w.grad is None or w.grad.abs().max().item() == 0
    attn.zero_grad()
    for g in gates(attn):
        g.data.fill_(0.1)
    attn(x).mean().backward()
    assert attn.self_attn["4"].attn.to_q.weight.grad.abs().max().item() > 0
    assert attn.ups[0].cross.attn.to_k.weight.grad.abs().max().item() > 0
    print("5  gradients: gates learn first, then attention projections receive gradient")

    # 6. stored attention maps are proper distributions and match the fused kernel
    attn.eval()
    with torch.no_grad():
        ref = attn(x)
        for _, m in attn.attention_modules():
            m.store_attention = True
        out = attn(x)
        for name, m in attn.attention_modules():
            a = m.last_attn
            assert torch.allclose(a.sum(-1), torch.ones_like(a.sum(-1)), atol=1e-4), name
        assert torch.allclose(ref, out, atol=1e-4)
        for _, m in attn.attention_modules():
            m.store_attention = False
    print("6  attention rows sum to 1; manual softmax path == fused kernel")

    # 7. other sizes and configurations
    with torch.no_grad():
        assert attn(torch.randn(1, 1, 192, 192)).shape == (1, 1, 192, 192)
        for kw in (dict(base_channels=16, depth=4, bilinear=True, self_attn_levels=(4,), cross_attn_levels=(3,)),
                   dict(base_channels=32, depth=3, self_attn_levels=(3, 2), cross_attn_levels=(2, 1, 0)),
                   dict(base_channels=32, depth=4, self_attn_levels=(), cross_attn_levels=(1,))):
            m = UNet(image_size=256, **kw).eval()
            assert m(torch.randn(1, 1, 256, 256)).shape == (1, 1, 256, 256), kw
    for bad in (dict(self_attn_levels=(0,)), dict(self_attn_levels=(5,)), dict(cross_attn_levels=(4,))):
        try:
            UNet(depth=4, **bad)
            raise AssertionError(f"should have rejected {bad}")
        except ValueError:
            pass
    print("7  other input size (192), bilinear, depth 3, per-level choices, invalid levels rejected: OK")

    # table
    print("\nParameters")
    print(f"  baseline U-Net            : {n_params(base):>11,}")
    print(f"  + self- & cross-attention : {n_params(attn):>11,}  (+{n_params(attn) - n_params(base):,})")
    print("\nAttention layers at 256x256 input (base_channels=32):")
    print("  kind   level  channels  queries  keys(after reduction)  reduction")
    ch = [32 * 2 ** i for i in range(5)]
    for kind, levels in (("self", (4, 3)), ("cross", (3, 2))):
        for l in levels:
            r = auto_reduction(256, l, 1024)
            q = (256 // 2 ** l) ** 2
            print(f"  {kind:5s}  {l:>5}  {ch[l]:>8}  {q:>7}  {q // (r * r):>21}  {r:>9}")
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
