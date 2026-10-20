"""End-to-end BF16 FlashAttention variants for the MinerU X model.

Both the original ``HWAttention`` and the earlier ``HWAttentionBF16`` are left
untouched.  This module adds a third, opt-in variant whose FlashAttention path
never narrows a value to float16:

* ``HWAttention`` calls ``flash_attn_func(q.half(), ...)`` and casts the result
  back with ``.to(q.dtype)``.  Float16 tops out at 65504, so a large attention
  output overflows to ``inf`` inside the kernel.
* ``HWAttentionBF16`` widened the kernel inputs to bfloat16 but still ended with
  ``.to(dtype=q_dtype)``.  Under the stock fp16 autocast ``q_dtype`` is float16,
  so the overflow simply moved to that final cast.
* ``HWAttentionBF16Autocast`` (below) performs no narrowing cast at all.  It is
  meant to run under ``torch.autocast(dtype=torch.bfloat16)``, where the inputs
  already arrive as bfloat16 and the casts are no-ops.  bfloat16 carries the
  full float32 exponent range, so the 65504 ceiling disappears entirely.
"""

import torch
import torch.nn as nn
from torch.nn.functional import scaled_dot_product_attention as sdpa

from . import block as _block
from .block import C3k, HWBlock, HWC2f
from .conv import Conv


class HWAttentionBF16Autocast(nn.Module):
    """Height-width attention that stays in BF16 through FlashAttention."""

    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        all_head_dim = head_dim * self.num_heads
        self.qk = Conv(dim, all_head_dim * 2, 1, act=False)
        self.v = Conv(dim, all_head_dim, 1, act=False)
        self.proj = Conv(all_head_dim, dim, 1, act=False)
        self.pe = Conv(all_head_dim, dim, 5, 1, 2, g=dim, act=False)

    @staticmethod
    def _flash(q, k, v):
        """Run FlashAttention in BF16 and return its BF16 result unchanged.

        No cast back to the caller's dtype: that final narrowing is exactly the
        step where the fp16 variants overflowed.  Under bfloat16 autocast the
        surrounding graph already expects bfloat16, so nothing downstream needs
        a conversion here.
        """
        return _block.flash_attn_func(
            q.contiguous().to(dtype=torch.bfloat16),
            k.contiguous().to(dtype=torch.bfloat16),
            v.contiguous().to(dtype=torch.bfloat16),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        qk = self.qk(x)
        v_orig = self.v(x)
        pp = self.pe(v_orig)

        qk_col = qk.permute(0, 3, 1, 2).reshape(B * W, 2 * C, H).transpose(1, 2)
        v_col = v_orig.permute(0, 3, 1, 2).reshape(B * W, C, H).transpose(1, 2)
        B_col, N_col, _ = qk_col.shape
        q_col, k_col = qk_col.split([C, C], dim=2)
        if x.is_cuda and _block.USE_FLASH_ATTN and not torch.onnx.is_in_onnx_export():
            q_col = q_col.view(B_col, N_col, self.num_heads, self.head_dim)
            k_col = k_col.view(B_col, N_col, self.num_heads, self.head_dim)
            v_col = v_col.view(B_col, N_col, self.num_heads, self.head_dim)
            x_col = self._flash(q_col, k_col, v_col).view(B_col, N_col, C)
        else:
            q_col = q_col.reshape(B_col, N_col, self.num_heads, self.head_dim).transpose(1, 2)
            k_col = k_col.reshape(B_col, N_col, self.num_heads, self.head_dim).transpose(1, 2)
            v_col = v_col.reshape(B_col, N_col, self.num_heads, self.head_dim).transpose(1, 2)
            x_col = sdpa(q_col, k_col, v_col, dropout_p=0.0, is_causal=False)
            x_col = x_col.transpose(1, 2).contiguous().reshape(B_col, N_col, C)
        x_col = x_col.reshape(B, W, H, C).permute(0, 3, 2, 1).contiguous()

        qk_row = qk.permute(0, 2, 1, 3).reshape(B * H, 2 * C, W).transpose(1, 2)
        v_row = v_orig.permute(0, 2, 1, 3).reshape(B * H, C, W).transpose(1, 2)
        B_row, N_row, _ = qk_row.shape
        q_row, k_row = qk_row.split([C, C], dim=2)
        if x.is_cuda and _block.USE_FLASH_ATTN and not torch.onnx.is_in_onnx_export():
            q_row = q_row.view(B_row, N_row, self.num_heads, self.head_dim)
            k_row = k_row.view(B_row, N_row, self.num_heads, self.head_dim)
            v_row = v_row.view(B_row, N_row, self.num_heads, self.head_dim)
            x_row = self._flash(q_row, k_row, v_row).view(B_row, N_row, C)
        else:
            q_row = q_row.reshape(B_row, N_row, self.num_heads, self.head_dim).transpose(1, 2)
            k_row = k_row.reshape(B_row, N_row, self.num_heads, self.head_dim).transpose(1, 2)
            v_row = v_row.reshape(B_row, N_row, self.num_heads, self.head_dim).transpose(1, 2)
            x_row = sdpa(q_row, k_row, v_row, dropout_p=0.0, is_causal=False)
            x_row = x_row.transpose(1, 2).contiguous().reshape(B_row, N_row, C)
        x_row = x_row.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        # The three addends can disagree in dtype when this module runs outside
        # bfloat16 autocast (the attention half is bfloat16, ``pp`` follows the
        # ambient dtype).  Promote to the widest of them so the sum never
        # silently narrows the attention result.
        parts = (x_col, x_row, pp)
        out_dtype = torch.promote_types(torch.promote_types(x_col.dtype, x_row.dtype), pp.dtype)
        x_col, x_row, pp = (part.to(dtype=out_dtype) for part in parts)
        return self.proj(x_col + x_row + pp)


class HWBlockBF16Autocast(HWBlock):
    """HWBlock using :class:`HWAttentionBF16Autocast`."""

    def __init__(self, dim, num_heads, mlp_ratio=1.2, area=1):
        nn.Module.__init__(self)
        self.attn = HWAttentionBF16Autocast(dim, num_heads=num_heads)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(Conv(dim, mlp_hidden_dim, 1), Conv(mlp_hidden_dim, dim, 1, act=False))
        self.apply(self._init_weights)


class HWC2fBF16Autocast(HWC2f):
    """HWC2f built from BF16-autocast attention blocks; interface unchanged."""

    def __init__(self, c1, c2, n=1, a2=True, area=1, residual=False, mlp_ratio=2.0, e=0.5, g=1, shortcut=True):
        nn.Module.__init__(self)
        c_ = int(c2 * e)
        assert c_ % 32 == 0, "Dimension of ABlock be a multiple of 32."
        num_heads = c_ // 32
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv((1 + n) * c_, c2, 1)
        init_values = 0.01
        self.gamma = nn.Parameter(init_values * torch.ones((c2)), requires_grad=True) if a2 and residual else None
        self.m = nn.ModuleList(
            nn.Sequential(*(HWBlockBF16Autocast(c_, num_heads, mlp_ratio, area) for _ in range(2)))
            if a2
            else C3k(c_, c_, 2, shortcut, g)
            for _ in range(n)
        )
