"""Pure-PyTorch Mamba (selective SSM, Gu & Dao 2023) DPD backbone.

Per-token output to match OpenDPD's (B, T, 2) contract.
Internal IQ augmentation (matches DGRU): (I, Q, |x|, |x|^3, cos, sin) → 6 features.
"""
__author__ = "Minho Choi"
__license__ = "Apache-2.0 License"

import math
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch._higher_order_ops import associative_scan


def _scan_combine(x, y):
    a1, b1 = x
    a2, b2 = y
    return (a1 * a2, a2 * b1 + b2)


def _assoc_scan(a: Tensor, b: Tensor) -> Tensor:
    _, h = associative_scan(_scan_combine, (a.contiguous(), b.contiguous()), dim=1)
    return h


class DiagScanFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        with torch.no_grad():
            h = _assoc_scan(a, b)
        ctx.save_for_backward(a, b, h)
        return h

    @staticmethod
    def backward(ctx, grad_h):
        a, b, h = ctx.saved_tensors
        Bsz, T, D = a.shape
        with torch.no_grad():
            grad_h_rev = grad_h.flip(1)
            a_rev = a.flip(1)
            zero_col = a.new_zeros(Bsz, 1, D)
            y = torch.cat([zero_col, a_rev[:, :-1, :]], dim=1)
            G_rev = _assoc_scan(y, grad_h_rev)
            g = G_rev.flip(1)
            grad_b = g
            h_prev = torch.cat([zero_col, h[:, :-1, :]], dim=1)
            grad_a = g * h_prev
        return grad_a, grad_b


def stable_diag_scan(a, b):
    return DiagScanFn.apply(a, b)


class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=4, d_conv=4, expand=2, dt_rank=None,
                 dt_min=1e-3, dt_max=1e-1, dt_init_floor=1e-4,
                 conv_bias=True, bias=False):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = expand * d_model
        self.dt_rank = max(1, math.ceil(d_model / 16)) if dt_rank is None else dt_rank

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=bias)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv,
                                 groups=self.d_inner, padding=0, bias=conv_bias)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        dt_init_std = self.dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)

    def _selective_scan(self, x, delta, A, B, C, D):
        Bsz, L, Din = x.shape
        N = A.shape[1]
        deltaA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        deltaB_u = delta.unsqueeze(-1) * B.unsqueeze(-2) * x.unsqueeze(-1)
        a_flat = deltaA.reshape(Bsz, L, Din * N)
        b_flat = deltaB_u.reshape(Bsz, L, Din * N)
        h_flat = stable_diag_scan(a_flat, b_flat)
        h = h_flat.view(Bsz, L, Din, N)
        y = torch.einsum('bldn,bln->bld', h, C)
        return y + D.view(1, 1, -1) * x

    def forward(self, x):
        Bsz, L, _ = x.shape
        xz = self.in_proj(x)
        x_branch, z = xz.chunk(2, dim=-1)
        xc = x_branch.transpose(1, 2)
        xc = F.pad(xc, (self.d_conv - 1, 0))
        xc = self.conv1d(xc)
        x_branch = xc.transpose(1, 2)
        x_branch = F.silu(x_branch)
        x_dbl = self.x_proj(x_branch)
        dt_raw, B_proj, C_proj = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt_raw))
        A = -torch.exp(self.A_log.float())
        y = self._selective_scan(x_branch, dt, A, B_proj, C_proj, self.D)
        y = y * F.silu(z)
        out = self.out_proj(y)
        return out


class MambaResidualLayer(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand, dt_rank, dropout, bias):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.block = MambaBlock(d_model=d_model, d_state=d_state, d_conv=d_conv,
                                  expand=expand, dt_rank=dt_rank, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.dropout(self.block(self.norm(x)))


class Mamba(nn.Module):
    """Mamba DPD backbone for OpenDPD — per-token output.

    Internal IQ augmentation (6 features) like DGRU.
    Input/output contract: (B, T, 2) → (B, T, 2)
    """

    def __init__(self,
                 d_model: int = 6,
                 d_state: int = 4,
                 d_conv: int = 4,
                 expand: int = 2,
                 num_layers: int = 1,
                 dropout: float = 0.0,
                 dt_rank: int = 0,
                 bias: bool = False,
                 **kwargs):
        super().__init__()
        self.input_size = 6
        self.output_size = 2
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.num_layers = num_layers
        self.dt_rank_arg = None if dt_rank <= 0 else dt_rank

        self.input_embedding = nn.Linear(self.input_size, d_model, bias=True)
        self.layers = nn.ModuleList([
            MambaResidualLayer(d_model=d_model, d_state=d_state, d_conv=d_conv,
                                 expand=expand, dt_rank=self.dt_rank_arg,
                                 dropout=dropout, bias=bias)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, self.output_size, bias=True)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.input_embedding.weight)
        if self.input_embedding.bias is not None:
            nn.init.zeros_(self.input_embedding.bias)
        nn.init.xavier_uniform_(self.fc_out.weight)
        if self.fc_out.bias is not None:
            nn.init.zeros_(self.fc_out.bias)
        # Layer-internal params (dt, A, D, conv) keep their custom init.

    @staticmethod
    def _iq_aug(x):
        i_x = x[..., 0:1]
        q_x = x[..., 1:2]
        eps = 1e-12
        amp = torch.sqrt(i_x.pow(2) + q_x.pow(2) + eps)
        amp3 = amp.pow(3)
        cos = i_x / amp
        sin = q_x / amp
        return torch.cat([i_x, q_x, amp, amp3, cos, sin], dim=-1)

    def forward(self, x, h_0=None):
        # h_0 unused (kept for interface compat)
        x = self._iq_aug(x)                # (B, T, 6)
        h = self.input_embedding(x)        # (B, T, d_model)
        for layer in self.layers:
            h = layer(h)
        h = self.final_norm(h)
        return self.fc_out(h)              # (B, T, 2)
