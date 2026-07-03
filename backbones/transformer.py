__author__ = "Minho Choi"
__license__ = "Apache-2.0 License"

"""Tiny Transformer-encoder DPD backbone with several inductive-bias variants.

Mirrors the DGRU input/output contract exactly:
  - Input  : (B, T, 2)  raw I/Q
  - Output : (B, T, 2)  per-timestep I/Q prediction
  - Internal IQ augmentation: (I, Q, |x|, |x|^3, cos, sin) → 6 features

Flags:
  use_pos_encoding         — adds sinusoidal positional encoding
  output_residual_concat   — concat IQ-augmented features to encoder output before fc_out
                              (DGRU-style residual that injects input into output layer)
  input_mlp_hidden         — if > 0, use 2-layer MLP (6 → h → d_model) for input embedding
  output_mlp_hidden        — if > 0, use 2-layer MLP (d_model → h → 2) for output projection
"""

import math
import torch
import torch.nn as nn


class _SinusoidalPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        self.d_model = d_model
        pe = self._make(max_len, d_model)
        self.register_buffer('pe', pe.unsqueeze(0))

    @staticmethod
    def _make(L: int, d: int) -> torch.Tensor:
        pe = torch.zeros(L, d)
        pos = torch.arange(0, L, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2, dtype=torch.float32) * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        if d % 2 == 0:
            pe[:, 1::2] = torch.cos(pos * div)
        else:
            pe[:, 1::2] = torch.cos(pos * div[:-1])
        return pe

    def forward(self, x):
        T = x.size(1)
        if T > self.pe.size(1):
            new_pe = self._make(((T // 1000) + 1) * 1000, self.d_model).unsqueeze(0).to(x.device)
            self.pe = new_pe
        return x + self.pe[:, :T, :]


class _SwiGLU(nn.Module):
    """SwiGLU FFN: y = W3( SiLU(W1·x) ⊙ (W2·x) )."""
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_model, d_ff)
        self.w3 = nn.Linear(d_ff, d_model)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.w3(self.act(self.w1(x)) * self.w2(x))


class _EncoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0,
                 local_window: int = 0, ffn_type: str = 'mlp'):
        super().__init__()
        self.local_window = local_window
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads,
                                           dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn_type = ffn_type
        if ffn_type == 'swiglu':
            self.ffn = _SwiGLU(d_model, d_ff)
        else:
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Linear(d_ff, d_model),
            )

    def _build_local_mask(self, T: int, device) -> torch.Tensor:
        """Banded mask: position i attends to j iff |i-j| <= window/2."""
        w = self.local_window // 2
        idx = torch.arange(T, device=device)
        d = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
        mask = torch.where(d <= w, torch.zeros_like(d, dtype=torch.float32),
                           torch.full_like(d, float('-inf'), dtype=torch.float32))
        return mask

    def forward(self, x):
        h = self.norm1(x)
        attn_mask = None
        if self.local_window > 0:
            attn_mask = self._build_local_mask(h.size(1), h.device)
        a, _ = self.attn(h, h, h, need_weights=False, attn_mask=attn_mask)
        x = x + a
        x = x + self.ffn(self.norm2(x))
        return x


class _GMPStem(nn.Module):
    """GMP-natural multiplicative cross-position stem.

    Two Conv1d branches:
      - signal branch on raw (I, Q): captures `Σ w_l · x[i+l]`
      - envelope branch on (|x|, |x|^3): captures `Σ w_lk · |x[i+l]|^k`
    Element-wise product → GMP-like cross-position multiplicative features.

    This is the NN-equivalent of GMP's c_klm · x(n-l) · |x(n-l+m)|^k cross term:
    the conv kernel size enables interactions across positions, and the Hadamard
    product makes the result a true multiplicative cross-term (not a linear sum).
    """
    def __init__(self, d_model: int, kernel_size: int = 5, env_powers=(1, 3)):
        super().__init__()
        self.env_powers = tuple(env_powers)
        pad = kernel_size // 2
        self.conv_signal = nn.Conv1d(2, d_model, kernel_size=kernel_size, padding=pad, bias=False)
        self.conv_env = nn.Conv1d(len(env_powers), d_model, kernel_size=kernel_size, padding=pad, bias=False)
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        # x: (B, T, 2) raw I/Q
        i_x = x[..., 0:1]; q_x = x[..., 1:2]
        amp = torch.sqrt(i_x.pow(2) + q_x.pow(2) + 1e-12)  # (B, T, 1)
        envs = [amp.pow(k) for k in self.env_powers]
        env = torch.cat(envs, dim=-1)                       # (B, T, K)
        sig_t = self.conv_signal(x.transpose(1, 2))         # (B, d, T)
        env_t = self.conv_env(env.transpose(1, 2))          # (B, d, T)
        h = (sig_t * env_t).transpose(1, 2) + self.bias     # (B, T, d) — multiplicative cross-position
        return h


class Transformer(nn.Module):
    def __init__(self, d_model: int = 6, n_heads: int = 2, d_ff: int = 18,
                 num_layers: int = 1, dropout: float = 0.0, bias: bool = True,
                 use_pos_encoding: bool = False,
                 output_residual_concat: bool = False,
                 input_mlp_hidden: int = 0,
                 output_mlp_hidden: int = 0,
                 conv_stem_kernel: int = 0,
                 local_attn_window: int = 0,
                 ffn_type: str = 'mlp',
                 use_gmp_stem: bool = False,
                 gmp_stem_kernel: int = 5,
                 **kwargs):
        super().__init__()
        self.input_size = 6      # I, Q, amp, amp^3, cos, sin (matches DGRU)
        self.output_size = 2
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.use_pos_encoding = use_pos_encoding
        self.output_residual_concat = output_residual_concat
        self.input_mlp_hidden = input_mlp_hidden
        self.output_mlp_hidden = output_mlp_hidden
        self.conv_stem_kernel = conv_stem_kernel
        self.local_attn_window = local_attn_window
        self.ffn_type = ffn_type
        self.use_gmp_stem = use_gmp_stem
        self.gmp_stem_kernel = gmp_stem_kernel

        # ---- Input embedding (optionally MLP / GMP-stem) ----
        if use_gmp_stem:
            self.input_embedding = _GMPStem(d_model, kernel_size=gmp_stem_kernel)
        elif input_mlp_hidden > 0:
            self.input_embedding = nn.Sequential(
                nn.Linear(self.input_size, input_mlp_hidden, bias=bias),
                nn.GELU(),
                nn.Linear(input_mlp_hidden, d_model, bias=bias),
            )
        else:
            self.input_embedding = nn.Linear(self.input_size, d_model, bias=bias)

        self.pos_encoding = _SinusoidalPE(d_model) if use_pos_encoding else None

        # Optional Conv1d stem (local sample-level context before global attention)
        if conv_stem_kernel > 0:
            pad = conv_stem_kernel // 2
            self.conv_stem = nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size=conv_stem_kernel, padding=pad, bias=bias),
                nn.GELU(),
            )
        else:
            self.conv_stem = None

        self.layers = nn.ModuleList([
            _EncoderBlock(d_model, n_heads, d_ff, dropout, local_window=local_attn_window,
                          ffn_type=ffn_type)
            for _ in range(num_layers)
        ])

        # ---- Output projection ----
        # If output_residual_concat, encoder output is concatenated with the
        # augmented input features before the output projection (DGRU-style).
        out_in_dim = d_model + (self.input_size if output_residual_concat else 0)
        if output_mlp_hidden > 0:
            self.fc_out = nn.Sequential(
                nn.Linear(out_in_dim, output_mlp_hidden, bias=bias),
                nn.GELU(),
                nn.Linear(output_mlp_hidden, self.output_size, bias=bias),
            )
        else:
            self.fc_out = nn.Linear(out_in_dim, self.output_size, bias=bias)

    def reset_parameters(self):
        # input embedding
        if isinstance(self.input_embedding, _GMPStem):
            nn.init.xavier_uniform_(self.input_embedding.conv_signal.weight)
            nn.init.xavier_uniform_(self.input_embedding.conv_env.weight)
            nn.init.zeros_(self.input_embedding.bias)
        elif isinstance(self.input_embedding, nn.Sequential):
            for m in self.input_embedding:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        else:
            nn.init.xavier_uniform_(self.input_embedding.weight)
            if self.input_embedding.bias is not None:
                nn.init.zeros_(self.input_embedding.bias)
        # conv stem
        if self.conv_stem is not None:
            for m in self.conv_stem:
                if isinstance(m, nn.Conv1d):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        # encoder blocks
        for blk in self.layers:
            for name, p in blk.attn.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(p)
                elif 'bias' in name:
                    nn.init.zeros_(p)
            if isinstance(blk.ffn, _SwiGLU):
                for m in (blk.ffn.w1, blk.ffn.w2, blk.ffn.w3):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            else:
                for m in blk.ffn:
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_uniform_(m.weight)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
        # fc_out
        if isinstance(self.fc_out, nn.Sequential):
            for m in self.fc_out:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        else:
            nn.init.xavier_uniform_(self.fc_out.weight)
            if self.fc_out.bias is not None:
                nn.init.zeros_(self.fc_out.bias)

    @staticmethod
    def _iq_aug(x: torch.Tensor) -> torch.Tensor:
        """raw (B, T, 2) → (B, T, 6) with amp, amp^3, cos, sin."""
        i_x = x[..., 0:1]
        q_x = x[..., 1:2]
        eps = 1e-12
        amp = torch.sqrt(i_x.pow(2) + q_x.pow(2) + eps)
        amp3 = amp.pow(3)
        cos = i_x / amp
        sin = q_x / amp
        return torch.cat([i_x, q_x, amp, amp3, cos, sin], dim=-1)

    def forward(self, x, h_0=None):
        # h_0 ignored — kept for interface compatibility with RNN backbones
        x_aug = self._iq_aug(x)                  # (B, T, 6)
        if self.use_gmp_stem:
            # GMP stem consumes raw (I, Q) and computes its own envelope features internally
            h_in = self.input_embedding(x[..., :2])
        else:
            h_in = self.input_embedding(x_aug)
        x = h_in                                 # (B, T, d_model)
        if self.pos_encoding is not None:
            x = self.pos_encoding(x)
        if self.conv_stem is not None:
            # Conv1d wants (B, C, T)
            x = self.conv_stem(x.transpose(1, 2)).transpose(1, 2)
        for blk in self.layers:
            x = blk(x)                           # (B, T, d_model)
        if self.output_residual_concat:
            x = torch.cat([x, x_aug], dim=-1)    # (B, T, d_model + 6)
        x = self.fc_out(x)                       # (B, T, 2) — per-timestep
        return x
