# spectral.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging 

def rfftfreq(n: int, d: float = 1.0, device=None):
    # PyTorch equivalent of numpy.fft.rfftfreq (length n, sample spacing d)
    freqs = torch.fft.rfftfreq(n, d=d)
    if device is not None:
        freqs = freqs.to(device)
    return freqs  # shape [n//2 + 1]

def _freq_scale(L: int, order: int, device=None, eps: float = 1e-6):
    """
    Bounded, invertible scale for (i*2*pi*f)^order:
      alpha = (w / w_ref)^order with w_ref = pi (Nyquist)
    This keeps |alpha| <= 1 (except at DC), avoiding huge values.
    """
    order = int(order)
    f = rfftfreq(L, d=1.0, device=device)                 # [F], float32
    w = (2 * math.pi * f).to(torch.float32)               # [F], angular freq
    w_ref = torch.tensor(math.pi, dtype=torch.float32, device=device)  # Nyquist
    # avoid exact 0 in both forward & inverse
    w_safe = torch.clamp(w / w_ref, min=eps)              # in (eps, 2], but rFFT max is pi ⇒ <=1
    alpha = w_safe ** order                               # [F], float32
    return alpha


def freq_derivative_multiplier(L: int, order: int, device=None, eps: float = 1e-6):
    order = int(order)
    f = rfftfreq(L, d=1.0, device=device)  # [F]
    f = torch.clamp(f, min=eps, max=1.0)  # Clamp frequencies to avoid extreme values
    w = 2 * math.pi * f  # angular freq
    one_r = torch.tensor(0.0, dtype=torch.float32, device=device)
    one_i = torch.tensor(1.0, dtype=torch.float32, device=device)
    i_unit = torch.complex(one_r, one_i)  # complex64
    i_factor = i_unit ** order
    w_power = torch.clamp(w ** order, min=-1e6, max=1e6)  # Clamp w^order before complex multiplication
    mult = i_factor * w_power  # complex64
    return mult


def apply_freq_derivative(X_cplx: torch.Tensor, order: int, L: int):
    device = X_cplx.device
    if not torch.isfinite(X_cplx).all():
        logging.warning("Non-finite values in X_cplx; clamping real and imaginary parts")
        X_cplx = torch.complex(
            torch.clamp(X_cplx.real, min=-1e6, max=1e6),
            torch.clamp(X_cplx.imag, min=-1e6, max=1e6)
        )
    mult = freq_derivative_multiplier(L=L, order=order, device=device)  # [F]
    return X_cplx * mult  # broadcast over [B, C, F]


def inverse_freq_derivative(Y_cplx: torch.Tensor, order: int, L: int, eps: float = 1e-6):
    """
    Invert using the SAME bounded scale. Clamp denom to avoid blow-ups at DC.
    """
    device = Y_cplx.device
    if not torch.isfinite(Y_cplx).all():
        logging.warning("Non-finite values in Y_cplx; clamping real and imaginary parts")
        Y_cplx = torch.complex(
            torch.clamp(Y_cplx.real, min=-1e6, max=1e6),
            torch.clamp(Y_cplx.imag, min=-1e6, max=1e6)
        )
    denom = freq_derivative_multiplier(L=L, order=order, device=device, eps=eps)  # [F]
    # Clamp denominator magnitude to prevent excessive amplification
    abs_d = torch.abs(denom)
    phase = denom / abs_d.clamp(min=1e-10)  # Normalize phase
    abs_clamp = abs_d.clamp(min=1e-4)  # Min magnitude to limit amplification (<= 1/1e-4 = 10000x)
    denom_clamp = phase * abs_clamp
    return Y_cplx / denom_clamp

class ComplexLinear(nn.Module):
    """
    Complex-valued 1x1 linear over channel dimension for rFFT tensors.
    Input/Output: complex tensor [B, C, F]
    Implemented as two real linears: (Re, Im) with shared weights.
    """
    def __init__(self, in_ch, out_ch, bias=True):
        super().__init__()
        self.wr = nn.Linear(in_ch, out_ch, bias=bias)
        self.wi = nn.Linear(in_ch, out_ch, bias=bias)

    def forward(self, z: torch.Tensor):
        # z: complex [B, C, F]
        zr, zi = z.real, z.imag  # [B, C, F]
        zr = zr.transpose(1, 2)  # [B, F, C]
        zi = zi.transpose(1, 2)
        yr = self.wr(zr) - self.wi(zi)  # [B, F, out_ch]
        yi = self.wr(zi) + self.wi(zr)
        yr = yr.transpose(1, 2)  # [B, out_ch, F]
        yi = yi.transpose(1, 2)
        return torch.complex(yr, yi)

class ComplexDepthwiseConvFreq(nn.Module):
    """
    Depthwise convolution along frequency bins (local mixing across nearby frequencies).
    Treats real/imag separately, then recombines.
    Input: complex [B, C, F]
    """
    def __init__(self, channels, kernel_size=5, padding="same"):
        super().__init__()
        self.dw_r = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size//2, groups=channels)
        self.dw_i = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size//2, groups=channels)

    def forward(self, z: torch.Tensor):
        zr, zi = z.real, z.imag  # [B,C,F]
        yr = self.dw_r(zr)
        yi = self.dw_i(zi)
        return torch.complex(yr, yi)

class BandMask(nn.Module):
    """
    Learnable band-pass mask per channel over frequency bins.
    Output in [0,1] via sigmoid; encourages sparse band selection with optional L1 on mask.
    """
    def __init__(self, channels, F):
        super().__init__()
        self.mask_logits = nn.Parameter(torch.zeros(1, channels, F))

    def forward(self, z: torch.Tensor):
        mask = torch.sigmoid(self.mask_logits)  # [1,C,F]
        return z * mask, mask
