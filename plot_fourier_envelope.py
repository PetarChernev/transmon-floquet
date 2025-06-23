#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Reconstruct a Gaussian-modulated cosine from its Fourier coefficients
and plot the result over one Floquet period.
"""

import math
import torch
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
#  Fourier coefficients  of Gaussian × cos  (period 2π)
# ----------------------------------------------------------------------
def cos_gauss_coeffs(N, sigma, floquet_period, carrier_period,
                     dtype=torch.float64, device="cpu"):
    """
    Return Fourier coefficients  c_k  for the T-periodic extension of
        f(t) = exp(-t²/(2σ²)) · cos(2π t / carrier_period)
    where  T = floquet_period.
    """
    k = torch.arange(-N, N + 1, dtype=torch.float64, device=device)
    ω_f = 2 * math.pi / floquet_period         # Floquet base frequency
    ω_c = 2 * math.pi / carrier_period         # Carrier frequency

    arg1 = sigma * (k * ω_f - ω_c)
    arg2 = sigma * (k * ω_f + ω_c)

    g1 = torch.exp(-arg1 ** 2 / 2)
    g2 = torch.exp(-arg2 ** 2 / 2)
    scale = math.sqrt(2*math.pi)*sigma / floquet_period
    return scale * 0.5 * (g1 + g2).to(dtype)           # shape: (2 N+1,)


# ----------------------------------------------------------------------
#  Main demo
# ----------------------------------------------------------------------
def main():
    # ---------- user-adjustable parameters ----------
    N               = 50                 # number of positive harmonics
    sigma           = 0.30               # Gaussian width
    floquet_period  = 2                  # = 2π   (period of the series)
    cycles_in_T     = 30                 # carrier cycles per Floquet period
    carrier_period  = floquet_period / cycles_in_T
    device          = "cpu"              # "cuda" if you like
    # ------------------------------------------------

    # 1. Fourier coefficients (real, by definition)
    c_real = cos_gauss_coeffs(
        N, sigma, floquet_period, carrier_period,
        dtype=torch.float64, device=device
    )
    k = torch.arange(-N, N + 1, dtype=torch.float64, device=device)
    c = c_real.to(torch.complex128)          # promote to complex for synthesis

    # 2. Reconstruct the truncated Fourier series over one period
    t = torch.linspace(
        -floquet_period / 2,
        floquet_period / 2,
        20000,
        dtype=torch.float64,
        device=device,
    )
    
    ω_f = 2 * math.pi / floquet_period
    f_rec = torch.sum(c[None, :] * torch.exp(1j * k[None, :] * ω_f * t[:, None]),
                      dim=1).real.cpu()

    # (Optional) original aperiodic Gaussian-cosine for reference
    f_orig = (
        torch.exp(-t ** 2 / (2 * sigma ** 2))
        * torch.cos(2 * math.pi * t / carrier_period)
    ).cpu()

    # 3. Plot
    plt.figure()
    plt.scatter(t, f_rec,               label="Reconstructed periodic signal")
    plt.scatter(t, f_orig,        label="Single-shot envelope")
    plt.xlabel(r"time $t$")
    plt.ylabel(r"$f(t)$")
    plt.title("Gaussian-modulated cosine and its truncated Fourier reconstruction")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
