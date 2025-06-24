#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Reconstruct a Gaussian-modulated cosine from its Fourier coefficients
and plot the result over one Floquet period.
"""

import math
import torch
import matplotlib.pyplot as plt

from envelope_fourier import omega_tilde_n
from transmon_floquet_propagator import omega_tilde_fft


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
    N               = 20                # number of positive harmonics
    sigma           = 1/6               # Gaussian width
    floquet_period  = 2                  # = 2π   (period of the series)
    device          = "cpu"              # "cuda" if you like
    # ------------------------------------------------

    k = torch.arange(-N, N + 1, dtype=torch.float64, device=device)
    
    t = torch.linspace(
        0,
        floquet_period,
        20000,
        dtype=torch.float64,
        device=device,
    )
    
    c_real_analytic = omega_tilde_n(
        k, sigma, floquet_period
    )
    c_analytic = c_real_analytic.to(torch.complex128)           

    c_real_fft = omega_tilde_fft(k, sigma, floquet_period,)
    c_fft = c_real_fft.to(torch.complex128)          

    ω_f = 2 * math.pi / floquet_period
    f_analytic = torch.sum(c_analytic * torch.exp(1j * k[None, :] * ω_f * t[:, None]),
                      dim=1).real.cpu()

    f_fft = torch.sum(c_fft * torch.exp(1j * k[None, :] * ω_f * t[:, None]),
                      dim=1).real.cpu()
    # 3. Plot
    plt.figure()
    plt.plot(t, f_analytic,               label="Analytic")
    plt.plot(t, f_fft,               label="FFT")
    plt.xlabel(r"time $t$")
    plt.ylabel(r"$f(t)$")
    plt.title("Gaussian-modulated cosine and its truncated Fourier reconstruction")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
