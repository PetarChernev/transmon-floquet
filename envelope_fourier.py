import math
import torch


def omega_tilde_n(n, sigma, pulse_period, *,
                  device=None,
                  dtype=torch.complex128):
    """
    Return  \\tilde{Ω}^{(n)}  for a Gaussian envelope tiled with period T_F.

    Parameters
    ----------
    n            :  int or tensor[int]
    sigma        :  float | torch scalar      (Gaussian rms width)
    pulse_period :  float | torch scalar      (T_F  = Floquet / pulse period)

    Notes
    -----
    Uses the analytic closed form obtained by extending the integral
    to (−∞,∞) – valid when  σ ≪ T_F/2  so the tails are negligible.
    The result is *fully differentiable* w.r.t.  σ or T_F.
    """
    # cast inputs to tensors ---------------------------------------------------
    n      = torch.as_tensor(n,      dtype=torch.float64, device=device)
    sigma  = torch.as_tensor(sigma,  dtype=torch.float64, device=device)
    T_F    = torch.as_tensor(pulse_period, dtype=torch.float64, device=device)

    ω_F    = 2 * math.pi / T_F                       # Floquet angular freq.

    # √(2π) σ / T_F · exp[−½(σ ω_F n)²] · exp[−i n ω_F T_F/2]
    pref   = torch.sqrt(torch.tensor(2*math.pi, device=device)) * sigma / T_F
    gauss  = torch.exp(-0.5 * (sigma * ω_F * n)**2)
    phase  = torch.exp(-1j   * n * ω_F * T_F / 2)

    return (pref * gauss * phase).to(dtype)



def omega_tilde_fft(n, sigma, period, *,
                    n_grid=4096,
                    device=None,
                    dtype=torch.complex128):
    """
    Discrete-Fourier coefficient Ω̃⁽ⁿ⁾ of a period-T tiled Gaussian,
    evaluated via FFT (fully differentiable).

    Parameters
    ----------
    n       : int | 1-D tensor[int]     − harmonic(s) wanted
    sigma   : float | tensor            − Gaussian rms width
    period  : float | tensor            − T
    n_grid  : int                       − # of sample points (power of 2 → FFT)
    """
    # tensors ------------------------------------------------------------------
    sigma  = torch.as_tensor(sigma,  dtype=torch.float64, device=device)
    T      = torch.as_tensor(period, dtype=torch.float64, device=device)
    n      = torch.as_tensor(n,      dtype=torch.int64,   device=device)

    # time grid  ---------------------------------------------------------------
    t  = torch.linspace(0.0, 1.0, n_grid, device=device, dtype=torch.float64,
                        requires_grad=sigma.requires_grad or T.requires_grad)
    t  = t * T                                  # rescale to [0,T)

    # envelope samples ---------------------------------------------------------
    env = torch.exp(-0.5 * ((t - T/2) / sigma)**2)        # shape (n_grid,)

    # FFT  (normalised so that sum_k exp(−i2πkn/N) == N δ_n0 )
    coeffs = torch.fft.fft(env.to(dtype)) / n_grid        # shape (n_grid,)

    # harmonic index shift because torch.fft output is [0,1,2,…,N-1]
    k = (n % n_grid).to(torch.int64)
    return coeffs[k]



################################################################################
#  Ω̃⁽ᵏ;ʲ,ˡ,±⁾  – Fourier coefficient of the *full* transmon drive term (no RWA)
################################################################################
def tilde_omega_kjl(k, j, l, sign, sigma,
                    pulse_period, carrier_period,
                    phi=0.0,
                    *,
                    device=None,
                    dtype=torch.complex128):
    """
    Return  \\tilde{Ω}^{(k; j,l, ±)}  appearing in the Sambe blocks
    of the non-RWA Hamiltonian (Eq. A13).

    Parameters
    ----------
    k, j, l : int or tensor[int]
        Harmonic index k and level indices j,l.
    sign    : {+1,-1}
        +1 ↔ the  e^{+i(ω_d t + φ)}  piece,  -1 ↔ the  e^{−i(ω_d t + φ)}  piece.
    sigma        : float | torch scalar   (Gaussian width)
    pulse_period : float | torch scalar   (T_F)
    carrier_period : float | torch scalar (T_d = 2π/ω_d)
    phi          : float (carrier phase)

    The expression evaluated is

        (½) e^{-i sign φ} · (1/T_F)
        ∫₀^{T_F} Ω(t) · exp[i ((j-l+sign) ω_d − k ω_F) t] dt ,

    again using the analytic Gaussian integral approximation.
    """
    # tensors ------------------------------------------------------------------
    k  = torch.as_tensor(k,  dtype=torch.float64, device=device)
    j  = torch.as_tensor(j,  dtype=torch.float64, device=device)
    l  = torch.as_tensor(l,  dtype=torch.float64, device=device)
    s  = torch.as_tensor(sign, dtype=torch.float64, device=device)

    sigma = torch.as_tensor(sigma, dtype=torch.float64, device=device)
    T_F   = torch.as_tensor(pulse_period,    dtype=torch.float64, device=device)
    T_d   = torch.as_tensor(carrier_period,  dtype=torch.float64, device=device)
    phi   = torch.as_tensor(phi, dtype=torch.float64, device=device)

    ω_F   = 2 * math.pi / T_F
    ω_d   = 2 * math.pi / T_d

    κ     = (j - l + s) * ω_d - k * ω_F          # frequency mismatch
    pref  = 0.5 * torch.exp(-1j * s * phi)       # ½ · e^{-i sign φ}

    # same Gaussian integral as before, but with κ instead of n ω_F
    gauss_int = (torch.sqrt(torch.tensor(2*math.pi, device=device)) *
                 sigma / T_F *
                 torch.exp(-0.5 * (sigma * κ)**2) *
                 torch.exp(-1j * κ * T_F / 2))

    return (pref * gauss_int).to(dtype)

def omega_tilde_batch(
    k, j, l, sign,
    *,
    sigma,
    pulse_period,
    carrier_period,
    phi=0.0,
    rabi_max=1.0,
    device=None,
    dtype=torch.complex128,
):
    """
    Return  Omega_tilde^{(k ; j,l, sign)}  for a *single* sign.
    Shapes:
        k  :  (H,)      or scalar
        j,l:  (d,d)     meshgrid
    Output:
        coef : (d,d,H)  complex
    """
    k   = torch.as_tensor(k, dtype=torch.float64, device=device)
    j   = torch.as_tensor(j, dtype=torch.float64, device=device)
    l   = torch.as_tensor(l, dtype=torch.float64, device=device)
    sgn = float(sign)                                   # ensure Python scalar

    sigma = torch.as_tensor(sigma, dtype=torch.float64, device=device)
    T_F   = torch.as_tensor(pulse_period, dtype=torch.float64, device=device)
    T_d   = torch.as_tensor(carrier_period, dtype=torch.float64, device=device)
    phi   = torch.as_tensor(phi,  dtype=torch.float64, device=device)
    rabi  = torch.as_tensor(rabi_max, dtype=torch.float64, device=device)

    w_F, w_d = 2 * math.pi / T_F, 2 * math.pi / T_d
    kappa = (j - l + sgn)[..., None] * w_d - k * w_F        # (d,d,H)

    pref  = 0.5 * torch.exp(-1j * sgn * phi) * rabi

    g = torch.sqrt(torch.tensor(2 * math.pi, device=device)) * sigma / T_F
    coef = pref * g * torch.exp(-0.5 * (sigma * kappa) ** 2)
    coef = coef * torch.exp(-1j * kappa * T_F / 2)

    return coef.to(dtype)                                  # (d,d,H)