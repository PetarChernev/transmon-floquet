import numpy as np
import torch
import math

from transmon_core import TransmonCore


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

# =====================================================================
#  1.  H^(0)  – choose RWA or full
# =====================================================================
def H0_block(
    mu, Lambda,
    *,
    sigma,
    pulse_period,
    carrier_period,
    phi=0.0,
    rabi_max=1.0,
    rwa=True,                  # <-- master switch
    device=None,
    dtype=torch.complex128,
):
    """
    Build  H^(0)  (static Floquet block).
      mu      : (d,)   diagonal energies  (already e_j - j*omega_d)
      Lambda  : (d,d)  coupling matrix    lambda_{j,l}
    """
    mu  = torch.as_tensor(mu).to(dtype).to(device)
    Lam = torch.as_tensor(Lambda).to(dtype).to(device)
    d   = mu.numel()

    H0 = torch.diag(mu.clone())             # static diagonal

    # drive term(s) at k = 0
    j, l = torch.meshgrid(
        torch.arange(d, device=device),
        torch.arange(d, device=device),
        indexing="ij",
    )

    # co-rotating ALWAYS included
    H_co = omega_tilde_batch(
        0, j, l, -1,
        sigma=sigma,
        pulse_period=pulse_period,
        carrier_period=carrier_period,
        phi=phi,
        rabi_max=rabi_max,
        device=device,
        dtype=dtype,
    ).squeeze(-1)

    H0 = H0 + Lam * H_co

    if not rwa:
        # add counter-rotating only when NOT in RWA
        H_ctr = omega_tilde_batch(
            0, j, l, +1,
            sigma=sigma,
            pulse_period=pulse_period,
            carrier_period=carrier_period,
            phi=phi,
            rabi_max=rabi_max,
            device=device,
            dtype=dtype,
        ).squeeze(-1)
        H0 = H0 + Lam * H_ctr

    # numerical Hermiticity
    H0 = 0.5 * (H0 + H0.conj().T)
    return H0


# =====================================================================
#  2.  H^(k)  for k != 0  – choose RWA or full
# =====================================================================
def Hk_block(
    k, Lambda,
    *,
    sigma,
    pulse_period,
    carrier_period,
    phi=0.0,
    rabi_max=1.0,
    rwa=True,
    device=None,
    dtype=torch.complex128,
):
    """
    Build  H^(k)  (k != 0).  Raises if k == 0.
    """
    if k == 0:
        raise ValueError("Use H0_block for k = 0")

    Lam = torch.as_tensor(Lambda).to(dtype).to(device)
    d   = Lam.shape[0]

    j, l = torch.meshgrid(
        torch.arange(d, device=device),
        torch.arange(d, device=device),
        indexing="ij",
    )

    # always include co-rotating
    Hk = Lam * omega_tilde_batch(
        k, j, l, -1,
        sigma=sigma,
        pulse_period=pulse_period,
        carrier_period=carrier_period,
        phi=phi,
        rabi_max=rabi_max,
        device=device,
        dtype=dtype,
    ).squeeze(-1)

    if not rwa:
        Hk = Hk + Lam * omega_tilde_batch(
            k, j, l, +1,
            sigma=sigma,
            pulse_period=pulse_period,
            carrier_period=carrier_period,
            phi=phi,
            rabi_max=rabi_max,
            device=device,
            dtype=dtype,
        ).squeeze(-1)

    return Hk



def build_sambe(mu, Lambda, L, *,
                sigma, pulse_period, carrier_period,
                phi=0.0, rabi_max=1.0, rwa=True,
                device=None, dtype=torch.complex128):
    mu = torch.as_tensor(mu, dtype=dtype, device=device)
    d = mu.numel()
    Q = torch.zeros(((2*L+1)*d,)*2, dtype=dtype, device=device)

    # diagonal blocks
    H0 = H0_block(mu, Lambda,
                  sigma=sigma, pulse_period=pulse_period,
                  carrier_period=carrier_period,
                  phi=phi, rabi_max=rabi_max, rwa=rwa,
                  device=device, dtype=dtype)
    wF = 2 * math.pi / pulse_period
    for m in range(-L, L+1):
        idx = slice((m+L)*d, (m+L+1)*d)
        Q[idx, idx] = H0 + torch.eye(d, device=device, dtype=dtype) * m * wF

    # off-diagonal blocks
    for k in range(-L, L+1):
        if k == 0:
            continue
        Hk = Hk_block(k, Lambda,
                      sigma=sigma, pulse_period=pulse_period,
                      carrier_period=carrier_period,
                      phi=phi, rabi_max=rabi_max, rwa=rwa,
                      device=device, dtype=dtype)
        for m in range(-L, L+1):
            n = m - k
            if -L <= n <= L:
                Q[(m+L)*d:(m+L+1)*d, (n+L)*d:(n+L+1)*d] = Hk
    return Q



# -------------------------------------------------------------------------
#  RE-USE THE BLOCK BUILDERS AND build_sambe FROM EARLIER
#    – make sure they are imported / defined above this cell
# -------------------------------------------------------------------------

def single_pulse_propagator(
    n_levels,            # number of transmon levels  (e.g. 2 for a qubit)
    rabi_max,            # peak Rabi frequency  (units: qubit angular freq)
    phi,                 # carrier phase  (units of pi, e.g. 0.5 => pi/2)
    detuning_ratio,      # omega_d / omega_qubit
    duration,            # pulse length  (time units: 1 / omega_qubit)
    anharmonicity,
    *,
    sigma_ratio=0.10,    # Gaussian width  sigma = sigma_ratio * duration
    L=10,                # Floquet harmonic cut-off
    rwa=True,            # True => RWA, False => keep counter-rotating
    device="cpu",
    dtype=torch.complex128,
):
    """
    Return the physical-space propagator U(T) for ONE tiled-Gaussian pulse.

    A quick units convention:
        - omega_qubit = 1  (angular units)
        - Time is measured in 1 / omega_qubit.
        - rabi_max is dimensionless (fraction of omega_qubit).

    Raises
    ------
    ValueError if `duration` does not contain an integer number of
    carrier periods  (within 1e-10 relative tolerance).
    """

    # ---------------------------------------------------------------
    # 0.  Validate carrier commensurability
    # ---------------------------------------------------------------
    carrier_period = 1.0 / detuning_ratio            # T_d  (see units note)
    n_cycles = duration / carrier_period
    if abs(n_cycles - round(n_cycles)) > 1e-10:
        raise ValueError(
            "duration does not contain an integer number "
            "of carrier periods:  n_cycles = {:.12f}".format(n_cycles)
        )
    n_cycles = int(round(n_cycles))

    # ---------------------------------------------------------------
    # 1.  Basic transmon / qubit model   (2-level by default)
    # ---------------------------------------------------------------
    EJ_EC_ratio = TransmonCore.find_EJ_EC_for_anharmonicity(anharmonicity)
    mu, Lambda = TransmonCore.compute_transmon_parameters(n_levels, n_charge=30, EJ_EC_ratio=EJ_EC_ratio)             # Hermitian conjugate

    # pulse parameters
    sigma = sigma_ratio * duration
    phi_rad = phi * math.pi

    # ---------------------------------------------------------------
    # 2.  Build the Sambe matrix Q   (size  (2L+1)*d  )
    # ---------------------------------------------------------------
    Q = build_sambe(
        mu,
        Lambda,
        L,
        sigma=sigma,
        pulse_period=duration,
        carrier_period=carrier_period,
        phi=phi_rad,
        rabi_max=rabi_max,
        rwa=rwa,
        device=device,
        dtype=dtype,
    )

    # ---------------------------------------------------------------
    # 3.  One-period propagator in the extended space
    # ---------------------------------------------------------------
    U_big = torch.linalg.matrix_exp(-1j * Q * duration)

    # ---------------------------------------------------------------
    # 4.  Project onto the physical subspace  (m = 0   Fourier block)
    # ---------------------------------------------------------------
    mu = torch.as_tensor(mu, dtype=dtype, device=device)

    d = mu.numel()
    idx = slice(L * d, (L + 1) * d)    # m = 0 row/col block
    U_phys = U_big[idx, idx]

    return U_phys        # shape:  (d, d)   (complex, differentiable)


def pulse_sequence_propagator(
    n_levels,
    rabi_list,          # list or 1-D tensor of peak Rabi rates (dimensionless)
    phase_list,         # list or 1-D tensor of phases   (units of pi)
    detuning_ratio,     # omega_d / omega_qubit
    duration,           # pulse length (same for every pulse, 1 / omega_qubit)
    anharmonicity,
    *,
    sigma_ratio = 0.10,
    L           = 10,
    rwa         = True,
    device      = "cpu",
    dtype       = torch.complex128,
):
    """
    Return the total propagator for a *time-ordered* sequence of pulses.

        U_total = U_N  @ ... @ U_2  @ U_1

    The i-th pulse uses  rabi_max = rabi_list[i]  and  phase = phase_list[i]*pi.

    Parameters
    ----------
    n_levels        : int          – number of transmon levels kept
    rabi_list       : list/tuple/tensor, length N
    phase_list      : list/tuple/tensor, length N  (each element in units of pi)
    detuning_ratio  : float        – omega_d / omega_qubit
    duration        : float        – common pulse duration
    anharmonicity   : float        – target transmon anharmonicity
    sigma_ratio     : float        – sigma = sigma_ratio * duration
    L               : int          – Floquet harmonic cut-off
    rwa             : bool         – True = RWA, False = full non-RWA
    device, dtype   : torch options (for autograd / CUDA, etc.)

    Returns
    -------
    U_total : (n_levels, n_levels) complex tensor
    """

    # ------------- sanity checks ---------------------------------------------
    if len(rabi_list) != len(phase_list):
        raise ValueError("rabi_list and phase_list must have the same length")

    # convert to tensors so that autograd can track them if needed
    rabi_list  = torch.as_tensor(rabi_list,  dtype=torch.float64, device=device)
    phase_list = torch.as_tensor(phase_list, dtype=torch.float64, device=device)

    # start with identity in the physical Hilbert space
    U_total = torch.eye(n_levels, dtype=dtype, device=device)

    # loop over pulses in chronological order
    for rabi_max, phase in zip(rabi_list, phase_list):
        U_pulse = single_pulse_propagator(
            n_levels          = n_levels,
            rabi_max          = rabi_max,
            phi               = phase,          # still in units of pi
            detuning_ratio    = detuning_ratio,
            duration          = duration,
            anharmonicity     = anharmonicity,
            sigma_ratio       = sigma_ratio,
            L                 = L,
            rwa               = rwa,
            device            = device,
            dtype             = dtype,
        )
        # state_{next} = U_pulse @ state_now
        U_total = U_pulse @ U_total

    return U_total


if __name__ == "__main__":
    dev, dtype = "cuda", torch.complex128
    n_levels = 6

    # From Table I - complete population transfer
    rabi_frequencies = np.array(
        [42.497, 69.996, 69.996, 69.761, 63.782, 69.996, 58.263]
    )

    rabi_frequencies = (
        rabi_frequencies / 7000
    )  # Convert to GHz from MHz and normalise by ω01

    phases = np.array(
        [-0.3875, 0.0188, 0.0191, 0.1258, 0.2469, 0.3139, 0.2516]
    ) * np.pi

    # Total time T = 20 ns
    total_time = 20.0 * 2 * np.pi * 7  # Convert to dimensionless units
    delta = -0.0429
    EJ_EC_ratio = TransmonCore.find_EJ_EC_for_anharmonicity(delta)
    
    seq = pulse_sequence_propagator(
        n_levels=n_levels,
        rabi_list=rabi_frequencies,
        phase_list=phases,
        L=50,
        sigma_ratio=1 / 6,
        anharmonicity=delta,
        detuning_ratio=1.0,         # resonant
        duration=math.ceil(total_time / len(rabi_frequencies)),
        device=dev,
        dtype=dtype,
    )
    
    print(seq)