import numpy as np
import torch

from typing import Optional, Sequence, Union

from transmon.transmon_dynamics_qutip import pulse_sequence_qutip

def floquet_propagators_square_rabi(
    rabi_frequencies: torch.Tensor,
    phases: torch.Tensor,
    energies: torch.Tensor, 
    couplings: torch.Tensor,
    omega_d: torch.Tensor,
    floquet_cutoff: int
) -> torch.Tensor:
    """
    Builds the propagators for a sequence period of the cosine drive with frequency omega_d
    Parameters:
        rabi_frequencies: Tensor of Rabi frequencies for each pulse (shape (n,)).
        phases: Tensor of phases for each pulse (shape (n,)).
        energies: Tensor of energies of the system (shape (d,)).
        couplings: Tensor of couplings between states (shape (d, d)).
        omega_d: Drive frequency (scalar or tensor).
        floquet_cutoff: Fourier cutoff used in Floquet formalism M.
    Returns:
        A PyTorch tensor representing the total propagator after applying all pulses in sequence.
        (shape (n, d, d), where n is the number of pulses and d is the dimension of the physical Hilbert space).
    """
    H_Fs = build_floquet_hamiltonians(
        rabi_frequencies, 
        phases,
        energies, 
        couplings,
        omega_d,
        floquet_cutoff
    )
    return get_physical_propagators(H_Fs, floquet_cutoff, omega_d)


# --- small helper -----------------------------------------------------------------
def _batched_kron(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Kronecker product between a 2-D matrix A (S×S) and a
    *batch* of square matrices B (n×d×d).

    Returns
    -------
    Tensor with shape (n, S*d, S*d):
        kron(A, B[i]) for each i.
    """
    S = A.size(0)
    d = B.size(-1)
    # (n, S, S, d, d)
    K = (A.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
         * B.unsqueeze(1).unsqueeze(2))
    # reorder so the (row-block,row-intra, col-block,col-intra) axes are adjacent
    return K.permute(0, 1, 3, 2, 4).reshape(B.size(0), S * d, S * d)


# --- main routine -----------------------------------------------------------------
def build_floquet_hamiltonians(
    rabi_frequencies: torch.Tensor,    # (n,)
    phases:           torch.Tensor,    # (n,)
    energies:         torch.Tensor,    # (d,)
    couplings:        torch.Tensor,    # (d,d)
    omega_d:          float | torch.Tensor,
    M:                int,
) -> torch.Tensor:
    """
    Fully-vectorised construction of Floquet Hamiltonians for a *large*
    Fourier cut-off M, using only GPU tensor ops (no Python loops).

    Output: (n, (2*M+1)*d, (2*M+1)*d)
    """
    # --------------------------------------------------------------------------
    device  = energies.device
    d       = energies.numel()
    n_pulse = rabi_frequencies.numel()
    S       = 2 * M + 1                      # number of Fourier blocks
    N       = S * d                          # full Floquet dimension

    cdtype = couplings.dtype
    energies  = energies.to(cdtype)
    couplings = couplings.to(cdtype)

    # --------------------------------------------------------------------------
    # pulse-dependent block building bricks
    C0 = torch.diag(energies).to(cdtype)                 # (d,d) – same for every pulse

    rabi   = 0.5 * rabi_frequencies.to(energies.real.dtype).view(n_pulse, 1, 1)
    e_iphi = torch.exp(1j * phases.to(energies.real.dtype)).view(n_pulse, 1, 1)

    C1  = (rabi * couplings * e_iphi).to(cdtype)         # (n,d,d)
    Cm1 = (rabi * couplings * e_iphi.conj()).to(cdtype)  # (n,d,d)

    # --------------------------------------------------------------------------
    # matrices that act in the Fourier index space  (size S×S)
    Id_S     = torch.eye(S,  dtype=cdtype, device=device)                    # Iₛ
    diag_m   = torch.diag(torch.arange(-M, M + 1, device=device)).to(cdtype) # diag(m)
    sub_diag = torch.diag(torch.ones(S - 1, device=device, dtype=cdtype), -1)  # idx = +1
    sup_diag = torch.diag(torch.ones(S - 1, device=device, dtype=cdtype),  1)  # idx = −1

    # --------------------------------------------------------------------------
    # terms that are identical for all pulses
    H_static = (
        torch.kron(Id_S, C0) +
        torch.kron(diag_m * omega_d, torch.eye(d, dtype=cdtype, device=device))
    )                                   # (N,N)  – no pulse index yet

    # --------------------------------------------------------------------------
    # pulse-dependent super- and sub-diagonal terms (batched Kronecker products)
    H_sub   = _batched_kron(sub_diag, C1)   # (n,N,N)  idx = +1  (row > col)
    H_super = _batched_kron(sup_diag, Cm1)  # (n,N,N)  idx = −1

    # --------------------------------------------------------------------------
    # final assembly, broadcasting the static part across pulses
    H_F = H_sub + H_super                                   # (n,N,N)
    H_F += H_static                                         # broadcast add

    return H_F



def get_physical_propagators(
    H_F:           torch.Tensor,        # (..., N, N)
    floquet_cutoff: int,                #  M
    omega_d:       Union[float, torch.Tensor],
) -> torch.Tensor:
    """
    Compute the physical propagator U(T,0) from one **or a batch** of
    Floquet Hamiltonians – entirely on the GPU and without looping over m.

    Parameters
    ----------
    H_F : (..., N, N) complex tensor
        Floquet Hamiltonian(s); the leading dimensions (if any) are treated
        as batch dimensions.
    floquet_cutoff : int
        Fourier cut-off M used to build H_F  ⇒  N = (2 M+1) d.
    omega_d : float or (broadcastable) tensor
        Drive frequency ω_d (T = 2π / ω_d).

    Returns
    -------
    U_phys : (..., d, d) complex tensor
        Physical propagator(s) at one drive period T.
    """
    # ------------ sizes ----------------------------------------------------
    S = 2 * floquet_cutoff + 1               # number of Fourier blocks
    N = H_F.shape[-1]
    d = N // S                               # physical Hilbert-space dimension

    # ------------ Floquet evolution operator --------------------------------
    T_period = 2 * torch.pi / omega_d
    U_F = torch.matrix_exp(-1j * H_F * T_period)      # same batch shape as H_F

    # ------------ reshape into 5-D block structure --------------------------
    # (..., S_row, d_row, S_col, d_col)
    blocks = U_F.reshape(*U_F.shape[:-2], S, d, S, d)

    # ------------ pick column block m' = 0  (index M) -----------------------
    col0_blocks = blocks[..., :, :, floquet_cutoff, :]   # (..., S, d, d)

    # ------------ sum over the row-block index m ----------------------------
    U_phys = col0_blocks.sum(dim=-3)                     # (..., d, d)

    return U_phys


# --------------------------------------------------------------------------- #
# Helper: batched exponentiation-by-squaring for a *different* power per item #
# --------------------------------------------------------------------------- #
def _batch_matrix_power(mats: torch.Tensor, exps: torch.Tensor) -> torch.Tensor:
    n, d, _ = mats.shape
    result  = torch.eye(d, dtype=mats.dtype, device=mats.device)\
                  .expand(n, d, d).clone()
    base    = mats.clone()
    exp     = exps.clone()

    while torch.any(exp):
        odd_mask = (exp & 1).bool()
        if odd_mask.any():
            result[odd_mask] = torch.bmm(base[odd_mask], result[odd_mask])
        exp >>= 1
        if torch.any(exp):
            base = torch.bmm(base, base)
    return result
# ---------------------------------------------------------------------------------

def floquet_propagator_square_sequence_stroboscopic_vectorized(
    rabi_frequencies:        torch.Tensor,   # (n,)
    phases:                  torch.Tensor,   # (n,)
    pulse_durations_periods: torch.Tensor,   # (n,) ints
    energies:                torch.Tensor,   # (d,)
    couplings:               torch.Tensor,   # (d,d)
    omega_d:                 Union[float, torch.Tensor],
    floquet_cutoff:          int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Vectorised total propagator for a stroboscopic square-pulse sequence.
    Now handles the single-pulse case without the batched helpers.
    """
    if device is None:
        device = energies.device

    # ---- build the single-period propagator(s) ---------------------------------
    U_single = floquet_propagators_square_rabi(
        rabi_frequencies.to(device),
        phases.to(device),
        energies.to(device),
        couplings.to(device),
        omega_d,
        floquet_cutoff,
    )                                         # shape (n, d, d)

    n, d, _ = U_single.shape

    # ---- special-case: only one pulse ------------------------------------------
    if n == 1:
        return torch.matrix_power(U_single[0], int(pulse_durations_periods.item()))

    # ---- general batched case --------------------------------------------------
    U_powered = _batch_matrix_power(U_single, pulse_durations_periods)   # (n, d, d)

    # chain product in physical order: last pulse acts first
    total_U = torch.linalg.multi_dot(U_powered.flip(0).unbind(0))

    return total_U


def floquet_propagator_square_sequence_batch(
    rabi:                    torch.Tensor,   # (B, P)
    phases:                  torch.Tensor,   # (B, P)
    pulse_durations_periods: torch.Tensor,   # (P,)  – identical for all chains
    energies:                torch.Tensor,   # (d,)
    couplings:               torch.Tensor,   # (d,d)
    omega_d:                 Union[float, torch.Tensor],
    floquet_cutoff:          int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Vectorised total propagator U(T_total, 0) for a *batch* of B chains,
    each consisting of P square pulses that repeat for an integer number
    of drive periods.

    Parameters
    ----------
    rabi, phases : (B, P)  – per–chain continuous parameters
    pulse_durations_periods : (P,)  – **same** integer duration per pulse
    energies, couplings, omega_d, floquet_cutoff : as before
    device : optional override; defaults to energies.device

    Returns
    -------
    U_total : (B, d, d)  – physical propagator of every chain
    """
    # ----------------------- housekeeping ----------------------------------
    if device is None:
        device = energies.device


    B, P = rabi.shape
    d    = energies.numel()

    # ----------------------- single-period propagators ---------------------
    # Flatten (B,P) → (B*P,) and build all Floquet propagators in one call
    U_single_flat = floquet_propagators_square_rabi(
        rabi.reshape(-1),
        phases.reshape(-1),
        energies,
        couplings,
        omega_d,
        floquet_cutoff,
    )                                           # (B*P, d, d)

    # ----------------------- raise to integer powers -----------------------
    # Repeat durations for every chain, flatten again for the helper
    exps_flat = pulse_durations_periods.expand(B, P).reshape(-1)           # (B*P,)

    U_powered_flat = _batch_matrix_power(                    # (B*P, d, d)
        U_single_flat, exps_flat
    )
    
    # ----------------------- chain multiplication --------------------------
    if P == 1:
        return U_powered_flat.view(B, d, d)     # (B, d, d)

    # reverse pulse order: last pulse acts first
    U_rev = U_powered_flat.reshape(B, P, d, d).flip(1)   # (B, P, d, d)

    total_U = torch.eye(d, dtype=U_rev.dtype, device=device) \
                  .expand(B, d, d).clone()

    for p in range(P):            # multiply **on the right**
        total_U = torch.bmm(total_U, U_rev[:, p])

    return total_U



# --------------------------------------------------------------------------- #
#  new: propagator for a *batch* of ωd values                                 #
# --------------------------------------------------------------------------- #
def floquet_propagator_square_sequence_omega_batch(
    rabi:                    torch.Tensor,   # (P,)
    phases:                  torch.Tensor,   # (P,)
    pulse_durations_periods: torch.Tensor,   # (P,)  ints  (identical for all chains)
    energies:                torch.Tensor,   # (d,)
    couplings:               torch.Tensor,   # (d,d)
    omega_d_batch:           torch.Tensor,   # (B,)   – one ω_d per chain
    floquet_cutoff:          int,
) -> torch.Tensor:
    """
    Physical propagator U(T_total,0) for a batch of B chains that share the
    same pulse sequence but have different drive frequencies ω_d.

    Returns
    -------
    U_total : (B, d, d) complex tensor
    """
    # ---------------- housekeeping -----------------------------------------
    device = rabi.device


    B = omega_d_batch.numel()   # batch size
    P = rabi.numel()
    d = energies.numel()
    S = 2 * floquet_cutoff + 1
    N = S * d

    cdtype = couplings.dtype

    # ---------------- pulse-dependent off-diagonal blocks ------------------
    # These do **not** depend on ω_d, so we build them once for all chains.
    #   C₀, C_{±1}
    C0 = torch.diag(energies.to(cdtype))                       # (d,d)

    rabi_p  = (0.5 * rabi).view(P, 1, 1)                      # (P,1,1)
    eip_phi = torch.exp(1j * phases).view(P, 1, 1)
    C1  = (rabi_p * couplings * eip_phi       ).to(cdtype)     # (P,d,d)
    Cm1 = (rabi_p * couplings * eip_phi.conj()).to(cdtype)     # (P,d,d)

    # Fourier-space helpers
    Id_S     = torch.eye(S,  dtype=cdtype, device=device)               # I_S
    diag_m   = torch.diag(torch.arange(-floquet_cutoff,
                                       floquet_cutoff + 1,
                                       device=device)).to(cdtype)       # diag(m)
    sub_diag = torch.diag(torch.ones(S - 1, device=device,
                                     dtype=cdtype), -1)                 # L_S
    sup_diag = torch.diag(torch.ones(S - 1, device=device,
                                     dtype=cdtype),  1)                 # U_S
    eye_d    = torch.eye(d, dtype=cdtype, device=device)

    # common static parts (ω-independent)
    H_static_common = torch.kron(Id_S, C0)                              # (N,N)
    H_shift_base    = torch.kron(diag_m, eye_d)                         # (N,N)

    # off-diagonal pulse blocks  (P,N,N)
    H_sub   = _batched_kron(sub_diag, C1)   # idx = +1
    H_super = _batched_kron(sup_diag, Cm1)  # idx = −1
    H_off   = H_sub + H_super               # (P,N,N)

    # ---------------- assemble Floquet Hamiltonians ------------------------
    # Broadcast shapes:
    #   H_off           -> (1, P, N, N)
    #   H_static_common -> (1, 1, N, N)
    #   H_shift_base    -> (1, 1, N, N)
    #   omega_d_batch   -> (B, 1, 1, 1)
    H_F = (
        H_off.unsqueeze(0) +
        H_static_common +
        omega_d_batch.view(B, 1, 1, 1) * H_shift_base
    )                                     # (B, P, N, N)

    # ---------------- single-period physical propagators -------------------
    U_single = get_physical_propagators(
        H_F, floquet_cutoff,
        omega_d_batch.view(B, 1, 1, 1)     # broadcast into get_physical_propagators
    )                                      # (B, P, d, d)

    # ---------------- raise to integer powers ------------------------------
    U_single_flat = U_single.reshape(B * P, d, d)          # (B*P, d, d)
    exps_flat     = pulse_durations_periods.expand(B, P).reshape(-1)     # (B*P,)

    U_powered_flat = _batch_matrix_power(U_single_flat, exps_flat)
    U_powered = U_powered_flat.view(B, P, d, d)            # (B, P, d, d)

    # ---------------- chain multiplication (right-to-left) -----------------
    if P == 1:
        return U_powered.squeeze(1)                        # (B, d, d)

    U_rev = U_powered.flip(1)                              # last pulse acts first
    total_U = torch.eye(d, dtype=cdtype, device=device).expand(B, d, d).clone()

    for p in range(P):
        total_U = torch.bmm(total_U, U_rev[:, p])

    return total_U      