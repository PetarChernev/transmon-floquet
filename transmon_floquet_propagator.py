import math
import torch
import numpy as np

from transmon_core import TransmonCore

def compute_fourier_coeffs(rabi: float,
                           phase: float,
                           couplings: torch.Tensor,
                           fourier_cutoff: int) -> dict:
    """
    Compute the Fourier coefficient matrices C^{(n)} for the drive Hamiltonian H1(t),
    including static (n=0) drive pieces and enforcing Hermiticity.
    """
    d = couplings.shape[0]
    M = fourier_cutoff
    # Initialize coefficients for n in [-M..M]
    C = {n: torch.zeros((d, d), dtype=torch.cfloat) for n in range(-M, M+1)}

    half_rabi = 0.5 * rabi
    exp_pos = torch.exp(1j * phase)
    exp_neg = torch.exp(-1j * phase)

    for j in range(d):
        for l in range(d):
            lam_fwd = couplings[j, l]        #  λ_{jl}
            lam_bwd = couplings[l, j]        #  λ_{lj}  ( = λ_{jl}* for a transmon)

            if lam_fwd == 0 and lam_bwd == 0:
                continue

            Δ = j - l                       # level difference
            # ---------- forward part  λ_{jl} ----------------------------------
            for offset, phase_factor in ((+1, exp_pos), (-1, exp_neg)):
                n = Δ + offset
                if -M <= n <= M:
                    C[n][j, l] += half_rabi * lam_fwd * phase_factor

            # ---------- backward part λ_{lj} ----------------------------------
            for offset, phase_factor in ((+1, exp_pos), (-1, exp_neg)):
                n = -Δ + offset             # NOTE the minus sign!
                if -M <= n <= M:
                    C[n][j, l] += half_rabi * lam_bwd * phase_factor
    # Enforce Hermiticity
    # for n in range(1, M+1):
    #     C[-n] = C[n].conj().T
    return C


def floquet_propagator_square_rabi_one_period(fourier_coeffs: dict,
                                          energies: torch.Tensor,
                                          omega_d: float,
                                          fourier_cutoff: int) -> torch.Tensor:
    """
    Build truncated Floquet Hamiltonian in the rotating frame and compute one-period propagator.
    """
    d = energies.numel()
    M = fourier_cutoff
    # Prepare zero block
    zero_block = torch.zeros((d, d), dtype=torch.cfloat)
    # Collect all Fourier blocks (drive + H0 later)
    C_total = {n: fourier_coeffs.get(n, zero_block.clone())
               for n in range(-M, M+1)}

    # Static H0 in rotating frame: diag(energies - j*omega_d)
    levels = torch.arange(d, dtype=energies.dtype)
    H0_rot = energies - levels * omega_d
    C_total[0] = C_total[0] + torch.diag(H0_rot).to(torch.cfloat)

    # Assemble Floquet Hamiltonian
    N = (2*M + 1) * d
    H_F = torch.zeros((N, N), dtype=torch.cfloat)
    for m in range(-M, M+1):
        row = (m + M) * d
        for n in range(-M, M+1):
            col = (n + M) * d
            idx = m - n
            block = C_total.get(idx, zero_block)
            H_F[row:row + d, col:col + d] = block
        H_F[row:row + d, row:row + d] += m * omega_d * torch.eye(d, dtype=torch.cfloat) 

    T = 2 * math.pi / omega_d  # one period of the drive
    return get_physical_propagator_strong_field(H_F, T, d, M)


def get_physical_propagator_strong_field(H_F, T, d, M):
    # 1. Diagonalize Floquet Hamiltonian
    epsilon, psi = torch.linalg.eigh(H_F)
    
    # 2. Initialize physical propagator
    U_phys = torch.zeros((d, d), dtype=torch.complex128)
    
    # 3. Sum over all Floquet eigenstates
    for alpha in range((2*M+1)*d):
        # Extract eigenstate components
        psi_alpha = psi[:, alpha]
        
        # Compute contribution to propagator
        phase = torch.exp(-1j * epsilon[alpha] * T)
        
        # Sum over all Fourier sectors m
        for j in range(d):
            for k in range(d):
                contrib = 0
                for m in range(-M, M+1):
                    idx_jm = j + (m + M) * d
                    idx_k0 = k + M * d  # m=0 for initial state
                    contrib += psi_alpha[idx_jm] * torch.conj(psi_alpha[idx_k0])
                
                U_phys[j, k] += phase * contrib
    
    return U_phys


if __name__ == "__main__":
    dev, dtype = "cuda", torch.complex128
    n_levels = 6

    # From Table I - complete population transfer
    rabi_frequencies = np.array([.01])

    phases = np.array([.5345234]) * np.pi

    total_time = 2 * np.pi 
    delta = -0.0429
    
    
    EJ_EC_ratio = TransmonCore.find_EJ_EC_for_anharmonicity(delta)
    energies, lambdas_full = TransmonCore.compute_transmon_parameters(
        n_levels, n_charge=30, EJ_EC_ratio=EJ_EC_ratio
    )

    
    rabi_frequencies = torch.tensor(rabi_frequencies, dtype=torch.float64, requires_grad=True)
    phases = torch.tensor(phases, dtype=torch.float64, requires_grad=True)
    
    
    energies = torch.tensor(energies, dtype=torch.float64)
    lambdas_full = torch.tensor(lambdas_full, dtype=torch.complex128)   
    # Compute Fourier coefficients
    fourier_coeffs = compute_fourier_coeffs(rabi_frequencies[0], phases[0], lambdas_full, 200)


    # Compute Floquet propagator for one period
    U = floquet_propagator_square_rabi_one_period(fourier_coeffs, energies, 1, 200)
    print("Floquet propagator for one period:")
    print(U)