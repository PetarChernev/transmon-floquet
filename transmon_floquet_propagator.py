import math
import torch

from typing import Optional, Sequence

def floquet_propagator_square_rabi(
    rabi: torch.Tensor, 
    phase: torch.Tensor,
    energies: torch.Tensor, 
    couplings: torch.Tensor,
    omega_d: float,
    floquet_cutoff: int,
    time: Optional[float] = None
) -> torch.Tensor:
    """
    Build truncated Floquet Hamiltonian in the lab frame.
    """

    H_F = build_floquet_hamiltonian(
        rabi, 
        phase,
        energies, 
        couplings,
        omega_d,
        floquet_cutoff
    )
    if time is None:
        time = 2 * math.pi / omega_d  # one period of the drive
    return get_physical_propagator_strong_field(H_F, time, floquet_cutoff)

def build_floquet_hamiltonian(
    rabi: torch.Tensor, 
    phase: torch.Tensor,
    energies: torch.Tensor, 
    couplings: torch.Tensor,
    omega_d: float,
    M: int
) -> torch.Tensor:
    d = energies.numel()
    # Prepare zero block
    # Collect all Fourier blocks (drive + H0 later)

    C_0 = torch.diag(energies)
    C_1 = (rabi / 2) * couplings * torch.exp(1j * phase)
    C_m1 = (rabi / 2) * couplings * torch.exp(-1j * phase)
    # Assemble Floquet Hamiltonian
    N = (2*M + 1) * d
    H_F = torch.zeros((N, N), dtype=torch.complex128, device=energies.device)
    for m in range(-M, M+1):
        row = (m + M) * d
        for n in range(-M, M+1):
            col = (n + M) * d
            idx = m - n
            if idx == 0:
                block = C_0
            elif idx == 1:
                block = C_1
            elif idx == -1:
                block = C_m1
            else:
                block = None
            if block is not None:
                H_F[row:row + d, col:col + d] = block
        H_F[row:row + d, row:row + d] += \
            m * omega_d * torch.eye(d, dtype=torch.complex128, device=energies.device) 
    return H_F

def get_physical_propagator_strong_field(H_F, time, floquet_cutoff):
    # 1. Diagonalize Floquet Hamiltonian
    epsilon, psi = torch.linalg.eigh(H_F)  # shape: [(2M+1)d, (2M+1)d]
    d = H_F.size(0) // (2 * floquet_cutoff + 1)
    
    # 3. Initialize physical propagator
    U_phys = torch.zeros((d, d), dtype=torch.complex128, device=H_F.device)
    
    # 4. Reshape psi for easier indexing
    psi = psi.view((2*floquet_cutoff+1), d, -1)  # shape: [(2M+1), d, (2M+1)d]

    # 5. Loop over Floquet eigenstates
    for alpha in range((2*floquet_cutoff+1)*d):
        psi_alpha = psi[:, :, alpha]  # shape: [(2M+1), d]
        
        # psi_alpha_m0: shape [d] for m = 0 sector
        psi_alpha_m0 = psi_alpha[floquet_cutoff, :]  # m = 0

        # Compute outer product of psi_alpha summed over all m
        contrib = torch.einsum('mj,k->jk', psi_alpha, torch.conj(psi_alpha_m0))
        # Add contribution with the phase
        U_phys += torch.exp(-1j * epsilon[alpha] * time) * contrib

    return U_phys



def floquet_propagator_square_sequence_stroboscopic(
    rabi_frequencies: Sequence[float],
    phases: Sequence[float],
    pulse_duration_periods: Sequence[int],
    energies: torch.Tensor,
    lambdas_full: torch.Tensor,
    omega_d: float,
    floquet_cutoff: int,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Compute the total propagator for a sequence of square pulses, each lasting an integer number
    of periods of the drive frequency omega_d.

    Parameters:
        rabi_frequencies: List of Rabi frequencies (one per pulse).
        phases: List of phases (one per pulse).
        pulse_duration_periods: List of pulse durations in number of periods (one per pulse).
        energies: Tensor of energies of the system (typically shape (n,)).
        lambdas_full: Tensor of couplings between states (shape compatible with Hamiltonian).
        omega_d: Drive frequency.
        floquet_cutoff: Fourier cutoff used in Floquet formalism.

    Returns:
        A PyTorch tensor representing the total propagator after applying all pulses in sequence.
    """
    assert len(rabi_frequencies) == len(phases) == len(pulse_duration_periods), \
        "Mismatched input lengths for rabi_frequencies, phases, and pulse_duration_periods."
    if device is None:
        device = energies.device

    dtype = energies.dtype if energies.is_complex() else torch.complex128
    total_propagator = torch.eye(energies.shape[0], dtype=dtype, device=device)

    for rabi, phase, duration in zip(rabi_frequencies, phases, pulse_duration_periods):
        # Compute single-period propagator
        U_single = floquet_propagator_square_rabi(
            rabi,
            phase,
            energies,
            lambdas_full,
            omega_d,
            floquet_cutoff
        )

        # Raise to the power of duration (number of periods)
        U_powered = torch.matrix_power(U_single, duration)

        # Compose with the total propagator
        total_propagator = U_powered @ total_propagator

    return total_propagator



def floquet_propagator_square_sequence(
    rabi_frequencies: Sequence[float],
    phases: Sequence[float],
    pulse_durations: Sequence[float],
    energies: torch.Tensor,
    lambdas_full: torch.Tensor,
    omega_d: float,
    floquet_cutoff: int,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Compute the total propagator for a sequence of square pulses, each lasting an integer number
    of periods of the drive frequency omega_d.

    Parameters:
        rabi_frequencies: List of Rabi frequencies (one per pulse).
        phases: List of phases (one per pulse).
        pulse_durations: List of pulse durations (one per pulse).
        energies: Tensor of energies of the system (typically shape (n,)).
        lambdas_full: Tensor of couplings between states (shape compatible with Hamiltonian).
        omega_d: Drive frequency.
        floquet_cutoff: Fourier cutoff used in Floquet formalism.

    Returns:
        A PyTorch tensor representing the total propagator after applying all pulses in sequence.
    """
    assert len(rabi_frequencies) == len(phases) == len(pulse_durations), \
        "Mismatched input lengths for rabi_frequencies, phases, and pulse_duration_periods."

    if device is None:
        device = energies.device
    dtype = energies.dtype if energies.is_complex() else torch.complex128
    total_propagator = torch.eye(energies.shape[0], dtype=dtype, device=device)

    for rabi, phase, duration in zip(rabi_frequencies, phases, pulse_durations):
        # Compute single-period propagator
        U_pulse = floquet_propagator_square_rabi(
            rabi,
            phase,
            energies,
            lambdas_full,
            omega_d,
            floquet_cutoff,
            time=duration
        )
        # Compose with the total propagator
        total_propagator = U_pulse @ total_propagator

    return total_propagator