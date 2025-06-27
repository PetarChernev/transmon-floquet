import numpy as np
import torch

from typing import Optional, Sequence

from transmon.transmon_dynamics_qutip import pulse_sequence_qutip

def floquet_propagator_square_rabi(
    rabi: torch.Tensor, 
    phase: torch.Tensor,
    energies: torch.Tensor, 
    couplings: torch.Tensor,
    omega_d: torch.Tensor,
    floquet_cutoff: int
) -> torch.Tensor:
    """
    Builds the propagator for a single period of the cosine drive with frequency omega_d
    """
    H_F = build_floquet_hamiltonian(
        rabi, 
        phase,
        energies, 
        couplings,
        omega_d,
        floquet_cutoff
    )
    return get_physical_propagator(H_F, floquet_cutoff, omega_d)

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

def get_physical_propagator(H_F, floquet_cutoff, omega_d):
    """
    Parameters:
    H_F: Floquet Hamiltonian.
    time: Final time at which to compute the propagator U(time, 0)
    floquet_cutoff: M
    omega_d: Frequency of the periodic physical Hamiltonian, T = 2 * pi / omega_d 
    """
    d = H_F.shape[0] // (2 * floquet_cutoff + 1)
    M = floquet_cutoff
    U_F = torch.matrix_exp(-1j * H_F * 2 * torch.pi / omega_d)
    U_phys = torch.zeros((d, d), dtype=torch.complex128, device=H_F.device)
    for m in range(-M, M+1):
        m_idx = m + M
        block_m0 = U_F[m_idx*d:(m_idx+1)*d, M*d:(M+1)*d]
        U_phys += block_m0
    return U_phys



def floquet_propagator_square_sequence_stroboscopic(
    rabi_frequencies: Sequence[float],
    phases: Sequence[float],
    pulse_durations_periods: Sequence[int],
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
        pulse_durations_periods: List of pulse durations in number of periods (one per pulse).
        energies: Tensor of energies of the system (typically shape (n,)).
        lambdas_full: Tensor of couplings between states (shape compatible with Hamiltonian).
        omega_d: Drive frequency.
        floquet_cutoff: Fourier cutoff used in Floquet formalism.

    Returns:
        A PyTorch tensor representing the total propagator after applying all pulses in sequence.
    """
    assert len(rabi_frequencies) == len(phases) == len(pulse_durations_periods), \
        "Mismatched input lengths for rabi_frequencies, phases, and pulse_durations_periods."
    if device is None:
        device = energies.device

    dtype = energies.dtype if energies.is_complex() else torch.complex128
    total_propagator = torch.eye(energies.shape[0], dtype=dtype, device=device)

    for rabi, phase, duration in zip(rabi_frequencies, phases, pulse_durations_periods):
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
        "Mismatched input lengths for rabi_frequencies, phases, and pulse_durations_periods."

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

if __name__ == "__main__":
    device = 'cuda'
    omega_d = 1.5


    test_params = dict(
        rabi_frequencies=torch.tensor([10.05, 0.10, 0.05], dtype=torch.complex128, device=device),
        phases=torch.tensor([0.1231, np.pi/2, np.pi], dtype=torch.complex128, device=device),
        energies=torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.complex128, device=device),
        lambdas_full=torch.tensor([
                [0., 1., 0., 0., 0., 0.],
                [1., 0., 1., 0., 0., 0.],
                [0., 1., 0., 1., 0., 0.],
                [0., 0., 1., 0., 1., 0.],
                [0., 0., 0., 1., 0., 1.],
                [0., 0., 0., 0., 1., 0.],
            ], dtype=torch.complex128, device=device),
        omega_d=torch.tensor([omega_d], dtype=torch.complex128, device=device),
        floquet_cutoff=3 
    )

    time_pulse_durations = [4.0, 5.0, 6.0]

    U_general = floquet_propagator_square_sequence(
        **test_params,
        pulse_durations=time_pulse_durations, 
    ).cpu().numpy()

    U_qutip_general = pulse_sequence_qutip(
        rabi_frequencies=test_params["rabi_frequencies"].cpu().numpy(),
        phases=test_params["phases"].cpu().numpy(),
        pulse_durations=time_pulse_durations, 
        epsilon=test_params["energies"].cpu().numpy(),
        lambda_matrix=test_params["lambdas_full"].cpu().numpy(),
        omega_d=omega_d,
        options={
            "atol": 1e-12,
            "rtol": 1e-12,
            "nsteps": 20000
        }
    ).full()


    import itertools
    import numpy as np

    # collect all propagators
    Us = {
        # "strob":               U_strob,
        "general":             U_general,
        "qutip_general":       U_qutip_general,
        # "qutip_strob":         U_qutip_strob,
    }

    dim = U_qutip_general.shape[0]
    I   = np.eye(dim)

    # 1) compute unitarity errors for each U: ‖U†U − I‖
    unit_err = {
        name: np.linalg.norm(U.conj().T @ U - I).item()
        for name, U in Us.items()
    }

    # 2) compute fidelities for each unordered pair (i<j):
    #    F(U,V) = |Tr(U† V)|/dim
    fidelities = {}
    for (name1, U1), (name2, U2) in itertools.combinations(Us.items(), 2):
        fid = np.abs(np.trace(U1.conj().T @ U2)) / dim
        fidelities[f"{name1} ↔ {name2}"] = fid.item()

    # 3) print results
    print("Unitarity errors:")
    for name, err in unit_err.items():
        print(f"  {name:15s}: {err:.2e}")

    print("\nPairwise fidelities:")
    for pair, fid in fidelities.items():
        print(f"  {pair:25s}: {fid:.12f}")

