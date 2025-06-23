import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
import torch

from transmon_core import TransmonCore
from transmon_dynamics import _base_envelope, simulate_transmon_propagator



def drive_envelope_array(times, rabi_frequencies, pulse_duration, pulse_type="square"):
    """Vectorised version that multiplies the unit envelope by the pulse-specific
    Rabi frequency.  Returns an array of Ω_R(t) in the same units as
    `rabi_frequencies`."""
    env = np.zeros_like(times, dtype=float)
    n_pulses = len(rabi_frequencies)
    for i, t in enumerate(times):
        pulse_idx = min(int(t // pulse_duration), n_pulses - 1)
        env[i] = (
            rabi_frequencies[pulse_idx]
            * base_envelope_tensor(t, pulse_idx, pulse_duration, pulse_type)
        )
    return env

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Dynamics
# -----------------------------------------------------------------------------

def base_envelope_tensor(t, pulse_idx, pulse_duration, pulse_type="square"):
    if pulse_type == "square":
        t_start = pulse_idx * pulse_duration
        t_end = (pulse_idx + 1) * pulse_duration
        return (t >= t_start) & (t < t_end)
    elif pulse_type == "gaussian":
        t_center = (pulse_idx + 0.5) * pulse_duration
        sigma = pulse_duration / 6
        return torch.exp(-((t - t_center) ** 2) / (2 * sigma ** 2))
    else:
        raise ValueError(f"Unknown pulse_type '{pulse_type}'")

def transmon_propagator_pytorch(
    rabi_frequencies,
    phases,
    energies,
    lambdas_full,
    *,
    n_levels=6,
    total_time=20.0,
    n_time_steps=2000,
    pulse_type="square",
    use_rwa=True,
    device="cpu"
):
    rabi_frequencies = rabi_frequencies.to(dtype=torch.float64, device=device)
    phases = phases.to(dtype=torch.float64, device=device)
    energies = energies.to(dtype=torch.float64, device=device)
    lambdas_full = lambdas_full.to(dtype=torch.cdouble, device=device)
    U_total = torch.eye(n_levels, dtype=torch.cdouble, device=device)
    if use_rwa:
        lambdas = torch.zeros(n_levels, dtype=torch.cdouble, device=device)
        for j in range(1, n_levels):
            lambdas[j] = lambdas_full[j, j - 1]

    pulse_duration = total_time / len(rabi_frequencies)
    times = torch.linspace(0, total_time, n_time_steps + 1, dtype=torch.float64, device=device)
    dt = times[1] - times[0]


    for i in range(n_time_steps):
        t = times[i]
        pulse_idx = min(int((t / pulse_duration).item()), len(rabi_frequencies) - 1)
        envelope = base_envelope_tensor(t, pulse_idx, pulse_duration, pulse_type)

        H = torch.zeros((n_levels, n_levels), dtype=torch.cdouble, device=device)
        for j in range(n_levels):
            H[j, j] = energies[j] - j

        omega_R = rabi_frequencies[pulse_idx] * envelope

        if use_rwa and omega_R != 0:
            omega_R *= torch.exp(-1j * phases[pulse_idx])
            for j in range(1, n_levels):
                H[j, j - 1] += lambdas[j] * omega_R / 2
                H[j - 1, j] += lambdas[j] * omega_R.conj() / 2
        elif not use_rwa and omega_R != 0:
            omega_d = 1.0
            exp_drive = torch.exp(-1j * (omega_d * t + phases[pulse_idx]))

            j_vals = torch.arange(n_levels, dtype=torch.float64, device=device)
            phase_mat = torch.exp(1j * omega_d * t * (j_vals[:, None] - j_vals[None, :]))

            H_drive = lambdas_full * phase_mat
            H += omega_R / 2 * (exp_drive * H_drive + exp_drive.conj() * H_drive.conj().T)

        U = torch.matrix_exp(-1j * H * dt)
        U_total = U @ U_total

    return U_total



if __name__ == "__main__":
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
    
    rabi_frequencies = torch.tensor(rabi_frequencies, dtype=torch.float64, requires_grad=True)
    phases = torch.tensor(phases, dtype=torch.float64, requires_grad=True)

    # Total time T = 20 ns
    total_time = 20.0 * 2 * np.pi * 7  # Convert to dimensionless units

    # Initial state |0⟩
    initial_state = torch.zeros(n_levels, dtype=torch.complex128)
    initial_state[0] = 1.0
    
    EJ_EC_ratio = TransmonCore.find_EJ_EC_for_anharmonicity(-0.0429)
    
    energies, lambdas_full = TransmonCore.compute_transmon_parameters(
        n_levels, n_charge=30, EJ_EC_ratio=EJ_EC_ratio
    )
    energies = torch.tensor(energies, dtype=torch.float64)
    lambdas_full = torch.tensor(lambdas_full, dtype=torch.complex128)


    # Simulate
    U = transmon_propagator_pytorch(
        rabi_frequencies,
        phases,
        energies=energies,
        lambdas_full=lambdas_full,
        n_levels=n_levels,
        total_time=total_time,
        pulse_type="square",
        n_time_steps=5000,
        use_rwa=False,

    )
    loss = -torch.abs(U[0, 3])**2
    loss.backward()
