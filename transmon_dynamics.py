import numpy as np
import scipy as sp
from scipy.linalg import expm
import matplotlib.pyplot as plt

from transmon_core import TransmonCore


def _base_envelope(t, pulse_idx, pulse_duration, pulse_type="square"):
    """Return the *unit* envelope (0…1) for a single pulse at instant t."""
    if pulse_type == "square":
        t_start = pulse_idx * pulse_duration
        t_end = (pulse_idx + 1) * pulse_duration
        return 1.0 if t_start <= t < t_end else 0.0
    elif pulse_type == "gaussian":
        t_center = (pulse_idx + 0.5) * pulse_duration
        sigma = pulse_duration / 6
        return np.exp(-((t - t_center) ** 2) / (2 * sigma ** 2))
    else:
        raise ValueError(f"Unknown pulse_type '{pulse_type}'.")


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
            * _base_envelope(t, pulse_idx, pulse_duration, pulse_type)
        )
    return env

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Dynamics
# -----------------------------------------------------------------------------

def simulate_transmon_dynamics(
    initial_state,
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
):
    """
    Simulate using equation (4) or the full Eq. (A13) depending on `use_rwa`.
    """

    # Extract λj = λ_{j,j-1} (nearest neighbours) only if RWA is used
    if use_rwa:
        lambdas = np.zeros(n_levels)
        for j in range(1, n_levels):
            lambdas[j] = lambdas_full[j, j - 1]

    # Time evolution
    n_pulses = len(rabi_frequencies)
    pulse_duration = total_time / n_pulses

    times = np.linspace(0, total_time, n_time_steps + 1)
    dt = times[1] - times[0]

    states_history = np.zeros((len(times), n_levels), dtype=complex)
    states_history[0] = initial_state
    current_state = initial_state.astype(complex).copy()

    for i, t in enumerate(times[:-1]):
        pulse_idx = min(int(t // pulse_duration), n_pulses - 1)

        # Pulse envelope
        envelope = _base_envelope(t, pulse_idx, pulse_duration, pulse_type)

        # Hamiltonian (equation 4 or A13) - MUST BE COMPLEX!
        H = np.zeros((n_levels, n_levels), dtype=complex)

        # Diagonal terms
        for j in range(n_levels):
            H[j, j] = energies[j] - j  # μj = ej - j

        omega_R = rabi_frequencies[pulse_idx] * envelope

        if use_rwa and omega_R != 0:
            omega_R *= np.exp(-1j * phases[pulse_idx])
            for j in range(1, n_levels):
                H[j, j - 1] += lambdas[j] * omega_R / 2
                H[j - 1, j] += lambdas[j] * np.conj(omega_R) / 2

        elif not use_rwa and omega_R != 0:
            omega_d = 1.0  # ω_d = ω_01 in our rescaled units
            exp_drive = np.exp(-1j * (omega_d * t + phases[pulse_idx]))

            # Phase matrix e^{i(j-l)ω_d t}
            phase_mat = np.exp(
                1j * omega_d * t * (np.arange(n_levels)[:, None] - np.arange(n_levels))
            )

            H_drive = lambdas_full * phase_mat  # λ_{j,l} e^{i(j-l)ω_d t}
            H += omega_R / 2 * (exp_drive * H_drive + np.conj(exp_drive) * H_drive.conj().T)

        # Time evolution
        U = expm(-1j * H * dt)
        current_state = U @ current_state
        states_history[i + 1] = current_state

    return current_state, states_history, times


def simulate_transmon_propagator(
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

):
    """
    Simulate using equation (4) or the full Eq. (A13) depending on `use_rwa`.
    """
    # Get transmon parameters


    # Extract λj = λ_{j,j-1} (nearest neighbours) only if RWA is used
    if use_rwa:
        lambdas = np.zeros(n_levels)
        for j in range(1, n_levels):
            lambdas[j] = lambdas_full[j, j - 1]

    # Time evolution
    n_pulses = len(rabi_frequencies)
    pulse_duration = total_time / n_pulses

    times = np.linspace(0, total_time, n_time_steps + 1)
    dt = times[1] - times[0]
    U_total = np.eye(n_levels, dtype=complex)


    for i, t in enumerate(times[:-1]):
        pulse_idx = min(int(t // pulse_duration), n_pulses - 1)

        # Pulse envelope
        envelope = _base_envelope(t, pulse_idx, pulse_duration, pulse_type)

        # Hamiltonian (equation 4 or A13) - MUST BE COMPLEX!
        H = np.zeros((n_levels, n_levels), dtype=complex)

        # Diagonal terms
        for j in range(n_levels):
            H[j, j] = energies[j] - j  # μj = ej - j

        omega_R = rabi_frequencies[pulse_idx] * envelope
        omega_R = np.asarray(omega_R, dtype=np.complex128) 
        if use_rwa and omega_R != 0:
            omega_R *= np.exp(-1j * phases[pulse_idx])
            for j in range(1, n_levels):
                H[j, j - 1] += lambdas[j] * omega_R / 2
                H[j - 1, j] += lambdas[j] * np.conj(omega_R) / 2

        elif not use_rwa and omega_R != 0:
            omega_d = 1.0  # ω_d = ω_01 in our rescaled units
            exp_drive = np.exp(-1j * (omega_d * t + phases[pulse_idx]))

            # Phase matrix e^{i(j-l)ω_d t}
            phase_mat = np.exp(
                1j * omega_d * t * (np.arange(n_levels)[:, None] - np.arange(n_levels))
            )

            H_drive = lambdas_full * phase_mat  # λ_{j,l} e^{i(j-l)ω_d t}
            exp_drive = np.asarray(exp_drive, dtype=np.complex64)
            H_drive = np.asarray(H_drive, dtype=np.complex64)
            omega_R = np.asarray(omega_R, dtype=np.complex64)
            term = exp_drive * H_drive + np.conj(exp_drive) * H_drive.conj().T
            H += (omega_R / 2) * term
        # Time evolution
        U = sp.linalg.expm(-1j * H * dt)
        U_total = U @ U_total

    return U_total


if __name__ == "__main__":
    n_levels = 6

    # From Table I - complete population transfer
    rabi_frequencies = np.array(
        [31.651, 44.988, 69.97, 60.608, 66.029, 68.771, 69.562, 66.971]
    )

    rabi_frequencies = (
        rabi_frequencies / 7000
    )  # Convert to GHz from MHz and normalise by ω01

    phases = np.array(
        [0.1779, 0.0499, 0.1239, 0.2538, 0.2886, 0.1688,0.1645, 0.1234]
    ) * np.pi
    
    print(phases)
    print(rabi_frequencies)

    # Total time T = 20 ns
    total_time = 20.0 * 2 * np.pi * 7  # Convert to dimensionless units

    # Initial state |0⟩
    initial_state = np.zeros(n_levels, dtype=complex)
    initial_state[0] = 1.0
    
    EJ_EC_ratio = TransmonCore.find_EJ_EC_for_anharmonicity(-0.0429)
    energies, lambdas_full = TransmonCore.compute_transmon_parameters(
        n_levels, n_charge=30, EJ_EC_ratio=EJ_EC_ratio
    )

    
    U = simulate_transmon_propagator(
        rabi_frequencies,
        phases,
        n_levels=n_levels,
        total_time=total_time,
        pulse_type="square",
        n_time_steps=5000,
        use_rwa=False,
        energies=energies,
        lambdas_full=lambdas_full
    )
    print(U)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    inner = np.trace(np.conj(X.T) @ U[:2, :2])
    F = np.abs(inner) / 2.0
    print(f"Fidelity: {F:.12f}")
    
    # Simulate
    final_state, states_history, times = simulate_transmon_dynamics(
        initial_state,
        rabi_frequencies,
        phases,
        n_levels=n_levels,
        total_time=total_time,
        pulse_type="square",
        n_time_steps=5000,
        use_rwa=False,
        energies=energies,
        lambdas_full=lambdas_full
    )
    # Plot
    populations = np.abs(states_history) ** 2

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    for level in range(4):
        plt.plot(times, populations[:, level], label=f"|{level}⟩")
    plt.xlabel("Time (dimensionless)")
    plt.ylabel("Population")
    plt.title("Population Transfer with Composite Pulses")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    pulse_duration = total_time / len(rabi_frequencies)
    pulse_plot = drive_envelope_array(
        times, rabi_frequencies, pulse_duration, pulse_type="gaussian"
    )
    plt.plot(times, pulse_plot, "r-", linewidth=2)
    plt.xlabel("Time (dimensionless)")
    plt.ylabel("Rabi frequency")
    plt.title("Pulse Sequence")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    print(f"\nFinal populations:")
    for level in range(4):
        print(f"|{level}⟩: {np.abs(final_state[level]) ** 2:.6f}")
