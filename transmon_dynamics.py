import numpy as np
from scipy.linalg import eigh, expm
import matplotlib.pyplot as plt


def construct_transmon_hamiltonian_charge_basis(n_charge=30, EJ_EC_ratio=50):
    """
    Construct the transmon Hamiltonian in the charge basis (equation A1).
    H = 4EC Σ_j (j-n_cut)² δ_jk - EJ/2 (δ_j+1,k + δ_j-1,k)
    """
    n_states = 2 * n_charge + 1
    H_charge = np.zeros((n_states, n_states))

    # Diagonal: charging energy
    for i in range(n_states):
        n = i - n_charge
        H_charge[i, i] = 4 * n ** 2  # In units of EC

    # Off-diagonal: Josephson tunnelling
    EJ = EJ_EC_ratio  # In units of EC
    for i in range(n_states - 1):
        H_charge[i, i + 1] = -EJ / 2
        H_charge[i + 1, i] = -EJ / 2

    return H_charge


def find_EJ_EC_for_anharmonicity(target_anharmonicity=-0.0429, n_charge=30):
    """
    Find EJ/EC ratio that gives the target anharmonicity.
    Only needs first 3 energy levels to compute anharmonicity.
    """
    ratios = np.linspace(20, 200, 100)
    anharms = []

    for ratio in ratios:
        H_charge = construct_transmon_hamiltonian_charge_basis(n_charge, ratio)
        eigenvalues, _ = eigh(H_charge)
        eigenvalues = np.sort(eigenvalues)

        # Anharmonicity only depends on first 3 levels
        E0, E1, E2 = eigenvalues[:3]
        omega_01 = E1 - E0
        omega_12 = E2 - E1
        anharm = (omega_12 - omega_01) / omega_01
        anharms.append(anharm)

    anharms = np.array(anharms)
    idx = np.argmin(np.abs(anharms - target_anharmonicity))

    print(f"Target anharmonicity: {target_anharmonicity:.4f}")
    print(f"Achieved anharmonicity: {anharms[idx]:.4f}")

    return ratios[idx]


def compute_transmon_parameters(n_levels=6, n_charge=30, target_anharmonicity=-0.0429):
    """
    Numerically derive the transmon parameters following the paper's procedure.

    Returns:
    - energies: Energy eigenvalues
    - lambdas: Matrix elements λi,j = ⟨i|n̂|j⟩ in energy eigenbasis
    """
    # Find appropriate EJ/EC ratio
    EJ_EC_ratio = find_EJ_EC_for_anharmonicity(target_anharmonicity, n_charge)
    print(f"Using EJ/EC = {EJ_EC_ratio:.1f}")

    # Construct Hamiltonian in charge basis
    H_charge = construct_transmon_hamiltonian_charge_basis(n_charge, EJ_EC_ratio)

    # Charge operator in charge basis: n̂|n⟩ = n|n⟩
    n_op_charge = np.diag(np.arange(-n_charge, n_charge + 1, dtype=float))

    # Diagonalise to get energy eigenbasis
    eigenvalues, eigenvectors = eigh(H_charge)

    # Sort by energy
    idx = eigenvalues.argsort()
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Transform charge operator to energy eigenbasis
    n_op_energy = eigenvectors.T @ n_op_charge @ eigenvectors

    # Extract first n_levels
    energies = eigenvalues[:n_levels]
    lambdas_full = n_op_energy[:n_levels, :n_levels]

    # Normalise so ω01 = 1
    omega_01 = energies[1] - energies[0]
    energies = (energies - energies[0]) / omega_01

    # Verify properties mentioned in the paper
    print(f"\nVerifying λ properties:")
    print(f"λ_ii (should be 0): {[lambdas_full[i, i] for i in range(min(3, n_levels))]}")
    print(f"λ*_ij = λ_ji? {np.allclose(lambdas_full, lambdas_full.T.conj())}")

    # For transmon, matrix elements should be real
    if np.max(np.abs(np.imag(lambdas_full))) > 1e-10:
        print(
            f"Warning: Imaginary parts found, max = {np.max(np.abs(np.imag(lambdas_full)))}"
        )

    return energies, np.real(lambdas_full)

# -----------------------------------------------------------------------------
# NEW: pulse-envelope helpers
# -----------------------------------------------------------------------------

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
    energies, lambdas_full = compute_transmon_parameters(
        n_levels, n_charge=30, target_anharmonicity=-0.3 / 7
    )

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

    # Total time T = 20 ns
    total_time = 20.0 * 2 * np.pi * 7  # Convert to dimensionless units

    # Initial state |0⟩
    initial_state = np.zeros(n_levels, dtype=complex)
    initial_state[0] = 1.0

    # Simulate
    final_state, states_history, times = simulate_transmon_dynamics(
        initial_state,
        rabi_frequencies,
        phases,
        n_levels=n_levels,
        total_time=total_time,
        pulse_type="gaussian",
        n_time_steps=5000,
        use_rwa=False
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
