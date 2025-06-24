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
        return 1.0 if t_start <= t <= t_end else 0.0
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
            omega_d = 1.0
            # build matrix of level-differences Δ_{jl} = j - l
            levels = np.arange(n_levels)
            delta = levels[:, None] - levels[None, :]

            # two sideband factors: e^{±i Δ ω_d t}
            e_pos = np.exp(1j * delta * omega_d * t)
            e_neg = np.exp(-1j * delta * omega_d * t)

            # drive coupling piece: λ_{jl} e^{iΔωt} + λ_{lj} e^{-iΔωt}
            H_drive = lambdas_full * e_pos + lambdas_full.T * e_neg

            # time-dependent envelope: (Ω) cos(ω_d t + φ)
            drive_factor = omega_R * np.cos(omega_d * t + phases[pulse_idx])

            # add all four sidebands at once
            H += drive_factor * H_drive
        # Time evolution
        U = expm(-1j * H * dt)
        current_state = U @ current_state
        states_history[i + 1] = current_state

    return current_state, states_history, times


def plot_sequence():
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
    energies, couplings = TransmonCore.compute_transmon_parameters(
        n_levels, n_charge=30, EJ_EC_ratio=EJ_EC_ratio
    )

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
        lambdas_full=couplings
    )
    
    # Simulate RWA
    final_state_rwa, states_history_rwa, times = simulate_transmon_dynamics(
        initial_state,
        rabi_frequencies,
        phases,
        n_levels=n_levels,
        total_time=total_time,
        pulse_type="square",
        n_time_steps=5000,
        use_rwa=True,
        energies=energies,
        lambdas_full=couplings
    )
    # Plot
    populations = np.abs(states_history) ** 2
    populations_rwa = np.abs(states_history_rwa) ** 2

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    level_colors = ['b', 'g', 'r', 'c']
    for level in range(4):
        plt.plot(times, populations[:, level], label=f"|{level}⟩", c=level_colors[level])
        plt.plot(times, populations_rwa[:, level], label=f"|{level}⟩_rwa", c=level_colors[level], ls="--")
    plt.xlabel("Time (dimensionless)")
    plt.ylabel("Population")
    plt.title("Population Transfer with Composite Pulses")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    pulse_duration = total_time / len(rabi_frequencies)
    pulse_plot = drive_envelope_array(
        times, rabi_frequencies, pulse_duration, pulse_type="square"
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
        
    
    print(f"\nFinal populations (RWA):")
    for level in range(4):
        print(f"|{level}⟩: {np.abs(final_state_rwa[level]) ** 2:.6f}")  
        
          
def plot_single_period():
    n_levels = 6

    # From Table I - complete population transfer
    rabi_frequencies = np.array(
        [1]
    )


    phases = np.array(
        [0]
    ) * np.pi
    
    print(phases)
    print(rabi_frequencies)

    # Total time T = 20 ns
    total_time = 2 * np.pi  # Convert to dimensionless units

    # Initial state |0⟩
    initial_state = np.zeros(n_levels, dtype=complex)
    initial_state[0] = 1.0
    
    EJ_EC_ratio = TransmonCore.find_EJ_EC_for_anharmonicity(-0.0429)
    energies, couplings = TransmonCore.compute_transmon_parameters(
        n_levels, n_charge=30, EJ_EC_ratio=EJ_EC_ratio
    )
    
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
        lambdas_full=couplings
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
        times, rabi_frequencies, pulse_duration, pulse_type="square"
    )
    omega_d = 1.0  # drive frequency in dimensionless units
    drive_signal = pulse_plot * np.cos(omega_d * times + phases[0])

    plt.plot(times, pulse_plot, "r-", linewidth=2)
    plt.plot(times, drive_signal, 'b--', linewidth=1, label='Ω_R cos(ω_d t + φ)')

    plt.xlabel("Time (dimensionless)")
    plt.ylabel("Rabi frequency")
    plt.title("Pulse Sequence")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    print(f"\nFinal populations:")
    for level in range(4):
        print(f"|{level}⟩: {np.abs(final_state[level]) ** 2:.6f}")


if __name__ == "__main__":
    plot_sequence()