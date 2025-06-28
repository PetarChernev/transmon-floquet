from matplotlib import pyplot as plt
import numpy as np
import qutip as qt

from transmon.transmon_core import TransmonCore


def make_driving_func(rabi_frequencies, phases, omega_d, boundaries):
    rabi   = np.asarray(rabi_frequencies, dtype=float)
    phase  = np.asarray(phases, dtype=float)
    t_max = boundaries[-1]
    def drive(t):
        t_eff = t if t <= t_max else t_max

        k = np.searchsorted(boundaries, t_eff, side='right')
        if k == len(rabi):
            k -= 1
        return rabi[k] * np.cos(omega_d * t_eff + phase[k])

    return drive

def pulse_sequence_qutip(rabi_frequencies, phases, pulse_durations, epsilon, lambda_matrix, omega_d, options=None, initial_state=False) -> np.array:
    n = len(epsilon)
    # Validate input dimensions
    assert lambda_matrix.shape == (n, n), f"lambda_matrix must be {n}x{n}"
    options = options or {}
    # H0: Time-independent diagonal Hamiltonian
    H0 = qt.Qobj(np.diag(epsilon))
    
    # H1: Time-dependent coupling Hamiltonian (without time dependence)
    H1 = qt.Qobj(lambda_matrix)
    
    if all(isinstance(d, int) for d in pulse_durations):
        T_period   = 2 * np.pi / omega_d
        boundaries = np.cumsum(pulse_durations, dtype=float) * T_period
        total_time = T_period * np.sum(pulse_durations)
    elif all(isinstance(d, float) for d in pulse_durations):
        boundaries = np.cumsum(pulse_durations, dtype=float)
        total_time = np.sum(pulse_durations)
    else:
        raise ValueError("pulse_durations must be all integers or all floats")
    drive = make_driving_func(rabi_frequencies, phases, omega_d, boundaries)
    args = {
        'omega': float(omega_d),
        'rabi': list(map(float, rabi_frequencies)),
        'phase': list(map(float, phases)),
    }
    H = [H0, [H1, drive]]

    if initial_state:
        tlist = np.linspace(0, total_time, options.get('nsteps', 5000))
        result = qt.sesolve(H, initial_state, tlist, args=args, options=options)
        return tlist, result.states
    U = qt.propagator(H, total_time, args=args, options=options)
    return U.full()


if __name__ == "__main__":
    omega_d = 1

    target = np.array([[1, 1], [1, -1]]) / np.sqrt(2)

    n_levels = 2
    EJ_EC_ratio = TransmonCore.find_EJ_EC_for_anharmonicity(-0.0429)
    energies, couplings = TransmonCore.compute_transmon_parameters(
        n_levels, n_charge=30, EJ_EC_ratio=EJ_EC_ratio
    )

    omega_d = 1.0                                 # driving frequency (rad s⁻¹)


    # Initial state: ground state |0>
    psi0 = qt.basis(2, 0)


    # Compute state vectors
    tlist, states = pulse_sequence_qutip(
        rabi_frequencies=[1,1,1],
        phases=[
            1.510568,
            4.856720,
            3.922234,
        ],
        pulse_durations=[7,1,7], 
        epsilon=energies,
        lambda_matrix=couplings,
        omega_d=omega_d,
        options={
            "atol": 1e-12,
            "rtol": 1e-12,
            "nsteps": 2000
        },
        initial_state=psi0
    )

    # Compute populations in energy basis
    populations = np.array([np.abs(state.full())**2 for state in states])
    pop_g = populations[:, 0, 0]
    pop_e = populations[:, 1, 0]

    # Plot
    plt.figure()
    plt.plot(tlist, pop_g, label='Ground')
    plt.plot(tlist, pop_e, label='Excited')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.legend()
    plt.title('State Populations vs Time for Two-Level Rabi Drive')
    plt.show()
