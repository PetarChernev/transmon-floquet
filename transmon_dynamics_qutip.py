import numpy as np
import qutip as qt
import matplotlib.pyplot as plt


def make_driving_func(rabi_frequencies, phases, omega_d, boundaries):
    rabi   = np.asarray(rabi_frequencies, dtype=float)
    phase  = np.asarray(phases, dtype=float)
    t_max = boundaries[-1]
    def drive(t):
        t_eff = t if t <= t_max else t_max
        
        k = np.searchsorted(boundaries, t_eff, side='left')
        return rabi[k] * np.cos(omega_d * t_eff + phase[k])

    return drive

def compute_propagator_sequence_qutip(rabi_frequencies, phases, pulse_durations, epsilon, lambda_matrix, omega_d, options=None):
    n = len(epsilon)
    # Validate input dimensions
    assert lambda_matrix.shape == (n, n), f"lambda_matrix must be {n}x{n}"
    
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
    U = qt.propagator(H, total_time, args=args, options=options)
    
    return U