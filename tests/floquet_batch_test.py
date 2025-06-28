import torch
import numpy as np
import itertools

from transmon.transmon_floquet_propagator import floquet_propagator_square_sequence



def run_batched_test(
    batch_size: int,
    rabi_vals: np.ndarray,
    phase_vals: np.ndarray,
    energies: np.ndarray,
    lambda_matrix: np.ndarray,
    omega_d_val: float,
    pulse_durations: list[float],
    floquet_cutoff: int = 3,
    device: str = 'cuda'
):
    # Build batched torch tensors
    # rabi_vals: shape (n_levels,) → (batch_size, n_levels)
    rabi = torch.tensor(rabi_vals, dtype=torch.complex128, device=device) \
               .unsqueeze(0).repeat(batch_size, 1)
    phases = torch.tensor(phase_vals, dtype=torch.complex128, device=device) \
                   .unsqueeze(0).repeat(batch_size, 1)

    # energies and lambda are the same for each batch element here;
    # if you want them random per batch, generate them before stacking
    energies_t = torch.tensor(energies, dtype=torch.complex128, device=device) \
                       .unsqueeze(0).repeat(batch_size, 1)
    lambdas_t = torch.tensor(lambda_matrix, dtype=torch.complex128, device=device) \
                       .unsqueeze(0).repeat(batch_size, 1, 1)

    omega_d = torch.tensor([omega_d_val], dtype=torch.complex128, device=device) \
                      .repeat(batch_size)

    # pulse durations can also be broadcasted if you vectorize inside the function;
    # here we assume it accepts a Python list (same for all batches)
    
    # Call the batched propagator routine once:
    U_batch = floquet_propagator_square_sequence(
        rabi_frequencies=rabi,
        phases=phases,
        energies=energies_t,
        couplings=lambdas_t,
        omega_d=omega_d,
        floquet_cutoff=floquet_cutoff,
        pulse_durations=pulse_durations
    )  # → tensor of shape (batch_size, dim, dim)
    
    U_batch = U_batch.cpu().numpy()
    dim = U_batch.shape[-1]
    I = np.eye(dim)

    # For each batch element, compute Qutip reference and metrics
    for i in range(batch_size):
        # extract the i-th parameters
        rabi_i   = rabi_vals
        phase_i  = phase_vals
        eps_i    = energies
        lam_i    = lambda_matrix
        omega_d0 = omega_d_val

        # Qutip compute (returns a single propagator)
        U_qutip = compute_propagator_sequence_qutip(
            rabi_frequencies=rabi_i,
            phases=phase_i,
            pulse_durations=pulse_durations,
            epsilon=eps_i,
            lambda_matrix=lam_i,
            omega_d=omega_d0,
            options={"atol":1e-12, "rtol":1e-12, "nsteps":20000}
        ).full()

        U_torch = U_batch[i]

        # unitarity error
        unit_err = np.linalg.norm(U_torch.conj().T @ U_torch - I)
        # cross‐fidelity
        fid = np.abs(np.trace(U_torch.conj().T @ U_qutip)) / dim

        print(f"Batch #{i:2d}:  unitarity error = {unit_err:.2e},   fidelity vs Qutip = {fid:.12f}")

if __name__ == "__main__":
    # reuse your single‐shot params:
    test_rabi   = np.array([10.05, 0.10, 0.05])
    test_phases = np.array([0.1231, np.pi/2, np.pi])
    test_eps    = np.arange(6, dtype=float)      # [0,1,2,3,4,5]
    test_lambda = np.diag([0,]*6)
    for i in range(5):
        # simple tridiagonal chain
        if i<5:
            test_lambda[i,i+1] = test_lambda[i+1,i] = 1
    test_lambda = test_lambda.astype(float)

    run_batched_test(
        batch_size=4,
        rabi_vals=test_rabi,
        phase_vals=test_phases,
        energies=test_eps,
        lambda_matrix=test_lambda,
        omega_d_val=1.0,
        pulse_durations=[4.0, 5.0, 6.0],
        floquet_cutoff=3,
        device='cuda'
    )
