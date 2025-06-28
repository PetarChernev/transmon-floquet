import time

import numpy as np
import torch

from optimization.cma_fidelity import PulseEncoding
from tests.floquet_parallel_test import get_system_params
from transmon.transmon_floquet_propagator import build_floquet_hamiltonian, floquet_propagator_square_sequence_stroboscopic
from transmon.transmon_floquet_propagator_parallel import build_floquet_hamiltonians, floquet_propagator_square_sequence_batch, floquet_propagator_square_sequence_stroboscopic_vectorized

def benchmark_build_hamiltonians(
    n_levels: int,
    M_values: list[int],
    n_pulses_values: list[int],
    device=torch.device("cuda"),
    dtype_real=torch.float64,
):
    """
    For each M in M_values and each n_pulses in n_pulses_values,
    generate a random rabi/phase sequence and time:

      1) building H_F sequentially  (build_floquet_hamiltonian in a loop)
      2) building H_F in parallel   (build_floquet_hamiltonians)

    Returns a dict keyed by (M, n_pulses) → (t_seq, t_par).
    """
    results = {}
    for M in M_values:
        # get system params once per M
        system_params = get_system_params(n_levels)
        system_params['M'] = M

        for n_pulses in n_pulses_values:
            # generate a random pulse vector p0 of length 3*n_pulses
            p0 = np.concatenate([
                np.random.rand(n_pulses),            # rabi
                np.random.rand(n_pulses) * 2*np.pi,  # phase
                np.random.randint(1, 5, size=n_pulses)  # dummy periods
            ])
            # decode into arrays
            rabi, phase, _ = PulseEncoding.decode(p0)
            rabi_t  = torch.tensor(rabi, dtype=dtype_real,   device=device)
            phase_t = torch.tensor(phase, dtype=dtype_real,  device=device)

            # Sequential
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            Hs_seq = [
                build_floquet_hamiltonian(r, p, **system_params)
                for r, p in zip(rabi_t, phase_t)
            ]
            torch.cuda.synchronize()
            t_seq = time.perf_counter() - t0

            # Parallel
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            Hs_par = build_floquet_hamiltonians(rabi_t, phase_t, **system_params)
            torch.cuda.synchronize()
            t_par = time.perf_counter() - t0

            print(f"M={M:3d}, n_pulses={n_pulses:3d} → seq: {t_seq:.4f}s, par: {t_par:.4f}s")
            results[(M, n_pulses)] = (t_seq, t_par)

    return results


def benchmark_propagators(
    n_levels: int,
    M_values: list[int],
    n_pulses_values: list[int],
    device=torch.device("cuda"),
    dtype_real=torch.float64,
    with_backpropagation: bool = False
):
    """
    For each M in M_values and each n_pulses in n_pulses_values,
    generate a random rabi/phase sequence and time:

      1) building H_F sequentially  (build_floquet_hamiltonian in a loop)
      2) building H_F in parallel   (build_floquet_hamiltonians)

    Returns a dict keyed by (M, n_pulses) → (t_seq, t_par).
    """
    results = {}
    for M in M_values:
        # get system params once per M
        system_params = get_system_params(n_levels)
        del system_params['M']  # remove M, not needed for propagator
        system_params['floquet_cutoff'] = M

        for n_pulses in n_pulses_values:
            # generate a random pulse vector p0 of length 3*n_pulses
            p0 = np.concatenate([
                np.random.rand(n_pulses),            # rabi
                np.random.rand(n_pulses) * 2*np.pi,  # phase
                np.random.randint(1, 5, size=n_pulses)  # dummy periods
            ])
            # decode into arrays
            rabi, phase, dur = PulseEncoding.decode(p0)
            rabi_t  = torch.tensor(rabi, dtype=dtype_real,   device=device)
            phase_t = torch.tensor(phase, dtype=dtype_real,  device=device)
            dur_t = torch.tensor(dur, dtype=torch.int16,  device=device)

            # Sequential
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            U_sequential = floquet_propagator_square_sequence_stroboscopic(
                rabi_t, phase_t, dur_t, **system_params, device=device
            )
            if with_backpropagation:
                U_sequential.requires_grad_(True)
                U_sequential.sum().abs().backward()
            torch.cuda.synchronize()
            t_seq = time.perf_counter() - t0

            # Parallel
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            U_parallel = floquet_propagator_square_sequence_stroboscopic_vectorized(
                rabi_frequencies=rabi_t, 
                phases=phase_t, 
                pulse_durations_periods=dur_t, 
                **system_params, 
                device=device
            )
            if with_backpropagation:
                U_parallel.requires_grad_(True)
                U_parallel.sum().abs().backward()
            torch.cuda.synchronize()
            t_par = time.perf_counter() - t0

            print(f"M={M:3d}, n_pulses={n_pulses:3d} → seq: {t_seq:.4f}s, par: {t_par:.4f}s")
            results[(M, n_pulses)] = (t_seq, t_par)

    return results


def benchmark_propagators_batched(
    n_levels: int,
    M_values: list[int],
    n_pulses_values: list[int],
    n_chains_values: list[int],           # NEW
    device=torch.device("cuda"),
    dtype_real=torch.float64,
    with_backpropagation: bool = False,
):
    """
    For each (M, n_pulses, B) triple:

        • draw a batch of B random pulse sequences that share the same
          integer duration vector;

        • measure the time spent computing the propagators with
          - the *old* per-chain loop over
            `floquet_propagator_square_sequence_stroboscopic_vectorized`,
          - the *new* `floquet_propagator_square_sequence_batch`.

    Returns
    -------
    dict keyed by (M, n_pulses, B) → (t_old, t_batch)
    """
    results = {}

    for M in M_values:
        # --- static chip/Floquet parameters for this M --------------------
        system_params = get_system_params(n_levels)
        system_params["floquet_cutoff"] = system_params.pop("M")

        for n_pulses in n_pulses_values:
            # integer pulse durations – identical for the whole batch
            dur_np = np.random.randint(1, 5, size=n_pulses)
            dur_t  = torch.tensor(dur_np, dtype=torch.int64, device=device)

            for B in n_chains_values:

                # -------------- draw random continuous parameters ----------
                rabi_np  = np.random.rand(B, n_pulses)
                phase_np = np.random.rand(B, n_pulses) * 2 * np.pi

                rabi_t  = torch.tensor(rabi_np,  dtype=dtype_real, device=device)
                phase_t = torch.tensor(phase_np, dtype=dtype_real, device=device)

                # ============================================================
                # ❶ OLD “parallel” path – Python loop over chains
                # ============================================================
                torch.cuda.synchronize()
                t0 = time.perf_counter()

                U_old = []
                for b in range(B):
                    U = floquet_propagator_square_sequence_stroboscopic_vectorized(
                        rabi_frequencies=rabi_t[b],
                        phases=phase_t[b],
                        pulse_durations_periods=dur_t,
                        **system_params,
                        device=device,
                    )
                    U_old.append(U)

                U_old = torch.stack(U_old)                # (B, d, d)

                if with_backpropagation:
                    U_old.requires_grad_(True)
                    U_old.sum().abs().backward()

                torch.cuda.synchronize()
                t_old = time.perf_counter() - t0

                # ============================================================
                # ❷ NEW batched implementation – one GPU call
                # ============================================================
                torch.cuda.synchronize()
                t0 = time.perf_counter()

                U_batch = floquet_propagator_square_sequence_batch(
                    rabi=rabi_t,
                    phases=phase_t,
                    pulse_durations_periods=dur_t,
                    **system_params,
                    device=device,
                )

                if with_backpropagation:
                    U_batch.requires_grad_(True)
                    U_batch.sum().abs().backward()

                torch.cuda.synchronize()
                t_batch = time.perf_counter() - t0

                print(
                    f"M={M:3d}, n_pulses={n_pulses:2d}, "
                    f"B={B:3d} → old-loop: {t_old:.4f}s, batched: {t_batch:.4f}s"
                )

                results[(M, n_pulses, B)] = (t_old, t_batch)

    return results


if __name__ == "__main__":
    print("All tests passed successfully!\n")

    # example benchmark
    M_values = [40]
    n_pulses_values = [10]
    timings = benchmark_propagators_batched(
        n_levels=6,
        M_values=M_values,
        n_pulses_values=n_pulses_values,
        n_chains_values=[10],
        device='cuda',
        with_backpropagation=True
    )