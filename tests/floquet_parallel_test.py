import gc
import numpy as np
import torch

from optimization.cma_fidelity import PulseEncoding
from transmon.transmon_core import TransmonCore
from transmon.transmon_floquet_propagator import build_floquet_hamiltonian, floquet_propagator_square_sequence_stroboscopic, get_physical_propagator
from transmon.transmon_floquet_propagator_parallel import build_floquet_hamiltonians, floquet_propagator_square_sequence_batch, floquet_propagator_square_sequence_omega_batch, floquet_propagator_square_sequence_stroboscopic_vectorized, get_physical_propagators


# technical params
device        = torch.device("cuda")
dtype_real = torch.float64
dtype_complex = torch.complex128


n_levels_cases = [2, 3, 6]
n_pulses_cases = [1, 2, 3, 5, 10, 13, 20]

U_target = torch.tensor([[0, 1], [1, 0]], dtype=dtype_complex, device=device)  # target unitary on first 2 levels

def get_system_params(n_levels):
    EJ_EC_ratio = TransmonCore.find_EJ_EC_for_anharmonicity(-0.0429)
    energies, couplings = TransmonCore.compute_transmon_parameters(
        n_levels, n_charge=30, EJ_EC_ratio=EJ_EC_ratio
    )
    energies = torch.tensor(energies, dtype=dtype_real, device=device)
    couplings = torch.tensor(couplings, dtype=dtype_complex, device=device)

    omega_d = torch.tensor(1.0, dtype=dtype_real, device=device)                              
    floquet_cutoff: int = 50

    return dict(
        energies=energies,
        couplings=couplings,
        omega_d=omega_d,
        M=floquet_cutoff
    )

def get_random_propagator(p0, system_params):

    rabi, phase, periods_i = PulseEncoding.decode(p0)
    # to torch
    rabi_t  = torch.tensor(rabi, dtype=dtype_real, device=device)
    phase_t = torch.tensor(phase, dtype=dtype_real, device=device)
    dur_t   = torch.tensor(periods_i,     dtype=torch.int8,  device=device) 
    return floquet_propagator_square_sequence_stroboscopic(
                rabi_t, phase_t, dur_t, **system_params, device=device
        )  



def test_sombe_hamiltonian_and_folding():
    for n_levels in n_levels_cases:
        system_params = get_system_params(n_levels)
        for n_pulses in n_pulses_cases:
            print(f"Testing n_levels={n_levels}, n_pulses={n_pulses}")
            p0 = np.concatenate([
                np.random.random(n_pulses),
                np.random.random(n_pulses) * 2 * np.pi,
                np.random.random(n_pulses)
            ])
            rabi, phase, periods_i = PulseEncoding.decode(p0)
            # to torch
            rabi_t  = torch.tensor(rabi, dtype=dtype_real, device=device)
            phase_t = torch.tensor(phase, dtype=dtype_real, device=device)
            dur_t   = torch.tensor(periods_i,     dtype=torch.int8,  device=device) 
            H_Fs_sequential = [
                build_floquet_hamiltonian(rabi, phase, **system_params)\
                for rabi, phase in zip(rabi_t, phase_t)
            ]
            H_Fs_parallel = build_floquet_hamiltonians(
                rabi_t, phase_t, **system_params
            )
            for i, H_F in enumerate(H_Fs_sequential):
                H_F_parallel = H_Fs_parallel[i, :, :]
                # Check if the Hamiltonians are equal
                assert torch.allclose(H_F, H_F_parallel, atol=1e-6), \
                    f"Hamiltonian mismatch at pulse {i} for n_levels={n_levels}, n_pulses={n_pulses}"
            
            propagators_sequential = [
                get_physical_propagator(H_F, floquet_cutoff=system_params['M'], omega_d=system_params['omega_d'])
                for H_F in H_Fs_sequential
            ]
            
            propagators_parallel = get_physical_propagators(
                H_Fs_parallel, 
                floquet_cutoff=system_params['M'], 
                omega_d=system_params['omega_d']
            )
            
            for i, U_i in enumerate(propagators_sequential):
                U_i_parallel = propagators_parallel[i, :, :]
                # Check if the Hamiltonians are equal
                assert torch.allclose(U_i, U_i_parallel, atol=1e-6), \
                    f"Propagator mismatch for n_levels={n_levels}, n_pulses={n_pulses}"


def test_full_propagator():
    for n_levels in n_levels_cases:
        system_params = get_system_params(n_levels)
        system_params['floquet_cutoff'] = system_params['M'] 
        del system_params['M']  # remove M, not needed for propagator
        for n_pulses in n_pulses_cases:
            print(f"Testing n_levels={n_levels}, n_pulses={n_pulses}")

            p0 = np.concatenate([
                np.random.random(n_pulses),
                np.random.random(n_pulses) * 2 * np.pi,
                np.random.random(n_pulses)
            ])
            rabi, phase, periods_i = PulseEncoding.decode(p0)
            # to torch
            rabi_t  = torch.tensor(rabi, dtype=dtype_real, device=device)
            phase_t = torch.tensor(phase, dtype=dtype_real, device=device)
            dur_t   = torch.tensor(periods_i,     dtype=torch.int8,  device=device) 
            U_sequential = floquet_propagator_square_sequence_stroboscopic(
                rabi_t, phase_t, dur_t, **system_params, device=device
            )
            U_parallel = floquet_propagator_square_sequence_stroboscopic_vectorized(
                rabi_frequencies=rabi_t, 
                phases=phase_t, 
                pulse_durations_periods=dur_t, 
                **system_params, 
            )
            
            assert torch.allclose(
                U_sequential, U_parallel, atol=1e-6
            ), f"Full propagator mismatch for n_levels={n_levels}, n_pulses={n_pulses}"
                
                
def test_full_propagator_batched():
    """
    Compare the new batched propagator with the trusted sequential version.

    We draw B random chains that *share* the same integer pulse-duration
    pattern and make sure every individual chain gives the same unitary
    as the old per-chain routine.
    """
    dtype_real  = torch.float64           # keep the original precision
    B           = 8                       # chains per test-case (feel free to raise)

    for n_levels in n_levels_cases:
        system_params = get_system_params(n_levels)

        # translate to the signature expected by the propagators
        system_params["floquet_cutoff"] = system_params.pop("M")

        for n_pulses in n_pulses_cases:

            # integer durations are identical across the whole batch
            periods_np = np.random.randint(1, 6, size=n_pulses)
            dur_t = torch.tensor(periods_np,
                                 dtype=torch.int64,
                                 device=device)

            print(
                f"Testing n_levels={n_levels}, n_pulses={n_pulses}, "
                f"batch={B}"
            )

            # ---------- draw a batch of random continuous parameters -----
            rabi_batch  = torch.rand(
                B, n_pulses, dtype=dtype_real, device=device
            )
            phase_batch = torch.rand(
                B, n_pulses, dtype=dtype_real, device=device
            ) * 2 * np.pi

            # ---------- NEW: batched propagator --------------------------
            U_batch = floquet_propagator_square_sequence_batch(
                rabi=rabi_batch,
                phases=phase_batch,
                pulse_durations_periods=dur_t,
                **system_params,
            )                                   # shape (B, d, d)

            # ---------- reference: old, per-chain version ----------------
            for b in range(B):
                U_ref = floquet_propagator_square_sequence_stroboscopic(
                    rabi_batch[b],
                    phase_batch[b],
                    dur_t,
                    **system_params,
                    device=device,
                )

                assert torch.allclose(
                    U_ref, U_batch[b], atol=1e-6
                ), (
                    f"Mismatch: n_levels={n_levels}, n_pulses={n_pulses}, "
                    f"chain={b}"
                )

def test_full_propagator_batched_omega():
    """
    Compare the new ω‐batched propagator with the trusted sequential version.

    * Every chain shares the **same** Rabi / phase / integer-duration pattern.
    * Each chain has its **own** drive frequency ω_d.

    For every chain b we require
        U_batch[b]  ==  floquet_propagator_square_sequence_stroboscopic(ω_d=ω_d[b])
    up to a chosen tolerance.
    """
    dtype_real = torch.float64
    B          = 8               # number of different ω_d values per test-case

    for n_levels in n_levels_cases:
        system_params = get_system_params(n_levels)

        # ---- adapt dict to the new signatures --------------------------------
        system_params["floquet_cutoff"] = system_params.pop("M")
        energies  = system_params["energies"]
        couplings = system_params["couplings"]

        # base ω_d from the helper; we'll draw around it
        omega_base = system_params.pop("omega_d")

        for n_pulses in n_pulses_cases:

            periods_np = np.random.randint(1, 6, size=n_pulses)
            dur_t = torch.tensor(periods_np, dtype=torch.int64, device=device)

            # ---------- shared continuous parameters --------------------------
            rabi_t  = torch.rand(n_pulses, dtype=dtype_real, device=device)
            phase_t = torch.rand(n_pulses, dtype=dtype_real, device=device) * 2 * np.pi

            # ---------- NEW: batch of drive frequencies -----------------------
            # Spread ±20 % around the base value
            omega_batch = omega_base * (0.8 + 0.4 * torch.rand(B, dtype=dtype_real, device=device))

            # ---------- NEW: batched propagator call --------------------------
            U_batch = floquet_propagator_square_sequence_omega_batch(
                rabi=rabi_t,
                phases=phase_t,
                pulse_durations_periods=dur_t,
                omega_d_batch=omega_batch,
                **system_params,            # energies, couplings, floquet_cutoff
            )                                # shape (B, d, d)

            # ---------- reference: per-chain (sequential) propagator ----------
            for b in range(B):
                U_ref = floquet_propagator_square_sequence_stroboscopic(
                    rabi_t,
                    phase_t,
                    dur_t,
                    energies,
                    couplings,
                    omega_batch[b],
                    system_params["floquet_cutoff"],
                    device=device,
                )

                assert torch.allclose(
                    U_ref, U_batch[b], atol=1e-6
                ), (
                    f"Mismatch: n_levels={n_levels}, n_pulses={n_pulses}, "
                    f"chain={b}, ω_d={omega_batch[b].item():.4f}"
                )
                # optional progress print
                print(f"✓  n_levels={n_levels}  n_pulses={n_pulses}  chain={b}")
         
if __name__ == "__main__":
    with torch.no_grad():
        test_sombe_hamiltonian_and_folding()
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()
        test_full_propagator()
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()
        test_full_propagator_batched()
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()
        test_full_propagator_batched_omega()
    print("All tests passed successfully!")