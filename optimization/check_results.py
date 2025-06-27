import itertools
import numpy as np
import torch
from transmon.transmon_core import TransmonCore
from transmon.transmon_dynamics_qutip import pulse_sequence_qutip
from transmon.transmon_floquet_propagator import floquet_propagator_square_sequence_stroboscopic


if __name__ == "__main__":
    device = 'cuda'
    omega_d = 1

    target = np.array([[1, 1], [1, -1]]) / np.sqrt(2)

    n_levels = 6
    EJ_EC_ratio = TransmonCore.find_EJ_EC_for_anharmonicity(-0.0429)
    energies, lambdas_full = TransmonCore.compute_transmon_parameters(
        n_levels, n_charge=30, EJ_EC_ratio=EJ_EC_ratio
    )
    energies = torch.tensor(energies, dtype=torch.float64, device=device)
    lambdas_full = torch.tensor(lambdas_full, dtype=torch.complex128, device=device)
    omega_d       = 1.0                                 # driving frequency (rad s⁻¹)
    floquet_cutoff: int = 40

    test_params = dict(
        rabi_frequencies=torch.tensor([0.95107027, 0.79220659, 1.0358085,  1.05042276, 1.0133318,  0.87797412,
 1.02144238, 0.64034026], dtype=torch.float64, device=device),
        phases=torch.tensor([2.15531313, 4.82939254, 3.9763329,  1.28857258, 4.64092377, 3.82731991,
 1.45256058, 4.74845489], dtype=torch.float64, device=device),
        energies=energies,
        lambdas_full=lambdas_full,
        omega_d=torch.tensor([omega_d], dtype=torch.float64, device=device),
        floquet_cutoff=floquet_cutoff 
    )

    time_pulse_durations = [7, 1, 7, 1, 7, 1, 7, 1,]

    U_general = floquet_propagator_square_sequence_stroboscopic(
        **test_params,
        pulse_durations_periods=time_pulse_durations, 
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
            "nsteps": 200000
        }
    )


    # collect all propagators
    Us = {
        # "strob":               U_strob,
        "general":             U_general[:2, :2],
        "qutip_general":       U_qutip_general[:2, :2],
        "target": target
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

    print("Qutip Unitary: ")
    print(U_qutip_general)
