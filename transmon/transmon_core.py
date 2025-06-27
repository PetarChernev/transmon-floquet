import math
from matplotlib.pylab import eigh
import numpy as np
import torch


class TransmonCore:
    @staticmethod
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


    @staticmethod
    def compute_transmon_parameters(n_levels=6, n_charge=30, EJ_EC_ratio=50):
        """
        Numerically derive the transmon parameters following the paper's procedure.

        Returns:
        - energies: Energy eigenvalues
        - lambdas: Matrix elements λi,j = ⟨i|n̂|j⟩ in energy eigenbasis
        """
        # Construct Hamiltonian in charge basis
        H_charge = TransmonCore.construct_transmon_hamiltonian_charge_basis(n_charge, EJ_EC_ratio)

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
        
        if not [np.isclose(lambdas_full[i, i], 0) for i in range(n_levels)]:
            print("Warning: λ_ii are not all zero!")

        # For transmon, matrix elements should be real
        if np.max(np.abs(np.imag(lambdas_full))) > 1e-10:
            print(
                f"Warning: Imaginary parts found, max = {np.max(np.abs(np.imag(lambdas_full)))}"
            )

        return energies, np.real(lambdas_full)


    @staticmethod
    def find_EJ_EC_for_anharmonicity(target_anharmonicity=-0.0429, n_charge=30):
        """
        Find EJ/EC ratio that gives the target anharmonicity.
        Only needs first 3 energy levels to compute anharmonicity.
        """
        ratios = np.linspace(20, 200, 1000)
        anharms = []

        for ratio in ratios:
            H_charge = TransmonCore.construct_transmon_hamiltonian_charge_basis(n_charge, ratio)
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
        
        if np.abs(anharms[idx] - target_anharmonicity) > 1e-4:
            print(f"Warning: Target anharmonicity {target_anharmonicity} not achieved!")

        return ratios[idx]