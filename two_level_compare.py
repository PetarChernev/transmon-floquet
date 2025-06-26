import numpy as np
import matplotlib.pyplot as plt
from qutip import sigmax, sigmaz, basis, propagator, Qobj
from transmon_floquet_compare import unitary_fidelity
import torch

from transmon_floquet_propagator import compute_fourier_coeffs, floquet_propagator_square_rabi_one_period


# ------------------------------------------------------------------
# 1)  Basic parameters ---------------------------------------------
# ------------------------------------------------------------------
omega_01   = 1.0            # bare |0⟩→|1⟩ transition (rad · units⁻¹)
omega_d    = 2.0            # drive frequency
delta     = omega_01 - omega_d       # detuning that survives in the rotating frame
Ω0    = 12            # peak Rabi rate  (  Ω(t)  below )
φ     = 0.0            # drive phase
T     = 2*np.pi/omega_d     # one drive period
N     = 1000           # number of time steps

# ------------------------------------------------------------------
# 2)  Operators for a two-level (truncated) transmon ---------------
# ------------------------------------------------------------------
ket0, ket1 = basis(2, 0), basis(2, 1)
proj0, proj1 = ket0*ket0.dag(), ket1*ket1.dag()

σ_plus  = ket1*ket0.dag()     # |1⟩⟨0|   (raises excitation)
σ_minus = ket0*ket1.dag()     # |0⟩⟨1|   (lowers excitation)

# ------------------------------------------------------------------
# 3)  Static part in the drive-rotating frame  ---------------------
#     H_det = (e₁ - omega_d) |1⟩⟨1|  = delta |1⟩⟨1|
# ------------------------------------------------------------------
H_det = delta * proj1

# ------------------------------------------------------------------
# 4)  Time-dependent drive pieces (outside the RWA) ----------------
#     Ω(t)/2 · [e^{-iφ}σ⁺ + e^{iφ}σ⁻]                    resonant part
#            + Ω(t)/2 · [e^{ i(2omega_d t+φ)}σ⁺ + e^{-i(2omega_d t+φ)}σ⁻]   counter-rot.
# ------------------------------------------------------------------
def Ω(t, args=None):          # envelope; insert your own pulse here
    return Ω0                 # constant for this example

def c_plus(t, args):
    return 0.5*Ω(t)*( np.exp(-1j*φ) + np.exp( 1j*(2*omega_d*t + φ)) )

def c_minus(t, args):
    return 0.5*Ω(t)*( np.exp( 1j*φ) + np.exp(-1j*(2*omega_d*t + φ)) )

# Hamiltonian list for QuTiP
H = [
     H_det,                           # static frame-transformed term
    [σ_plus,  c_plus],                # σ⁺ pieces (both resonant & CR)
    [σ_minus, c_minus]                # σ⁻ pieces (both resonant & CR)
]

# ------------------------------------------------------------------
# 5)  Propagate one period -----------------------------------------
# ------------------------------------------------------------------
tlist = np.linspace(0, T, N)
U_list = propagator(H, tlist, [], args={})   # U(t_k) for each k
U_sim    = U_list[-1]                          # total unitary after one T




# 6) Set up Floquet calculation
# For a transmon, the energies are [0, ω_01]
energies = torch.tensor([0.0, omega_01], dtype=torch.float64)

# The coupling matrix elements for (a + a†) are:
# lambdas[0,1] = ⟨0|a + a†|1⟩ = 1
# lambdas[1,0] = ⟨1|a + a†|0⟩ = 1
lambdas_full = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
# Compute Fourier coefficients
fourier_coeffs = compute_fourier_coeffs(torch.tensor(Ω0, dtype=torch.float64), torch.tensor(0, dtype=torch.float64), lambdas_full, 20)


# Compute Floquet propagator for one periodz
U_floquet_torch = floquet_propagator_square_rabi_one_period(
    fourier_coeffs, energies, omega_d, 20
)

# --- 2. Convert that tensor to a NumPy array ---
U_floquet = U_floquet_torch.detach().cpu().numpy()   # shape (2, 2), dtype=complex128
                  # identity (R†(0))


# --- 5. Print / compare ---
print("QuTiP propagator:\n", U_sim.full())   # U_sim.full() is already NumPy
print("Floquet propagator (NumPy):\n", U_floquet)
print(unitary_fidelity(U_floquet, U_sim.full()))

