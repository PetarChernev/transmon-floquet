import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Define Pauli matrices
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

# Hamiltonian in RWA for a given Rabi frequency Omega and phase phi
# H = (Omega/2) * (sigma_x * cos(phi) + sigma_y * sin(phi))
def H_rwa(omega, phi):
    return 0.5 * omega * (sigma_x * np.cos(phi) + sigma_y * np.sin(phi))

# Time derivative: dpsi/dt = -i H psi
def schrodinger_rhs(t, psi, H):
    return -1j * H.dot(psi)

# Define sequence of pulses: list of (Omega, phi)
# Modify this list with your desired Rabi frequencies and phases
phi = [6.168583, 2.075657, 0.878826, 2.576394, 5.741978, 4.636668, 6.955457, 5.046248, 5.458592, 1.572531, 0.618, 4.36956, 0]
areas = [np.pi/2] * len(phi)
pulse_sequence = zip(areas, phi)
pulse_duration = 1.0  # Duration of each pulse
points_per_pulse = 200  # Time resolution within each pulse

# Prepare arrays to store time and state
times = []
states = []

# Initial state: ground state |0> = [1, 0]
psi0 = np.array([1.0, 0.0], dtype=complex)
current_time = 0.0

# Loop over pulses
total_points = 0
for omega, phi in pulse_sequence:
    H = H_rwa(omega, phi)
    t_span = (current_time, current_time + pulse_duration)
    t_eval = np.linspace(current_time,
                         current_time + pulse_duration,
                         points_per_pulse,
                         endpoint=True)
    
    # Integrate Schrödinger equation for this pulse
    sol = solve_ivp(schrodinger_rhs,
                   t_span,
                   psi0,
                   args=(H,),
                   t_eval=t_eval,
                   atol=1e-9,
                   rtol=1e-7)
    
    # Append results
    times.extend(sol.t)
    states.extend(sol.y.T)
    
    # Update for next pulse
    psi0 = sol.y[:, -1]
    current_time += pulse_duration
    total_points += len(t_eval)

# Convert to arrays
times = np.array(times)
states = np.array(states)  # shape (N, 2)

# Compute probabilities
probs_0 = np.abs(states[:, 0])**2
probs_1 = np.abs(states[:, 1])**2

# Plot dynamics
plt.figure(figsize=(8, 4))
plt.plot(times, probs_0, label=r'$|c_0|^2$')
plt.plot(times, probs_1, label=r'$|c_1|^2$')
plt.xlabel('Time')
plt.ylabel('Probability')
plt.title('Two-level system dynamics under pulse sequence (RWA)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()