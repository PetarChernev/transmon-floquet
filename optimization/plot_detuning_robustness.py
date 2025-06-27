import numpy as np
import matplotlib.pyplot as plt

# Assuming rabi_frequencies, phases, pulse_durations, epsilon, lambda_matrix, and options 
# are already defined in your environment

# Sweep drive frequency ω_d from 0.95 to 1.05
omega_values = np.linspace(0.95, 1.05, 500)
probabilities = []

for omega in omega_values:
    # Compute propagator for this drive frequency
    U = compute_propagator_sequence_qutip(
        rabi_frequencies, 
        phases, 
        pulse_durations, 
        epsilon, 
        lambda_matrix, 
        omega, 
        options
    )
    # Convert to NumPy array if it's a Qobj
    U_mat = np.array(U)
    # Extract probability of |0> → |1>
    probabilities.append(np.abs(U_mat[1, 0])**2)

# Plot the transfer probability versus drive frequency
plt.figure()
plt.plot(omega_values, probabilities)
plt.xlabel('Drive frequency ω_d')
plt.ylabel('Transfer probability |⟨1|U|0⟩|^2')
plt.title('Transfer probability vs. drive frequency')
plt.show()
