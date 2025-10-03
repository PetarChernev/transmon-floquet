from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.transpiler import generate_preset_pass_manager
from qiskit.circuit import QuantumCircuit
import numpy as np

def print_rx_theta_info(backend_name: str, theta: float):
    # Initialize service and backend with fractional gates enabled
    service = QiskitRuntimeService()
    backend = service.backend(backend_name, use_fractional_gates=True)

    # Build a simple 1-qubit circuit with the fractional RX(θ)
    qc = QuantumCircuit(1)
    qc.rx(theta, 0)

    # Transpile with backend to embed gate into its Target and timing
    pm = generate_preset_pass_manager(optimization_level=3, backend=backend)
    t_qc = pm.run(qc)

    # Now inspect the backend.target instruction metadata
    target = backend.target
    inst = target.get_instruction("rx")

    if inst is None:
        print(f"No fractional RX gate found in backend '{backend_name}'.")
        return

    # Duration in units of backend.dt
    duration_dt = inst.duration
    dt = backend.dt  # in seconds

    print(f"Fractional RX({theta}) gate on backend '{backend_name}':")
    print(f"  Duration: {duration_dt} × dt")
    print(f"  dt = {dt * 1e9:.3f} ns → total duration ≈ {duration_dt * dt * 1e9:.3f} ns")

    # Additional info if available (such as error)
    if inst.error is not None:
        print(f"  Reported gate error (approximate): {inst.error:.3e}")
    if inst.u_channels:
        print(f"  Uses u-channels: {inst.u_channels}")
    if inst.unit:
        print(f"  Duration unit: {inst.unit}")

if __name__ == "__main__":
    backend_name = "ibm_torino"  # or any fractional-gate-enabled backend
    theta = np.pi / 4  # 45° rotation
    print_rx_theta_info(backend_name, theta)
