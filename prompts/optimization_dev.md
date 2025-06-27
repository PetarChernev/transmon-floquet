## CMA

I have a function 

```
def floquet_propagator_square_sequence_stroboscopic(
    rabi_frequencies: torch.Tensor[float],
    phases: torch.Tensor[float],
    pulse_durations_periods: torch.Tensor[int],
    energies: torch.Tensor,
    lambdas_full: torch.Tensor,
    omega_d: float,
    floquet_cutoff: int,
    device: Optional[torch.device] = None
) -> torch.Tensor:
```

I want to CMA-ES to find combinations of rabi_frequencies, phases and pulse_durations_periods which optimiza the fidelity of the resulting propagator against a specific 2-level system propagator limited to the first 2 levels. Develop a script with the following functionality:
- wrap the floquet_propagator_square_sequence_stroboscopic function so that we have a function `(pulse_params: np.array[complex], pulse_durations_periods) -> np.array`. The pulse_params would represent both the real rabi frequencies (as real(pulse_params)) and the real pulse phases (as imag(pulse_params)). In this way we naturally constraint the phases in (0, 2pi). 
- define the loss function as (- unitarity - fidelity), where unitarity is the unitarity of the top `[:2, :2]` block of the result of the above function and the fidelity is the fidelity of the same block wrt a globally defined 2x2 matrix
- perform CMA-ES on this to find solutions which are perfectly unitary and have perfect fidelity, with tolerance 10-e6. You have to set up the CME is such a way that the pulse_durations_periods are constrained as integers
