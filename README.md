The repo contains code to build the propagator of a pulse sequence for a 6-level transmon

The current implementation computes the propagator for a single period of the drive field.
The propagator for a sequence of pulses is to be implemented. It would be inder the restriction that a single pulse has a duration which is an integer number of the drive field period.


### File descriptions

transmon_dynamics.py - simulates a system by integraing the Schrodinger equation
transmon_floquet_propagator.py - computes the propagator for a single period of the drive field
transmon_floquet_compare.py - compares the propagator from the above file to a propagator reconstructed from the simulation
prompts/floquet_check_theory_prompt.md - contains the theory behind the Floquet computations