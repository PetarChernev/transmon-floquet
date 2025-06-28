## Floquet Theory for Periodic Hamiltonians
### Setup
Given a periodic Hamiltonian with period $T$:
$$H(t) = \sum_{n=-\infty}^{\infty} C^{(n)} e^{in\omega t}$$
where $\omega = 2\pi/T$ and $C^{(n)}$ are the Fourier coefficient matrices.
Note that hermiticity requires: $C^{(n)} = [C^{(-n)}]^\dagger$
### Extended Hilbert Space
In Floquet theory, we work in an extended Hilbert space:
- Original space: $|j\rangle$ with $j = 0, 1, ..., d-1$ (d-dimensional)
- Extended space: $|j, m\rangle$ where $m \in \mathbb{Z}$ is the Fourier index
### The Floquet Hamiltonian
The Floquet Hamiltonian $H_F$ acts on the extended space and has the block structure:
$$H_F = \begin{pmatrix}
\ddots & \ddots & \ddots & & \\
\ddots & C^{(0)} - \omega & C^{(-1)} & C^{(-2)} & \ddots \\
\ddots & C^{(1)} & C^{(0)} & C^{(-1)} & \ddots \\
\ddots & C^{(2)} & C^{(1)} & C^{(0)} + \omega & \ddots \\
& \ddots & \ddots & \ddots & \ddots
\end{pmatrix}$$
More explicitly, the matrix elements are:
$$[H_F]_{(j,m),(k,n)} = C^{(m-n)}_{jk} + m\omega \delta_{mn}\delta_{jk}$$
where:
- $(j,m)$ labels the state $|j,m\rangle$
- $C^{(m-n)}_{jk}$ is the $(j,k)$ element of the Fourier coefficient matrix $C^{(m-n)}$
- The term $m\omega$ appears only on the diagonal
### Key Properties
1. $H_F$ is infinite-dimensional but has a block-Toeplitz structure
2. Each block is $d \times d$ (size of original Hilbert space)
3. The coupling between Fourier sectors $m$ and $n$ is given by $C^{(m-n)}$
### Practical Implementation
For numerical work, we truncate the Fourier index to $m \in \{-M, ..., M\}$, giving a finite $(2M+1)d \times (2M+1)d$ matrix.
## Index Mapping for Truncated Floquet Hamiltonian
### Basis Ordering
We arrange the basis states in lexicographic order:
$$|0,-M\rangle, |1,-M\rangle, ..., |d-1,-M\rangle, |0,-M+1\rangle, ..., |d-1,M\rangle$$
### Forward Mapping: $(j,m) \to a$
For a state $|j,m\rangle$ where:
- $j \in \{0, 1, ..., d-1\}$
- $m \in \{-M, -M+1, ..., M\}$
The single index is:
$$a = j + (m + M) \cdot d$$
Similarly for $(k,n) \to b$:
$$b = k + (n + M) \cdot d$$
### Inverse Mapping: $a \to (j,m)$
Given a single index $a$:
$$j = a \bmod d$$
$$m = \lfloor a/d \rfloor - M$$
### Explicit Matrix Element
The $(a,b)$ element of the $(2M+1)d \times (2M+1)d$ Floquet Hamiltonian is:
$$[H_F]_{a,b} = C^{(m-n)}_{jk} + m\omega \delta_{mn}\delta_{jk}$$
where:
- $j = a \bmod d$, $m = \lfloor a/d \rfloor - M$
- $k = b \bmod d$, $n = \lfloor b/d \rfloor - M$
- $C^{(m-n)}_{jk}$ is the $(j,k)$ element of the $(m-n)$-th Fourier coefficient matrix
- $\omega = 2\pi/T$
### Example
For $d=6$ (6-level transmon) and $M=2$:
- Total matrix size: $(2 \cdot 2 + 1) \cdot 6 = 30 \times 30$
- State $|2, -1\rangle$ maps to: $a = 2 + (-1 + 2) \cdot 6 = 2 + 6 = 8$
- State $|4, 1\rangle$ maps to: $b = 4 + (1 + 2) \cdot 6 = 4 + 18 = 22$
- Element $[H_F]_{8,22}$ would be $C^{(-1-1)}_{2,4} = C^{(-2)}_{2,4}$
This gives us the complete prescription for building the truncated Floquet Hamiltonian matrix.
## Concrete Hamiltonian
We work with Hamiltonian
$$H = H_0 + H_1(t)$$
## Fourier Decomposition of $H_0$
Since $H_0$ is constant in time:
$$H_0 = \sum_{i=0}^{2n_{\text{cut}}} \mu_i |i\rangle\langle i| = \sum_{i=0}^{2n_{\text{cut}}} (e_i - i\omega_d) |i\rangle\langle i|$$
The Fourier series of a constant function only has a zeroth-order term:
$$H_0 = C_0^{(0)} + \sum_{n \neq 0} C_0^{(n)} e^{in\omega t}$$
where:
- $C_0^{(0)} = H_0$ (the entire time-independent Hamiltonian)
- $C_0^{(n)} = 0$ for all $n \neq 0$
### Explicit Form
For our diagonal $H_0$:
$$C_0^{(0)} = \begin{pmatrix}
\mu_0 & 0 & 0 & \cdots \\
0 & \mu_1 & 0 & \cdots \\
0 & 0 & \mu_2 & \cdots \\
\vdots & \vdots & \vdots & \ddots
\end{pmatrix}$$
$$C_0^{(n)} = 0 \text{ for all } n \neq 0$$
So the time-independent part only contributes to the diagonal blocks ($m = n$) of the Floquet Hamiltonian through the $C^{(0)}$ term.
## Fourier Decomposition of $H_1(t)$
### Revised derivation of the Fourier coefficients for the drive $H_1(t)$
---
#### 1.  Starting point
In the rotating frame the matrix elements of the driven term are
$$
\boxed{%
\bigl[H_1(t)\bigr]_{jl}\;=\;
\Omega\cos\!\bigl(\omega_dt+\phi\bigr)\;
\Bigl[\lambda_{jl}\,e^{\,i(j-l)\omega_dt}
      +\lambda_{lj}\,e^{-i(j-l)\omega_dt}\Bigr]
}
\tag{A-1}
$$
Because $\cos x=\tfrac12(e^{ix}+e^{-ix})$, each matrix element contains **two** extra side-band factors $e^{\pm i\omega_dt}$.  After multiplication, four distinct Fourier frequencies appear.
---
#### 2.  Separate the genuinely independent pieces
Hermiticity demands
$$
C_1^{(n)}=\bigl[C_1^{(-n)}\bigr]^{\!\dagger}\!.
\tag{A-2}
$$
Therefore we only need to tabulate **one** of each Hermitian pair; the other is fixed automatically.
A convenient choice is to keep the coefficients with **positive** Fourier index $n>0$ and reconstruct the negative-$n$ partners via (A-2).
---
#### 3.  Fourier coefficients for $j\neq l$
Define the level difference
$$
\Delta_{jl}\equiv j-l\;\;(\text{may be negative}).
$$
From (A-1) the exponents are $(\Delta_{jl}\pm1)\omega_dt$.  Collecting terms with the same frequency gives
$$
\boxed{%
\begin{aligned}
\bigl[C_1^{(\,\Delta_{jl}+1\,)}\bigr]_{jl}&=\frac{\Omega}{2}\,
\lambda_{jl}\,e^{\,i\phi},\\[4pt]
\bigl[C_1^{(\,\Delta_{jl}-1\,)}\bigr]_{jl}&=\frac{\Omega}{2}\,
\lambda_{jl}\,e^{-i\phi}.
\end{aligned}}
\tag{A-3}
$$
All other $C_1^{(n)}$ with $n>0$ are zero in the $(j,l)$ entry.
The negative-$n$ blocks follow from (A-2).
---
#### 4.  Special case $j=l$
When $j=l$ there is no $e^{\pm i\Delta_{jl}\omega_dt}$ phase factor and the only remaining frequencies are $n=\pm1$:
$$
[C_1^{(\pm1)}]_{jj}=\frac{\Omega}{2}\,\lambda_{jj}\,e^{\pm i\phi}.
\tag{A-4}
$$
For a transmon‐type coupling one usually has $\lambda_{jj}=0$, so **diagonal drive terms disappear altogether**; include them only if your model keeps finite $\lambda_{jj}$.
---
#### 5.  Compact summary
Putting everything together:
$$
\boxed{%
[C_1^{(n)}]_{jl}=
\begin{cases}
\dfrac{\Omega}{2}\,\lambda_{jl}\,e^{\,i\phi}, & n=\Delta_{jl}+1,\\[6pt]
\dfrac{\Omega}{2}\,\lambda_{jl}\,e^{-i\phi}, & n=\Delta_{jl}-1,\\[6pt]
0, & \text{otherwise,}
\end{cases}}
\qquad n>0.
\tag{A-5}
$$
Coefficients with $n<0$ are fixed by Hermiticity, cf. (A-2).
Equation (A-5) avoids double counting and stores the minimal set of independent Fourier blocks; numerical implementations only need to loop over those non-zero $(n,j,l)$ combinations.
With this refinement the coefficient table is half the size, **guaranteed Hermitian**, and makes the vanishing of diagonal drive terms explicit when $\lambda_{jj}=0$.
## Extracting the Physical Propagator from $H_F$
### The Extended Space Evolution
In the extended Floquet space, the evolution operator is:
$$U_F(\tau) = e^{-iH_F \tau}$$
### Physical Propagator Extraction
The physical propagator $U(T,0)$ in the original $d$-dimensional space is obtained by:
1. **Computing the extended propagator**: $U_F(T) = e^{-iH_F T}$
2. **Extracting the central block**: The physical propagator is the $d \times d$ block corresponding to transitions from Fourier sector $m=0$ to $n=0$
### Explicit Extraction
If we order our basis as described earlier with indices $a = j + (m+M) \cdot d$, then:
$$[U(T,0)]_{jk} = [U_F(T)]_{a,b}$$
where:
- $a = j + M \cdot d$ (corresponding to state $|j,0\rangle$)
- $b = k + M \cdot d$ (corresponding to state $|k,0\rangle$)
### In Matrix Form
If we partition $U_F(T)$ into $(2M+1) \times (2M+1)$ blocks of size $d \times d$:
$$U_F(T) = \begin{pmatrix}
U_{-M,-M} & U_{-M,-M+1} & \cdots & U_{-M,M} \\
U_{-M+1,-M} & U_{-M+1,-M+1} & \cdots & U_{-M+1,M} \\
\vdots & \vdots & \ddots & \vdots \\
U_{M,-M} & U_{M,-M+1} & \cdots & U_{M,M}
\end{pmatrix}$$
Then the physical propagator is:
$$U(T,0) = U_{0,0}$$
This is the central $d \times d$ block.
### Implementation Note
In practice:
```python
# Compute extended propagator
U_F = torch.matrix_exp(-1j * H_F * T)
# Extract central block (assuming M Fourier modes on each side)
start_idx = M * d
end_idx = (M + 1) * d
U_physical = U_F[start_idx:end_idx, start_idx:end_idx]
```
This gives us the propagator for one period in the original transmon space.

## Implementation check
I have the following code which uses Floquet theory to estimate the propagator of a single period of the drive field for a 6 level system transmon system in the strong field regime:

```
import torch
import numpy as np
def compute_fourier_coeffs(rabi: float,
                           phase: float,
                           couplings: torch.Tensor,
                           floquet_cutoff: int) -> dict:
    """
    Compute the Fourier coefficient matrices C^{(n)} for the drive Hamiltonian H1(t),
    including static (n=0) drive pieces and enforcing Hermiticity.
    """
    d = couplings.shape[0]
    M = floquet_cutoff
    # Initialize coefficients for n in [-M..M]
    C = {n: torch.zeros((d, d), dtype=torch.cfloat) for n in range(-M, M+1)}
    half_rabi = 0.5 * rabi
    exp_pos = torch.exp(1j * phase)
    exp_neg = torch.exp(-1j * phase)
    for j in range(d):
        for l in range(d):
            lam_fwd = couplings[j, l]        #  λ_{jl}
            lam_bwd = couplings[l, j]        #  λ_{lj}  ( = λ_{jl}* for a transmon)
            if lam_fwd == 0 and lam_bwd == 0:
                continue
            Δ = j - l                       # level difference
            # ---------- forward part  λ_{jl} ----------------------------------
            for offset, phase_factor in ((+1, exp_pos), (-1, exp_neg)):
                n = Δ + offset
                if -M <= n <= M:
                    C[n][j, l] += half_rabi * lam_fwd * phase_factor
            # ---------- backward part λ_{lj} ----------------------------------
            for offset, phase_factor in ((+1, exp_pos), (-1, exp_neg)):
                n = -Δ + offset             # NOTE the minus sign!
                if -M <= n <= M:
                    C[n][j, l] += half_rabi * lam_bwd * phase_factor
    # Enforce Hermiticity
    # for n in range(1, M+1):
    #     C[-n] = C[n].conj().T
    return C
def floquet_hamiltonian_const_rabi_period(fourier_coeffs: dict,
                                          energies: torch.Tensor,
                                          omega_d: float,
                                          floquet_cutoff: int) -> torch.Tensor:
    """
    Build truncated Floquet Hamiltonian in the rotating frame and compute one-period propagator.
    """
    d = energies.numel()
    M = floquet_cutoff
    # Prepare zero block
    zero_block = torch.zeros((d, d), dtype=torch.cfloat)
    # Collect all Fourier blocks (drive + H0 later)
    C_total = {n: fourier_coeffs.get(n, zero_block.clone())
               for n in range(-M, M+1)}
    # Static H0 in rotating frame: diag(energies - j*omega_d)
    levels = torch.arange(d, dtype=energies.dtype)
    H0_rot = energies - levels * omega_d
    C_total[0] = C_total[0] + torch.diag(H0_rot).to(torch.cfloat)
    # Assemble Floquet Hamiltonian
    N = (2*M + 1) * d
    H_F = torch.zeros((N, N), dtype=torch.cfloat)
    for m in range(-M, M+1):
        row = (m + M) * d
        for n in range(-M, M+1):
            col = (n + M) * d
            idx = m - n
            block = C_total.get(idx, zero_block)
            H_F[row:row + d, col:col + d] = block
        H_F[row:row + d, row:row + d] += m * omega_d * torch.eye(d, dtype=torch.cfloat) 
    # Propagator for one period T = 2π/ω_d
    T = 2 * np.pi / omega_d
    U_F = torch.linalg.matrix_exp(-1j * H_F * T)
    # Extract central block (m=0→n=0)
    start = M * d
    U_physical = U_F[start:start + d, start:start + d]
    return U_physical
```

I have the following code which simulates the system by numericaly intergrating Schoridnger's equation:

```
import numpy as np
import scipy as sp
from scipy.linalg import expm
import matplotlib.pyplot as plt
from transmon_core import TransmonCore
def *base*envelope(t, pulse_idx, pulse_duration, pulse_type="square"):
    """Return the *unit* envelope (0...1) for a single pulse at instant t."""
    if pulse_type == "square":
        t_start = pulse_idx * pulse_duration
        t_end = (pulse_idx + 1) * pulse_duration
        return 1.0 if t_start <= t <= t_end else 0.0
    elif pulse_type == "gaussian":
        t_center = (pulse_idx + 0.5) * pulse_duration
        sigma = pulse_duration / 6
        return np.exp(-((t - t_center) ** 2) / (2 * sigma ** 2))
    else:
        raise ValueError(f"Unknown pulse_type '{pulse_type}'.")
def drive_envelope_array(times, rabi_frequencies, pulse_duration, pulse_type="square"):
    """Vectorised version that multiplies the unit envelope by the pulse-specific
    Rabi frequency.  Returns an array of Ω_R(t) in the same units as
    `rabi_frequencies`."""
    env = np.zeros_like(times, dtype=float)
    n_pulses = len(rabi_frequencies)
    for i, t in enumerate(times):
        pulse_idx = min(int(t // pulse_duration), n_pulses - 1)
        env[i] = (
            rabi_frequencies[pulse_idx]
            * *base*envelope(t, pulse_idx, pulse_duration, pulse_type)
        )
    return env
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Dynamics
# -----------------------------------------------------------------------------
def simulate_transmon_dynamics(
    initial_state,
    rabi_frequencies,
    phases,
    energies,
    couplings,
    *,
    n_levels=6,
    total_time=20.0,
    n_time_steps=2000,
    pulse_type="square",
    use_rwa=True,
):
    """
    Simulate using equation (4) or the full Eq. (A13) depending on `use_rwa`.
    """
    # Extract λj = λ_{j,j-1} (nearest neighbours) only if RWA is used
    if use_rwa:
        lambdas = np.zeros(n_levels)
        for j in range(1, n_levels):
            lambdas[j] = couplings[j, j - 1]
    # Time evolution
    n_pulses = len(rabi_frequencies)
    pulse_duration = total_time / n_pulses
    times = np.linspace(0, total_time, n_time_steps + 1)
    dt = times[1] - times[0]
    states_history = np.zeros((len(times), n_levels), dtype=complex)
    states_history[0] = initial_state
    current_state = initial_state.astype(complex).copy()
    for i, t in enumerate(times[:-1]):
        pulse_idx = min(int(t // pulse_duration), n_pulses - 1)
        # Pulse envelope
        envelope = *base*envelope(t, pulse_idx, pulse_duration, pulse_type)
        # Hamiltonian (equation 4 or A13) - MUST BE COMPLEX!
        H = np.zeros((n_levels, n_levels), dtype=complex)
        # Diagonal terms
        for j in range(n_levels):
            H[j, j] = energies[j] - j  # μj = ej - j
        omega_R = rabi_frequencies[pulse_idx] * envelope
        if use_rwa and omega_R != 0:
            omega_R *= np.exp(-1j * phases[pulse_idx])
            for j in range(1, n_levels):
                H[j, j - 1] += lambdas[j] * omega_R / 2
                H[j - 1, j] += lambdas[j] * np.conj(omega_R) / 2
        elif not use_rwa and omega_R != 0:
            omega_d = 1.0
            # build matrix of level-differences Δ_{jl} = j - l
            levels = np.arange(n_levels)
            delta = levels[:, None] - levels[None, :]
            # two sideband factors: e^{±i Δ ω_d t}
            e_pos = np.exp(1j * delta * omega_d * t)
            e_neg = np.exp(-1j * delta * omega_d * t)
            # drive coupling piece: λ_{jl} e^{iΔωt} + λ_{lj} e^{-iΔωt}
            H_drive = couplings * e_pos + couplings.T * e_neg
            # time-dependent envelope: (Ω) cos(ω_d t + φ)
            drive_factor = omega_R * np.cos(omega_d * t + phases[pulse_idx])
            # add all four sidebands at once
            H += drive_factor * H_drive
        # Time evolution
        U = expm(-1j * H * dt)
        current_state = U @ current_state
        states_history[i + 1] = current_state
    return current_state, states_history, times
```

I have this code which compares the propagator from Floquet theory to the propagator reconstructed from simulating the dynamics of the system for a single period of the drive field:

```
import math
import numpy as np
import torch
from transmon_core import TransmonCore
from transmon_dynamics import simulate_transmon_dynamics
from transmon_floquet_propagator import compute_fourier_coeffs, floquet_hamiltonian_const_rabi_period
_simulate = None
_dim = None
def *init*worker(simulate, dim):
    global *simulate, *dim
    *simulate, *dim = simulate, dim
def *run*simulation(j):
    e_j = np.zeros(_dim, dtype=complex)
    e_j[j] = 1.0
    return *simulate(e*j)
def estimate_unitary(simulate, dim):
    """
    Reconstructs the unitary (up to a global phase) implemented by
    the black-box `simulate(initial_state)`.
    Assumes:
      * `simulate` takes a size-d 1-D complex array (|ψ_in⟩)
      * returns the size-d 1-D complex array (|ψ_out⟩)
      * Dynamics is closed and therefore unitary.
    """
    # Work out Hilbert-space dimension from a single call
    U_est = np.zeros((dim, dim), dtype=complex)
    # Standard basis {|0⟩, |1⟩, ...}
    for j in range(dim):
        e_j = np.zeros(dim, dtype=complex)
        e_j[j] = 1.0
        ψ_out = simulate(e_j)
        U_est[:, j] = ψ_out      # each output column is U|j⟩
    # Remove a global phase (optional but usually convenient)
    phase = np.exp(-1j * np.angle(U_est[0, 0]))
    return phase * U_est
def estimate_transmon_unitary(*args, **kwargs):
    def *f(initial*state):
        final_state, *, * = simulate_transmon_dynamics(
            initial_state,
            *args,
            **kwargs
        )
        return final_state
    return estimate_unitary(_f, n_levels)
def unitary_fidelity(U_1, U_2):
    if torch.is_tensor(U_1):
        U_1 = U_1.detach().cpu().numpy()
    if torch.is_tensor(U_2):
        U_2 = U_2.detach().cpu().numpy()
    d = U_1.shape[0]
    return abs(np.trace(U_2.conj().T @ U_1)) / d
if **name** == "__main__":
    dev, dtype = "cuda", torch.complex128
    n_levels = 6
    # From Table I - complete population transfer
    rabi_frequencies = np.array([1])
    phases = np.array([.2345]) * np.pi
    total_time = 2 * np.pi 
    delta = -0.0429
    
    
    EJ_EC_ratio = TransmonCore.find_EJ_EC_for_anharmonicity(delta)
    energies, couplings = TransmonCore.compute_transmon_parameters(
        n_levels, n_charge=30, EJ_EC_ratio=EJ_EC_ratio
    )
    unitary_est = estimate_transmon_unitary(
        rabi_frequencies,
        phases,
        energies,
        couplings,
        n_levels=n_levels,
        total_time=total_time,
        pulse_type="square",
        n_time_steps=10000,
        use_rwa=False,
    )
    
    
    rabi_frequencies = torch.tensor(rabi_frequencies, dtype=torch.float64, requires_grad=True)
    phases = torch.tensor(phases, dtype=torch.float64, requires_grad=True)
    
    
    energies = torch.tensor(energies, dtype=torch.float64)
    couplings = torch.tensor(couplings, dtype=torch.complex128)   
    # Compute Fourier coefficients
    fourier_coeffs = compute_fourier_coeffs(rabi_frequencies[0], phases[0], couplings, 100)
    # Compute Floquet propagator for one period
    U = floquet_hamiltonian_const_rabi_period(fourier_coeffs, energies, 1, 100)
    
    print(f"Fidelity: {unitary_fidelity(unitary_est, U)}")
```

I get
Fidelity: 0.9358950223580421
This is not affected in a meaningful way by changing the number of time steps used in the integration or the Fourier cutoff, so it does not appear to be an error due to precision. 
When I change the phase used to

```
    phases = np.array([1.5]) * np.pi
```

I get 
Fidelity: 0.7681768144045015
When I instead reduce the Rabi frequency to

```
    rabi_frequencies = np.array([.1])
```

I get 
Fidelity: 0.9950550682077361
When I change both the phase and the Rabi frequency to

```
phases = np.array([1.5]) * np.pi
rabi_frequencies = np.array([.1])
```
I get
Fidelity: 0.9911680907392192

It appears that the strong field terms are the main contributors to the discrepancy. 
Analyze both pieces code. Are there any discepancies, in particular relating to the the fast oscilating terms? Are there any place were you can identify a mismatch between the two implementations or between any one of the implementation and the theory?