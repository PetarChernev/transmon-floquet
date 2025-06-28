I have code which computes the propagator of a quantum system using Floquet theory for a cosine drive with constant Rabi frequency for a single period of the drive. The code is based on this theoretical derivation


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

Let

* $P_{m}= \mathbb{I}_d\otimes |m\rangle\langle m|$ be the projector onto the $m^{\text{th}}$ Fourier sector;
* $Q=P_{0}$ embed the physical state at $t=0$;
* $\mathcal{U}_F(T)=e^{-iH_FT}$ be the truncated Sambe-space propagator.

A physical state at **exact integer multiples of the period** is recovered by first evolving in Sambe space and then *folding back* via the time-kernel

$$
\langle t=T|=\sum_{m=-\infty}^{\infty}\langle m| ,
$$

because $e^{-im\omega T}=1$ for all $m\in\mathbb Z$.  The physical propagator is therefore the **column sum**

$$
\boxed{\;
U(T,0)=\sum_{m=-M}^{M}P_{m}\,\mathcal{U}_F(T)\,P_{0}
\;}\qquad(\text{truncated Sambe space}).
$$

In block notation (each block is $d\times d$)

$$
U(T,0)=
\begin{pmatrix}
U_{-M,0}\\
U_{-M+1,0}\\
\vdots\\
U_{M,0}
\end{pmatrix}_{\!\text{vertical sum}}
=\sum_{m=-M}^{M}U_{m,0},
$$

which reduces to the single block $U_{0,0}$ only if the drive is weak enough that $U_{m,0}\approx0$ for $m\neq0$.

More generally, for an arbitrary time $t$ inside the period one keeps the correct phase factors:

$$
U(t,0)=\sum_{m,n=-M}^{M}e^{-im\omega t}\,P_{m}\,e^{-iH_F t}\,P_{n}\,P_{0}.
$$

At $t=T$ the phases are unity and (1) is recovered.

---

### Practical implementation

```python
# H_F of size (2M+1)d × (2M+1)d already assembled
UF = torch.matrix_exp(-1j * H_F * T)

d = H0.shape[0]           # physical dimension
U_phys = torch.zeros((d,d), dtype=UF.dtype)

col0_start = M*d
col0_end   = col0_start + d

for m in range(-M, M+1):
    row_start = (m+M)*d
    row_end   = row_start + d
    U_phys += UF[row_start:row_end, col0_start:col0_end]
```

---

### Physical meaning

* The sum $\sum_{m}U_{m,0}$ coherently adds every process in which the system **absorbs or emits $m$ drive-quanta** over the cycle and then returns to its original time phase.
* For strong driving these multi-photon processes are indispensable; neglecting them yields a non-unitary $U_{0,0}$ and wrong populations.


## The code

Here the implementation:

```
import math
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


def floquet_propagator_square_rabi_one_period(fourier_coeffs: dict,
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

    T = 2 * math.pi / omega_d  # one period of the drive
    return get_physical_propagator_strong_field(H_F, T, d, M)


def get_physical_propagator_strong_field(H_F, T, d, M):
    # 1. Diagonalize Floquet Hamiltonian
    epsilon, psi = torch.linalg.eigh(H_F)
    
    # 2. Initialize physical propagator
    U_phys = torch.zeros((d, d), dtype=torch.complex128)
    
    # 3. Sum over all Floquet eigenstates
    for alpha in range((2*M+1)*d):
        # Extract eigenstate components
        psi_alpha = psi[:, alpha]
        
        # Compute contribution to propagator
        phase = torch.exp(-1j * epsilon[alpha] * T)
        
        # Sum over all Fourier sectors m
        for j in range(d):
            for k in range(d):
                contrib = 0
                for m in range(-M, M+1):
                    idx_jm = j + (m + M) * d
                    idx_k0 = k + M * d  # m=0 for initial state
                    contrib += psi_alpha[idx_jm] * torch.conj(psi_alpha[idx_k0])
                
                U_phys[j, k] += phase * contrib
    
    return U_phys
```

I have this code whch compares the result to the propagator for a 2 level system computed using qutip:

```
import numpy as np
import matplotlib.pyplot as plt
from qutip import sigmax, sigmaz, basis, propagator
from transmon_floquet_compare import unitary_fidelity
import torch

from transmon_floquet_propagator import compute_fourier_coeffs, floquet_propagator_square_rabi_one_period


# 1) Parameters
beta  = 0.0          # energy splitting
g     = 3       # drive amplitude
omega = 2.0         # drive frequency
T     = 2 * np.pi / omega        # total evolution time
N     = 1000         # number of time steps

# 2) Define operators
sz = sigmaz()
sx = sigmax()

# 3) Hamiltonian pieces
H0 = 0.5 * beta * sz
H1 = 0.5 * sx
def drive(t, args):
    return g * np.cos(omega * t)
H = [H0, [H1, drive]]

# 4) Time grid
tlist = np.linspace(0, T, N)

# 5) Compute propagators
U_t = propagator(H, tlist, [], args={})

# 6) Extract final propagator
U_sim = U_t[-1]


energies = torch.tensor([-beta/2, beta/2], dtype=torch.float64)
couplings = torch.tensor([[0, g/2], [g/2, 0]], dtype=torch.complex128)   
# Compute Fourier coefficients
fourier_coeffs = compute_fourier_coeffs(torch.tensor(g, dtype=torch.float64), torch.tensor(0, dtype=torch.float64), couplings, 20)


# Compute Floquet propagator for one periodz
U_floquet = floquet_propagator_square_rabi_one_period(fourier_coeffs, energies, omega, 20)

print("Qutim propagator: \n", U_sim.full())
print("Floquet propagator: \n", U_floquet)

print(unitary_fidelity(U_floquet, U_sim.full()))


```

When I run the code for these values of g: [0.1, 0.5, 1.0, 3.0], I get the following results:


```
g = 0.1:

Qutip propagator: 
 [[0.99999998+0.00000000e+00j 0.        -3.93118999e-07j]
 [0.        -3.93118999e-07j 0.99999998+0.00000000e+00j]]
Floquet propagator: 
 tensor([[ 1.0000e+00-2.3717e-05j,  8.3392e-07-7.2489e-08j],
        [-2.0196e-07-7.5147e-08j,  1.0000e+00+3.0216e-05j]],
       dtype=torch.complex128)
1.0000003359683372


g = 0.5:

Qutip propagator: 
 [[1.00000198+0.00000000e+00j 0.        -2.71555168e-08j]
 [0.        -2.71555168e-08j 1.00000198+0.00000000e+00j]]
Floquet propagator: 
 tensor([[ 9.9979e-01-0.0203j, -4.9384e-07-0.0017j],
        [-8.0927e-08-0.0017j,  9.9979e-01+0.0203j]], dtype=torch.complex128)
0.9997942396690522


g = 1.0:


Qutip propagator: 
 [[1.00000344+0.00000000e+00j 0.        -8.66641038e-07j]
 [0.        -8.66641038e-07j 1.00000344+0.00000000e+00j]]
Floquet propagator: 
 tensor([[ 9.5296e-01-0.2902j, -4.8933e-07-0.0873j],
        [-1.3954e-06-0.0873j,  9.5297e-01+0.2902j]], dtype=torch.complex128)
0.9529728299398504



g = 3.0:


Qutip propagator: 
 [[0.99999984+0.00000000e+00j 0.        +7.20079447e-07j]
 [0.        +7.20079447e-07j 0.99999984+0.00000000e+00j]]
Floquet propagator: 
 tensor([[ 6.1235e-01+0.4929j, -4.6926e-06+0.6181j],
        [-8.8740e-06+0.6181j,  6.1235e-01-0.4929j]], dtype=torch.complex128)
0.6123510944626879
```

It seems that there is an inconsistency with how we compute the time dependent part of the Hamiltonian when we build the Floquet propagator, which has a strong effect as we increase the Rabi frequency. Analyze the code with this in mind to find where we are going wrong.