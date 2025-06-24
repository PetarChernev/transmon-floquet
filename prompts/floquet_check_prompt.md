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
H = H_0 + H_1(t)

## Fourier Decomposition of $H_0$

Since $H_0$ is constant in time:
$$H_0(t) = \sum_{i=0}^{2n_{\text{cut}}} \mu_i |i\rangle\langle i|$$

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

Given:
$$[H_1(t)]_{jl} = \frac{\Omega}{2} \cos(\omega_d t + \phi) \left[\lambda_{jl} e^{i(j-l)\omega_d t} + \lambda_{lj} e^{-i(j-l)\omega_d t}\right]$$

### Step 1: Expand the cosine
$$\cos(\omega_d t + \phi) = \frac{1}{2}[e^{i(\omega_d t + \phi)} + e^{-i(\omega_d t + \phi)}]$$

### Step 2: Substitute and expand
$$[H_1(t)]_{jl} = \frac{\Omega}{4} [e^{i(\omega_d t + \phi)} + e^{-i(\omega_d t + \phi)}] \left[\lambda_{jl} e^{i(j-l)\omega_d t} + \lambda_{lj} e^{-i(j-l)\omega_d t}\right]$$

Expanding the product:
$$[H_1(t)]_{jl} = \frac{\Omega}{4} \left[\lambda_{jl} e^{i\phi} e^{i(j-l+1)\omega_d t} + \lambda_{jl} e^{-i\phi} e^{i(j-l-1)\omega_d t} + \lambda_{lj} e^{i\phi} e^{i(l-j+1)\omega_d t} + \lambda_{lj} e^{-i\phi} e^{i(l-j-1)\omega_d t}\right]$$

### Step 3: Identify Fourier coefficients
Since $H_1(t) = \sum_n C_1^{(n)} e^{in\omega_d t}$, we need to collect terms by their frequency:

For coefficient $C_1^{(n)}$, the $(j,l)$ element receives contributions from terms with frequency $n\omega_d$:

$$[C_1^{(n)}]_{jl} = \begin{cases}
\frac{\Omega}{4} \lambda_{jl} e^{i\phi} & \text{if } n = j-l+1 \\
\frac{\Omega}{4} \lambda_{jl} e^{-i\phi} & \text{if } n = j-l-1 \\
\frac{\Omega}{4} \lambda_{lj} e^{i\phi} & \text{if } n = l-j+1 \\
\frac{\Omega}{4} \lambda_{lj} e^{-i\phi} & \text{if } n = l-j-1 \\
0 & \text{otherwise}
\end{cases}$$

### Compact Form
More compactly:
$$[C_1^{(n)}]_{jl} = \frac{\Omega}{4} \left[\lambda_{jl} e^{i\phi} \delta_{n,j-l+1} + \lambda_{jl} e^{-i\phi} \delta_{n,j-l-1} + \lambda_{lj} e^{i\phi} \delta_{n,l-j+1} + \lambda_{lj} e^{-i\phi} \delta_{n,l-j-1}\right]$$

### Key Observations
1. Each $(j,l)$ element contributes to at most 4 different Fourier components
2. The non-zero Fourier indices are: $n \in \{j-l \pm 1, l-j \pm 1\}$

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

This gives us the propagator for one period in the original transmon space!

## Task

Your task is to check the code below. Make sure that it implements the mathematical operations defined above properly. Be thorough - check every line and try to match it to the equations. Make sure that the matrices are properly structured.

Here's the code:


import torch
import math


def compute_fourier_coeffs(rabi, phase, couplings, fourier_cutoff):
    """
    Compute the Fourier coefficient matrices C^{(n)} for the drive Hamiltonian H1(t).
    Args:
        rabi (float): Rabi frequency of the drive.
        phase (float): Phase of the drive.
        couplings (torch.Tensor): Coupling strengths matrix \lambda_{jl}, shape (d, d).
        fourier_cutoff (int): Cutoff M for Fourier indices (n \in [-M, M]).
    Returns:
        dict: Mapping n -> C^{(n)} (torch.Tensor of shape (d, d), complex dtype).
    """
    d = couplings.shape[0]
    M = fourier_cutoff
    # Initialize Fourier blocks
    C = {n: torch.zeros((d, d), dtype=torch.cfloat) for n in range(-M, M+1)}
    coeff = rabi / 4.0
    exp_ip = torch.exp(1j * phase)
    exp_im = torch.exp(-1j * phase)
    # Loop over matrix elements
    for j in range(d):
        for l in range(d):
            lam_jl = couplings[j, l]
            lam_lj = couplings[l, j]
            # Four possible contributions
            for n, lam, exp_val in [(j - l + 1, lam_jl, exp_ip),
                                     (j - l - 1, lam_jl, exp_im),
                                     (l - j + 1, lam_lj, exp_ip),
                                     (l - j - 1, lam_lj, exp_im)]:
                if -M <= n <= M:
                    C[n][j, l] += coeff * lam * exp_val
    return C


def floquet_hamiltonian_square_period(fourier_coeffs, energies, omega_d, fourier_cutoff):
    """
    Compute the Floquet propagator U(T) for one period of a periodic drive using the truncated Floquet Hamiltonian.

    Args:
        fourier_coeffs (dict): Mapping n -> C^{(n)} Fourier coefficient matrices of shape (d, d).
        energies (torch.Tensor): Energy levels \mu_i of the unperturbed Hamiltonian H0, shape (d,).
        omega_d (float): Drive frequency (\omega_d = 2\pi/T).
        fourier_cutoff (int): Fourier index cutoff M.

    Returns:
        torch.Tensor: Floquet propagator U of shape ((2M+1)d, (2M+1)d), complex dtype.
    """
    d = energies.shape[0]
    M = fourier_cutoff
    omega = omega_d
    T = 2 * math.pi / omega

    # Diagonal H0
    H0 = torch.diag(energies)
    # Master dict of Fourier blocks including H0
    C = {n: fourier_coeffs.get(n, torch.zeros((d, d), dtype=torch.cfloat)) for n in range(-M, M+1)}
    C[0] = C[0] + H0

    # Pre-cook zero block for out-of-range accesses
    zero_block = torch.zeros((d, d), dtype=torch.cfloat)

    # Assemble truncated Floquet Hamiltonian
    N = (2 * M + 1) * d
    Hf = torch.zeros((N, N), dtype=torch.cfloat)
    for a in range(N):
        j = a % d
        m = a // d - M
        for b in range(N):
            k = b % d
            n_ = b // d - M
            # Fourier coupling, default zero for |m-n_|>M
            block = C.get(m - n_, zero_block)
            Hf[a, b] = block[j, k]
            # Add diagonal m*omega term
            if m == n_ and j == k:
                Hf[a, b] += m * omega

    # Compute one-period propagator
    Uf = torch.matrix_exp(-1j * Hf * T)
    

    # Extract central block (assuming M Fourier modes on each side)
    start_idx = M * d
    end_idx = (M + 1) * d
    U_physical = Uf[start_idx:end_idx, start_idx:end_idx]
    return U_physical

