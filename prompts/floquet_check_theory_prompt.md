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

---

### Further reading

Equation (12) in Ref. \[Phys. Rev. B 99, 094303 (2019)] shows exactly how the “fold-back’’ from Sambe to physical space is carried out by **summing the extended-space blocks**, i.e. the prescription above.&#x20;

For a comprehensive pedagogical treatment see the review by P. Hänggi *et al.*, Chap. 5 “Driven Quantum Systems’’ (in particular the discussion around Eqs. 5.116–5.118 on reconstructing physical states from Sambe space). ([physik.uni-augsburg.de][1])

[1]: https://www.physik.uni-augsburg.de/theo1/hanggi/Chapter_5.pdf "qtad_090797.dvi"


## Task
Your task is to check the methematical theory above. Does this match the Floquet thoery from literature? Is the Floquet Hamiltonian defined correctly using the Fourier coefficients? Are the Fourier coefficients derived correctly for this Hamiltonian? Are there any other issues you observe?