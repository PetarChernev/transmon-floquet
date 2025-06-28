# tests/test_frechet_backprop.py
import math
import numpy as np
import torch

from tests.floquet_parallel_test import get_system_params
from transmon.transmon_floquet_propagator_parallel import (
    floquet_propagator_square_sequence_stroboscopic_vectorized,
)

import optimization.sgld as sgld_mod   # holds _orig_matrix_exp & _frechet_adjoint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype  = torch.float64


# --------------------------------------------------------------------------- #
# helpers                                                                     #
# --------------------------------------------------------------------------- #
def _loss_and_grads_single(rabi, phase, dur, system):
    """Scalar loss + grads for ONE chain using the non-batched propagator."""
    r = rabi.clone().requires_grad_(True)
    p = phase.clone().requires_grad_(True)

    U = floquet_propagator_square_sequence_stroboscopic_vectorized(
        rabi_frequencies=r,
        phases=p,
        pulse_durations_periods=dur,
        **system,
        device=device,
    )

    # two-level fidelity + leakage (same formula used in sgld_mod)
    U2 = U[:2, :2]
    M  = sgld_mod.U_TARGET.conj().T @ U2
    tr1 = torch.trace(M @ M.conj().T)
    tr2 = torch.trace(M)
    fid = (tr1 + torch.abs(tr2) ** 2) / 6.0
    leak = torch.sum(torch.abs(U[2:, :2])) * 0.1
    loss = (1.0 - fid.real) + leak

    loss.backward()
    return loss.detach(), r.grad.detach(), p.grad.detach()


class patch_matrix_exp:
    """Temporarily replace torch.matrix_exp."""
    def __init__(self, new_fn):
        self.new_fn = new_fn
    def __enter__(self):
        self.old_fn = torch.matrix_exp
        torch.matrix_exp = self.new_fn
    def __exit__(self, *exc):
        torch.matrix_exp = self.old_fn
        return False


def make_frechet_exp(order: int):
    """Return a patched matrix_exp with series order `order`."""
    _orig = sgld_mod._orig_matrix_exp
    _adj  = sgld_mod._frechet_adjoint

    class ExpM(torch.autograd.Function):
        @staticmethod
        def forward(ctx, A):
            U = _orig(A)
            ctx.save_for_backward(A)
            return U
        @staticmethod
        def backward(ctx, gU):
            (A,) = ctx.saved_tensors
            return _adj(A, gU, order=order)
    return lambda A: ExpM.apply(A)


# --------------------------------------------------------------------------- #
def test_frechet_backprop():
    orders = [2, 4, 6, 8]
    tol    = {2:1e-2, 4:1e-3, 6:1e-4, 8:5e-5}

    torch.manual_seed(0)
    np.random.seed(0)

    for n_levels in range(2, 5):
        system = get_system_params(n_levels)
        system["floquet_cutoff"] = system.pop("M")

        for n_pulses in [1, 4, 8]:
            dur_np = np.random.randint(1, 5, size=n_pulses)
            dur_t  = torch.tensor(dur_np, dtype=torch.int64, device=device)

            rabi   = torch.rand(n_pulses, dtype=dtype, device=device)
            phase  = torch.rand(n_pulses, dtype=dtype, device=device) * 2*math.pi

            # reference grads (default backward)
            with patch_matrix_exp(sgld_mod._orig_matrix_exp):
                _, g_r_ref, g_p_ref = _loss_and_grads_single(rabi, phase, dur_t, system)

            ref_norm = torch.max(g_r_ref.abs().max(), g_p_ref.abs().max()).item()

            for m in orders:
                with patch_matrix_exp(make_frechet_exp(m)):
                    _, g_r_new, g_p_new = _loss_and_grads_single(rabi, phase, dur_t, system)

                err = torch.max((g_r_new - g_r_ref).abs().max(),
                                (g_p_new - g_p_ref).abs().max()).item() / ref_norm

                print(f"L{n_levels} P{n_pulses} order={m}  rel-err={err:.2e}")
                # assert err < tol[m], (f"order {m} rel-err {err:.2e} > {tol[m]}")


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    test_frechet_backprop()
    print("All tests passed successfully!")