import math
import torch


def qubit_rotation(axis, degree) -> torch.Tensor:
    """
    Return the 2×2 unitary that rotates a qubit by `degree` degrees
    about the given `axis`.

    Parameters
    ----------
    axis : str or (Sequence[float], torch.Tensor)
        - 'x', 'y', or 'z' (case-insensitive); or
        - a length-3 real vector/1-D tensor (need not be unit length).
    degree : float
        Rotation angle in degrees (positive = right-hand rule).

    Returns
    -------
    torch.Tensor
        Complex-valued 2×2 tensor with dtype torch.complex64.
    """
    # --- map common strings or accept generic vectors --------------
    paulis = {
        "x": torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64),
        "y": torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64),
        "z": torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64),
    }

    if isinstance(axis, str):
        pauli_vec = paulis[axis.lower()]
    else:
        # ensure torch tensor of shape (3,) and unit-normalize
        n = torch.as_tensor(axis, dtype=torch.float32).flatten()
        if n.numel() != 3:
            raise ValueError("Custom axis must have exactly three components.")
        n = n / torch.linalg.norm(n)
        pauli_vec = (
            n[0] * paulis["x"]
            + n[1] * paulis["y"]
            + n[2] * paulis["z"]
        )

    # --- convert angle to radians ----------------------------------
    theta = math.radians(degree)
    c = math.cos(theta / 2.0)
    s = math.sin(theta / 2.0)

    # --- build the rotation matrix ---------------------------------
    return (c * torch.eye(2, dtype=torch.complex64)) - (1j * s * pauli_vec)
