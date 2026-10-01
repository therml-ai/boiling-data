from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from boiling_data.boiling_data import Field, FloatArray

ComplexArray = npt.NDArray[np.complex128]


@dataclass
class DMD:
    """Exact DMD (Tu et al. 2014) of a field, x(t) ~ sum_k modes[k] b_k e^(omega_k t).

    modes: [rank, *frame_shape], each mode laid out like one frame of the field.
    eigenvalues: discrete-time eigenvalues lambda_k of the one-step operator.
    amplitudes: b_k fitting the first frame.
    singular_values: every singular value of the snapshot matrix, not only the
        rank kept, to judge where to truncate.
    """

    modes: ComplexArray
    eigenvalues: ComplexArray
    amplitudes: ComplexArray
    singular_values: FloatArray
    dt: float
    mean: FloatArray | None = None

    @property
    def continuous_eigenvalues(self) -> ComplexArray:
        """omega_k: real part is the growth rate, imaginary part the angular
        frequency, both per unit simulation time."""
        return (np.log(self.eigenvalues) / self.dt).astype(np.complex128)

    @property
    def frequencies(self) -> FloatArray:
        return np.abs(self.continuous_eigenvalues.imag) / (2 * np.pi)

    def order_by_amplitude(self) -> npt.NDArray[np.intp]:
        """Mode indices from the largest |b_k| times ||mode_k|| down."""
        norms = np.linalg.norm(self.modes.reshape(len(self.modes), -1), axis=1)
        return np.argsort(-np.abs(self.amplitudes) * norms)

    def reconstruct(self, num_frames: int) -> FloatArray:
        powers = self.eigenvalues[:, None] ** np.arange(num_frames)
        dynamics = self.amplitudes[:, None] * powers
        frames: FloatArray = np.einsum("k...,kt->t...", self.modes, dynamics).real
        return frames if self.mean is None else frames + self.mean


def compute_dmd(
    data: FloatArray, dt: float, rank: int, subtract_mean: bool = True
) -> DMD:
    """data: [T, ...] frames sampled every dt. rank caps the SVD truncation and is
    lowered to the number of nonzero singular values when the snapshots span less."""
    frame_shape = data.shape[1:]
    snapshots = data.reshape(len(data), -1).T
    mean = None
    if subtract_mean:
        mean = snapshots.mean(axis=1)
        snapshots = snapshots - mean[:, None]
    before, after = snapshots[:, :-1], snapshots[:, 1:]

    u, s, vh = np.linalg.svd(before, full_matrices=False)
    tolerance = s[0] * max(before.shape) * np.finfo(np.float64).eps if s.size else 0
    rank = min(rank, int((s > tolerance).sum()))
    u, s_kept, v = u[:, :rank], s[:rank], vh[:rank].conj().T

    reduced_operator = u.conj().T @ after @ v / s_kept
    eigenvalues, eigenvectors = np.linalg.eig(reduced_operator)
    modes = after @ v / s_kept @ eigenvectors
    amplitudes = np.linalg.lstsq(modes, snapshots[:, 0], rcond=None)[0]

    return DMD(
        modes=modes.T.reshape(rank, *frame_shape),
        eigenvalues=eigenvalues.astype(np.complex128),
        amplitudes=amplitudes.astype(np.complex128),
        singular_values=s,
        dt=dt,
        mean=None if mean is None else mean.reshape(frame_shape),
    )


def uniform_timestep(time: FloatArray, rtol: float = 1e-2) -> float:
    """The mean step, once every step is within rtol of it: stored times carry
    float rounding, so frames written at a fixed interval are not exactly even."""
    steps = np.diff(time)
    if steps.size == 0:
        raise ValueError("DMD needs at least two frames")
    dt = float(steps.mean())
    if not np.allclose(steps, dt, rtol=rtol, atol=0):
        raise ValueError(
            f"DMD needs uniformly spaced frames; steps range over "
            f"[{steps.min()}, {steps.max()}]"
        )
    return dt


def vorticity(velx: Field, vely: Field) -> Field:
    """Out-of-plane vorticity dv/dx - du/dy of cell-centered 2d velocities."""
    if velx.data.shape != vely.data.shape:
        raise ValueError(
            f"velx {velx.data.shape} and vely {vely.data.shape} must be sampled "
            "on the same grid"
        )
    data = np.gradient(vely.data, vely.grid_x, axis=-1) - np.gradient(
        velx.data, velx.grid_y, axis=-2
    )
    return Field(data, grid_x=velx.grid_x, grid_y=velx.grid_y)
