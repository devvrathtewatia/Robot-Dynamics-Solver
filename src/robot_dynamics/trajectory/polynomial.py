"""Polynomial joint-space trajectories.

Coefficients are obtained by **solving the boundary-condition system** rather than
by substituting memorised closed-form expressions. The two are mathematically
identical, but the linear-solve form is self-evidently correct, extends to any
polynomial order by changing one number, and cannot harbour a transcription error in
a coefficient — which is a real risk, since a wrong :math:`a_4` still produces a
plausible-looking smooth curve that quietly violates its end conditions.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


class PolynomialTrajectory:
    r"""A polynomial trajectory in joint space, per joint, of arbitrary odd order.

    Parameters
    ----------
    q0, qf:
        Start and end configurations, shape ``(n,)``.
    duration:
        Total time :math:`T > 0`.
    qd0, qdf, qdd0, qddf:
        Optional boundary velocity and acceleration, default zero.
    order:
        ``5`` (quintic, default) matches position, velocity **and** acceleration at
        both ends, giving continuous torque. ``3`` (cubic) matches only position and
        velocity, leaving an acceleration step at :math:`t=0` — a discontinuous
        torque demand no real actuator can follow, which is exactly why quintic is
        the default here.
    """

    def __init__(self, q0: ArrayLike, qf: ArrayLike, duration: float,
                 qd0: ArrayLike | None = None, qdf: ArrayLike | None = None,
                 qdd0: ArrayLike | None = None, qddf: ArrayLike | None = None,
                 order: int = 5) -> None:
        if duration <= 0.0:
            raise ValueError(f'duration must be positive, got {duration}')
        if order not in (3, 5):
            raise ValueError(f'order must be 3 or 5, got {order}')
        self.q0 = np.asarray(q0, dtype=float).reshape(-1)
        self.qf = np.asarray(qf, dtype=float).reshape(-1)
        if self.q0.shape != self.qf.shape:
            raise ValueError('q0 and qf must have the same shape')
        self.n = self.q0.size
        self.T = float(duration)
        self.order = order

        z = np.zeros(self.n)
        self.qd0 = z if qd0 is None else np.asarray(qd0, float).reshape(self.n)
        self.qdf = z if qdf is None else np.asarray(qdf, float).reshape(self.n)
        self.qdd0 = z if qdd0 is None else np.asarray(qdd0, float).reshape(self.n)
        self.qddf = z if qddf is None else np.asarray(qddf, float).reshape(self.n)
        self.coeffs = self._solve_coefficients()

    def _solve_coefficients(self) -> NDArray:
        """Solve for polynomial coefficients from the boundary conditions."""
        T, m = self.T, self.order + 1
        rows, rhs = [], []

        def powers(t: float, derivative: int) -> NDArray:
            """Row of the design matrix: d^k/dt^k of [1, t, t^2, ...]."""
            row = np.zeros(m)
            for j in range(derivative, m):
                coef = 1.0
                for d in range(derivative):
                    coef *= (j - d)
                row[j] = coef * t ** (j - derivative)
            return row

        conditions = [(0.0, 0, self.q0), (T, 0, self.qf),
                      (0.0, 1, self.qd0), (T, 1, self.qdf)]
        if self.order == 5:
            conditions += [(0.0, 2, self.qdd0), (T, 2, self.qddf)]
        for t, deriv, value in conditions:
            rows.append(powers(t, deriv))
            rhs.append(value)

        A = np.array(rows)                       # (m, m)
        B = np.array(rhs)                        # (m, n)
        return np.linalg.solve(A, B)             # (m, n)

    def evaluate(self, t: float) -> tuple[NDArray, NDArray, NDArray]:
        r"""Return :math:`(q_d(t), \dot q_d(t), \ddot q_d(t))`.

        Times outside ``[0, T]`` are clamped, so a controller that runs past the end
        of the trajectory holds the final pose rather than extrapolating a polynomial
        off to infinity.
        """
        tc = min(max(float(t), 0.0), self.T)
        m = self.order + 1
        tp = np.array([tc ** j for j in range(m)])
        tv = np.array([j * tc ** (j - 1) if j >= 1 else 0.0 for j in range(m)])
        ta = np.array([j * (j - 1) * tc ** (j - 2) if j >= 2 else 0.0
                       for j in range(m)])
        return tp @ self.coeffs, tv @ self.coeffs, ta @ self.coeffs

    def sample(self, dt: float = 0.01) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        """Sample the whole trajectory: ``(t, q, qd, qdd)`` with time along axis 0."""
        ts = np.arange(0.0, self.T + dt / 2, dt)
        out = [self.evaluate(t) for t in ts]
        return (ts,
                np.array([o[0] for o in out]),
                np.array([o[1] for o in out]),
                np.array([o[2] for o in out]))


class MultiPointTrajectory:
    """A sequence of quintic segments through waypoints, at rest at each one.

    Coming to rest at every waypoint keeps acceleration continuous across the joins
    without needing to solve a global spline. That is a deliberate simplification and
    is stated as a limitation: a true minimum-time path would not stop.
    """

    def __init__(self, waypoints: ArrayLike, durations: ArrayLike) -> None:
        wp = np.asarray(waypoints, dtype=float)
        if wp.ndim != 2 or wp.shape[0] < 2:
            raise ValueError('waypoints must be (num_points, n) with at least 2 '
                             'points')
        dur = np.asarray(durations, dtype=float).reshape(-1)
        if dur.size != wp.shape[0] - 1:
            raise ValueError(f'need {wp.shape[0] - 1} durations for '
                             f'{wp.shape[0]} waypoints, got {dur.size}')
        if np.any(dur <= 0):
            raise ValueError('all durations must be positive')
        self.segments = [PolynomialTrajectory(wp[i], wp[i + 1], dur[i])
                         for i in range(len(dur))]
        self.starts = np.concatenate([[0.0], np.cumsum(dur)])
        self.T = float(self.starts[-1])
        self.n = wp.shape[1]

    def evaluate(self, t: float) -> tuple[NDArray, NDArray, NDArray]:
        """Evaluate the active segment at time ``t``."""
        tc = min(max(float(t), 0.0), self.T)
        idx = int(np.searchsorted(self.starts, tc, side='right') - 1)
        idx = min(max(idx, 0), len(self.segments) - 1)
        return self.segments[idx].evaluate(tc - self.starts[idx])

    def sample(self, dt: float = 0.01) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        """Sample the whole trajectory: ``(t, q, qd, qdd)``."""
        ts = np.arange(0.0, self.T + dt / 2, dt)
        out = [self.evaluate(t) for t in ts]
        return (ts,
                np.array([o[0] for o in out]),
                np.array([o[1] for o in out]),
                np.array([o[2] for o in out]))


def quintic(q0: ArrayLike, qf: ArrayLike, duration: float) -> PolynomialTrajectory:
    """Convenience: a rest-to-rest quintic trajectory."""
    return PolynomialTrajectory(q0, qf, duration, order=5)


def cubic(q0: ArrayLike, qf: ArrayLike, duration: float) -> PolynomialTrajectory:
    """Convenience: a rest-to-rest cubic trajectory."""
    return PolynomialTrajectory(q0, qf, duration, order=3)
