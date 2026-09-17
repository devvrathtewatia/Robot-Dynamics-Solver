"""Backend cost against the number of degrees of freedom.

Confirms the asymptotic claims empirically rather than only asserting them:
RNEA and ABA are O(n), CRBA is O(n^2), and the Lagrangian route is O(n^3) because
it assembles per-link Jacobians and, for the Coriolis matrix, differentiates the
mass matrix n times.

The Lagrangian backend is not the one to use in a loop. It exists because it is the
formulation being taught, and because it validates the fast ones.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'src'))

import robot_dynamics as rd
from robot_dynamics.dynamics import crba, lagrangian, rnea


def timeit(fn, repeats: int = 30) -> float:
    """Median seconds per call, which is robust to a stray slow sample."""
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return float(np.median(samples))


def main() -> None:
    rng = np.random.default_rng(0)
    sizes = [2, 3, 4, 6, 8, 12, 16, 24]
    header = (f'{"n":>4} {"RNEA id":>11} {"Lagr id":>11} {"CRBA M":>11} '
              f'{"Lagr M":>11} {"ABA fd":>11} {"Lagr fd":>11}')
    print(header)
    print('-' * len(header))
    rows = []
    for n in sizes:
        robot = rd.spatial_chain(n)
        q = robot.random_configuration(rng)
        qd, qdd = rng.normal(size=n), rng.normal(size=n)
        tau = rng.normal(size=n)
        # Loop variables are bound as default arguments so each closure captures
        # the current robot rather than the last one - the classic late-binding trap.
        def _rnea_id(r=robot, a=q, b=qd, c=qdd):
            return rnea.inverse_dynamics(r, a, b, c)

        def _lag_id(r=robot, a=q, b=qd, c=qdd):
            return lagrangian.inverse_dynamics(r, a, b, c)

        def _crba_mass(r=robot, a=q):
            return crba.mass_matrix(r, a)

        def _lag_mass(r=robot, a=q):
            return lagrangian.mass_matrix(r, a)

        def _aba_fd(r=robot, a=q, b=qd, c=tau):
            return crba.forward_dynamics(r, a, b, c)

        def _lag_fd(r=robot, a=q, b=qd, c=tau):
            return lagrangian.forward_dynamics(r, a, b, c)

        t = {
            'rnea_id': timeit(_rnea_id),
            'lag_id': timeit(_lag_id, 10),
            'crba_M': timeit(_crba_mass),
            'lag_M': timeit(_lag_mass),
            'aba_fd': timeit(_aba_fd),
            'lag_fd': timeit(_lag_fd, 10),
        }
        rows.append((n, t))
        print(f'{n:>4} ' + ' '.join(f'{t[k] * 1e6:>10.1f}u' for k in
                                    ('rnea_id', 'lag_id', 'crba_M', 'lag_M',
                                     'aba_fd', 'lag_fd')))

    print('\nempirical scaling exponent (fit of log time vs log n, n >= 4):')
    ns = np.array([r[0] for r in rows if r[0] >= 4], dtype=float)
    for key, expected in (('rnea_id', 1), ('crba_M', 2), ('lag_id', 3),
                          ('aba_fd', 1), ('lag_fd', 3)):
        ys = np.array([r[1][key] for r in rows if r[0] >= 4])
        slope = np.polyfit(np.log(ns), np.log(ys), 1)[0]
        print(f'  {key:<10} {slope:4.2f}   (expected about {expected})')


if __name__ == '__main__':
    main()
