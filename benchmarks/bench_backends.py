"""Measure runtime scaling of the dynamics backends."""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np

import robot_dynamics as rd
from robot_dynamics.dynamics import crba, lagrangian, rnea


def timeit(fn, repeats: int = 30) -> float:
    """Return the median runtime per call."""
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

        timings = {
            'rnea_id': timeit(_rnea_id),
            'lag_id': timeit(_lag_id, 10),
            'crba_M': timeit(_crba_mass),
            'lag_M': timeit(_lag_mass),
            'aba_fd': timeit(_aba_fd),
            'lag_fd': timeit(_lag_fd, 10),
        }
        rows.append((n, timings))
        print(f'{n:>4} ' + ' '.join(
            f'{timings[k] * 1e6:>10.1f}u'
            for k in ('rnea_id', 'lag_id', 'crba_M', 'lag_M', 'aba_fd', 'lag_fd')
        ))

    print('\nempirical scaling exponent (fit of log time vs log n, n >= 4):')
    ns = np.array([n for n, _ in rows if n >= 4], dtype=float)
    for key, expected in (
        ('rnea_id', 1), ('crba_M', 2), ('lag_id', 3),
        ('aba_fd', 1), ('lag_fd', 3),
    ):
        ys = np.array([row[key] for n, row in rows if n >= 4])
        slope = np.polyfit(np.log(ns), np.log(ys), 1)[0]
        print(f'  {key:<10} {slope:4.2f}    (expected about {expected})')


if __name__ == '__main__':
    main()
