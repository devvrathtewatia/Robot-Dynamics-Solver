"""Shared setup for the examples: import path and plot styling."""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FIGURES = os.path.join(HERE, '..', 'docs', 'figures')
sys.path.insert(0, os.path.join(HERE, '..', 'src'))


def figure_path(name: str) -> str:
    os.makedirs(FIGURES, exist_ok=True)
    return os.path.join(FIGURES, name)


def style():
    """Return pyplot with consistent styling, or None if matplotlib is absent.

    The examples degrade to printing numbers rather than failing, so the library
    stays usable without matplotlib installed.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams.update({'figure.dpi': 110, 'font.size': 9,
                             'axes.grid': True, 'grid.alpha': 0.3,
                             'axes.spines.top': False,
                             'axes.spines.right': False,
                             'lines.linewidth': 1.4})
        return plt
    except ImportError:
        return None
