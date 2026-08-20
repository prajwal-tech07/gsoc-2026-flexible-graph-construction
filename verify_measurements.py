"""
Reproduce the triangular-mesh edge-length measurements quoted in the report.

Standalone: needs only networkx and numpy, no weather-model-graphs install.

    python verify_measurements.py

It compares the two ways of scaling a unit-edge triangular lattice onto a
coordinate domain:

* per-axis  - stretch x and y independently so the lattice fills the domain
              exactly. This was the old behaviour, and it skews the triangles.
* uniform   - one scale factor for both axes, so the triangles stay equilateral
              whatever the aspect ratio of the domain.

The reported figure is max/min edge length. Equilateral means exactly 1.
"""

import networkx as nx
import numpy as np

ROW_HEIGHT = np.sqrt(3) / 2  # row spacing of a unit-edge triangular lattice


def raw_lattice(m, n):
    """Unit-edge triangular lattice and its natural extent."""
    graph = nx.triangular_lattice_graph(m, n)
    positions = {u: np.asarray(graph.nodes[u]["pos"], dtype=float) for u in graph}
    return graph, positions, ((n + 1) / 2, m * ROW_HEIGHT)


def scale_per_axis(positions, span, domain):
    """The old behaviour: stretch x and y independently to fill the domain."""
    sx, sy = domain[0] / span[0], domain[1] / span[1]
    return {u: p * np.array([sx, sy]) for u, p in positions.items()}


def scale_uniform(positions, span, domain):
    """The fix: one scale factor for both axes."""
    s = min(domain[0] / span[0], domain[1] / span[1])
    return {u: p * s for u, p in positions.items()}


def edge_lengths(graph, positions):
    return np.array(
        [np.linalg.norm(positions[u] - positions[v]) for u, v in graph.edges]
    )


def main():
    print(f"{'domain':>18} | {'per-axis max/min':>16} | {'uniform max/min':>15}")
    print("-" * 58)
    for label, domain in [
        ("100 x 100", (100.0, 100.0)),
        ("200 x 60", (200.0, 60.0)),
        ("60 x 200", (60.0, 200.0)),
    ]:
        graph, positions, span = raw_lattice(12, 12)
        old = edge_lengths(graph, scale_per_axis(positions, span, domain))
        new = edge_lengths(graph, scale_uniform(positions, span, domain))
        print(
            f"{label:>18} | {old.max() / old.min():>16.6f}"
            f" | {new.max() / new.min():>15.6f}"
        )


if __name__ == "__main__":
    main()
