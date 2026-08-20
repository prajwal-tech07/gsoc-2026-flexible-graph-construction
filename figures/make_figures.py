"""
Regenerate the figures used in the report.

Standalone: needs only networkx, numpy and matplotlib, no weather-model-graphs
install. It reproduces the two scaling strategies for a triangular lattice so
the numbers quoted in the report can be checked independently.

    python figures/make_figures.py
"""

import matplotlib.pyplot as plt
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
    """The fix: one scale factor for both axes, so triangles stay equilateral."""
    s = min(domain[0] / span[0], domain[1] / span[1])
    return {u: p * s for u, p in positions.items()}


def edge_lengths(graph, positions):
    return np.array(
        [np.linalg.norm(positions[u] - positions[v]) for u, v in graph.edges]
    )


def report_ratios():
    """Print the edge-length statistics quoted in the report."""
    for label, domain in [
        ("square  100 x 100", (100.0, 100.0)),
        ("wide    200 x  60", (200.0, 60.0)),
        ("tall     60 x 200", (60.0, 200.0)),
    ]:
        graph, positions, span = raw_lattice(12, 12)
        old = edge_lengths(graph, scale_per_axis(positions, span, domain))
        new = edge_lengths(graph, scale_uniform(positions, span, domain))
        print(
            f"{label}:  per-axis max/min = {old.max() / old.min():.6f}"
            f"   uniform max/min = {new.max() / new.min():.6f}"
        )


def draw(ax, graph, positions, title):
    for u, v in graph.edges:
        x0, y0 = positions[u]
        x1, y1 = positions[v]
        ax.plot([x0, x1], [y0, y1], linewidth=0.8, color="#4C72B0")
    pts = np.array(list(positions.values()))
    ax.scatter(pts[:, 0], pts[:, 1], s=9, color="#C44E52", zorder=3)
    ax.set_title(title, fontsize=10)
    # Without this the triangles are distorted by the axes, not by the maths,
    # which makes the two panels impossible to compare honestly.
    ax.set_aspect(1)
    ax.set_xticks([])
    ax.set_yticks([])


def main():
    report_ratios()

    # Same lattice as the table above, so the figure and the quoted ratios agree.
    domain = (200.0, 60.0)
    graph, positions, span = raw_lattice(12, 12)
    old = scale_per_axis(positions, span, domain)
    new = scale_uniform(positions, span, domain)

    l_old, l_new = edge_lengths(graph, old), edge_lengths(graph, new)

    fig, axes = plt.subplots(2, 1, figsize=(9, 5))
    draw(
        axes[0],
        graph,
        old,
        f"Per-axis scaling: edge lengths {l_old.min():.2f} to {l_old.max():.2f} "
        f"(max/min = {l_old.max() / l_old.min():.3f})",
    )
    draw(
        axes[1],
        graph,
        new,
        f"Uniform scaling: every edge {l_new.min():.2f} "
        f"(max/min = {l_new.max() / l_new.min():.3f})",
    )
    fig.suptitle(
        "Triangular mesh on a 200 x 60 domain", fontsize=12, fontweight="bold"
    )
    fig.tight_layout()
    fig.savefig("figures/triangular-equilateral.png", dpi=160, bbox_inches="tight")
    print("wrote figures/triangular-equilateral.png")


if __name__ == "__main__":
    main()
