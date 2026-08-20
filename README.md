<div align="center">
  <a href="https://summerofcode.withgoogle.com/programs/2026/projects/1GFUKY85">
    <img src="assets/gsoc-logo.png" alt="Google Summer of Code" height="80" />
  </a>
</div>

<h1 align="center">Flexible Graph Construction for Neural Weather Prediction</h1>

<div align="center">
  <a href="https://github.com/mllam">
    <img src="assets/mllam-logo.png" alt="MLLAM" height="96" />
  </a>
</div>

<p align="center"><strong>Google Summer of Code 2026 - final report</strong></p>

<p align="center">
  <a href="https://prajwal-tech07.github.io/gsoc-2026-flexible-graph-construction/">Read as a page</a>
  &nbsp;·&nbsp;
  <a href="https://github.com/prajwal-tech07/gsoc-2026-flexible-graph-construction">Source on GitHub</a>
</p>

| | |
| :--- | :--- |
| **Contributor** | Prajwal Hawaldar ([@prajwal-tech07](https://github.com/prajwal-tech07)) |
| **Organisation** | [MLLAM](https://github.com/mllam) |
| **Project** | [Flexible Graph Construction](https://summerofcode.withgoogle.com/programs/2026/projects/1GFUKY85) (350 hours) |
| **Mentors** | Hauke Schulz, Leif Denby, Joel Oskarsson |
| **Repositories** | [weather-model-graphs](https://github.com/mllam/weather-model-graphs), [neural-lam](https://github.com/mllam/neural-lam) |

---

## Summary

Graph-based weather models learn on a mesh. Before this project, the tooling that
builds those meshes assumed the mesh was a regular rectangular lattice, and that
assumption was hardcoded in enough places that no other topology could be used
without rewriting the pipeline.

The goal was to make mesh construction topology-agnostic across the two
repositories that MLLAM maintains: `weather-model-graphs`, which builds and
stores graphs, and `neural-lam`, which trains models on them.

I opened **10 pull requests and 15 issues** across both repos. Three PRs are
merged and shipped, six are open and under review, and the work splits into four
strands: separating mesh geometry from mesh connectivity, connecting the two
repositories so they share one graph format, adding automated performance
regression checking to CI, and migrating `neural-lam` onto PyTorch Geometric's
`HeteroData`.

Most of it began as a written design proposal rather than as code. I opened the
architecture issues, argued for an API, revised it in discussion with the
maintainers, and started implementing once the design was agreed. The section
[below](#designing-before-building) traces that path from issue to merged PR.

### Contributions

| PR | Repo | What it does | Status |
| :--- | :--- | :--- | :--- |
| [#81](https://github.com/mllam/weather-model-graphs/pull/81) | wmg | `mesh_layout` argument, two-step coordinate/connectivity split | **Merged**, in v0.4.0 |
| [#123](https://github.com/mllam/weather-model-graphs/pull/123) | wmg | `to_torch_tensors_on_disk`, writes the shared graph format | **Merged**, in v0.4.0 |
| [#147](https://github.com/mllam/weather-model-graphs/pull/147) | wmg | Benchmark regression check in CI, phase 1 | **Merged** 2026-08-17 |
| [#150](https://github.com/mllam/weather-model-graphs/pull/150) | wmg | Median of repeated benchmark timings, phase 2 | Open, CI green |
| [#596](https://github.com/mllam/neural-lam/pull/596) | neural-lam | `create_graph_with_wmg` CLI, replaces the duplicated builder | Open, approved feedback addressed |
| [#92](https://github.com/mllam/weather-model-graphs/pull/92) | wmg | `mesh_layout="triangular"` | Open, in review |
| [#91](https://github.com/mllam/weather-model-graphs/pull/91) | wmg | `mesh_layout="prebuilt"` for user-supplied meshes | Open |
| [#711](https://github.com/mllam/neural-lam/pull/711) | neural-lam | Load flat graphs into `pyg.HeteroData` | Open, changes requested |
| [#713](https://github.com/mllam/neural-lam/pull/713) | neural-lam | Hierarchical graphs in `HeteroData` | Open, stacked on #711 |

Issues opened along the way include
[#144](https://github.com/mllam/weather-model-graphs/issues/144) (CI benchmarking),
[#149](https://github.com/mllam/weather-model-graphs/issues/149) (dependency floor),
[#661](https://github.com/mllam/neural-lam/issues/661) and
[#714](https://github.com/mllam/neural-lam/issues/714).

---

## The problem

`neural-lam` uses an encode-process-decode architecture. Grid points carrying the
weather state are encoded onto a coarser mesh (`g2m`), message passing happens on
the mesh (`m2m`), and the result is decoded back to the grid (`m2g`). The mesh is
the thing the model actually computes on, so how you build it matters.

Two problems sat underneath this.

First, `weather-model-graphs` built mesh coordinates and mesh connectivity in the
same step. You could not ask for "the same connectivity rules, but on a different
node layout", because the layout was not a separate concept.

Second, `neural-lam` had its own graph builder, `neural_lam/create_graph.py`, 963
lines long, duplicating logic that already existed in `weather-model-graphs`. It
called `networkx.grid_2d_graph` in two places, so a rectangular mesh was not a
default that could be changed, it was an assumption baked into the file. Any
irregular or non-rectangular data source was blocked by it, and every improvement
to graph construction had to be made twice.

---

## Designing before building

The architecture in this project was not handed to me as a specification. I
proposed it, and the proposals came before the code.

| Proposed | Issue | Became |
| :--- | :--- | :--- |
| 2026-02-25 | [wmg#68](https://github.com/mllam/weather-model-graphs/issues/68) grid-node area weights | open |
| 2026-02-26 | [wmg#69](https://github.com/mllam/weather-model-graphs/issues/69) logic bug in CRS handling | fixed, closed |
| 2026-02-26 | [wmg#71](https://github.com/mllam/weather-model-graphs/issues/71) decouple topology from connectivity | revised into #78 |
| 2026-03-01 | [wmg#78](https://github.com/mllam/weather-model-graphs/issues/78) introduce `mesh_layout` | [PR #81](https://github.com/mllam/weather-model-graphs/pull/81), merged |
| 2026-03-01 | [wmg#80](https://github.com/mllam/weather-model-graphs/issues/80) triangular layout | [PR #92](https://github.com/mllam/weather-model-graphs/pull/92) |
| 2026-03-01 | [wmg#79](https://github.com/mllam/weather-model-graphs/issues/79) prebuilt layout | [PR #91](https://github.com/mllam/weather-model-graphs/pull/91) |
| 2026-06-26 | [wmg#144](https://github.com/mllam/weather-model-graphs/issues/144) CI benchmark regression | [PR #147](https://github.com/mllam/weather-model-graphs/pull/147), merged |

The `mesh_layout` design is the clearest example of the pattern, and also of the
fact that the first version was not the right one. I opened
[#71](https://github.com/mllam/weather-model-graphs/issues/71) in February
arguing that mesh topology and mesh connectivity should be separated. It was
closed, because the design needed reworking, and I reopened the argument in a
tighter form as [#78](https://github.com/mllam/weather-model-graphs/issues/78).

What followed was a long design discussion covering the call signatures for all
three graph archetypes, what belonged in `mesh_layout_kwargs` against
`m2m_connectivity_kwargs`, and how coordinate creation should pass its implied
adjacency knowledge downstream so the connectivity step does not have to rederive
it. Several of my own suggestions were corrected in that thread. The API we
converged on there is, with small changes, the one that shipped in v0.4.0.
[PR #81](https://github.com/mllam/weather-model-graphs/pull/81) was opened the day
after that agreement.

The same shape repeats elsewhere. The CI benchmark check began as
[#144](https://github.com/mllam/weather-model-graphs/issues/144), a written
argument for same-runner A/B measurement, and was implemented only after the
approach was agreed. The dependency and specification issues
([#149](https://github.com/mllam/weather-model-graphs/issues/149),
[neural-lam#714](https://github.com/mllam/neural-lam/issues/714)) came out of
questions raised during review of my own PRs.

The practical effect is that by the time I wrote code, the hard decisions had
already been argued out in public, and review was about implementation rather than
about direction.

### Reviewing, and unblocking other people's work

The benchmarking work depended on
[PR #140](https://github.com/mllam/weather-model-graphs/pull/140) by
[@yuvraajnarula](https://github.com/yuvraajnarula), which added memory profiling
and JSON output to the scaling benchmark. Two things came out of reviewing it.

First, a scheduling problem. #140 and
[#117](https://github.com/mllam/weather-model-graphs/pull/117) were both adding
the same `tests/benchmarks/` directory from the same base commit, so whichever
merged second would conflict, and #140 was effectively a superset of #117. I
raised this on #144 and proposed resolving it rather than letting the two
collide. The outcome was that #117 was closed and #140 replaced it on the v0.4.0
roadmap.

Second, a real bug. On review I found that `--output-plot-memory` produced no
file at all: `output_path` was passed into both plot functions but never used,
and the only `savefig` call targeted the runtime plot, so running with
`--track-memory --output-plot-memory` wrote the memory plot on top of the runtime
one. It was fixed before the PR merged.

I also confirmed the JSON schema was what the CI check would consume before
depending on it, which is the cheapest possible time to find out that it is not.

---

## Separating layout from connectivity

[PR #81](https://github.com/mllam/weather-model-graphs/pull/81) introduced a
`mesh_layout` argument and split mesh construction into two independent steps:

1. **Coordinate creation** decides where mesh nodes go.
2. **Connectivity creation** decides which nodes are joined, given those positions.

Each layout is a module exposing the same interface, so adding a new topology
means adding one file rather than touching the pipeline. Connectivity code no
longer knows or cares how the coordinates were produced.

This is the piece everything else stands on. It shipped in v0.4.0 and both the
triangular ([#92](https://github.com/mllam/weather-model-graphs/pull/92)) and
prebuilt ([#91](https://github.com/mllam/weather-model-graphs/pull/91)) layouts
are built on top of it.

---

## One graph format, two repositories

With layout pluggable, the next problem was that the two repositories could not
exchange graphs.

[PR #123](https://github.com/mllam/weather-model-graphs/pull/123) added
`wmg.save.to_torch_tensors_on_disk`, which writes a graph in the on-disk tensor
format `neural-lam` reads, following the graph storage specification the
community agreed on. It shipped in v0.4.0 (released 2026-07-28).

[PR #596](https://github.com/mllam/neural-lam/pull/596) is the other half:
a `create_graph_with_wmg` CLI that builds a graph from a datastore using
`weather-model-graphs` and writes it out in that format, replacing the 963-line
duplicate with **215 lines**. The old entry point still works and emits a
deprecation warning.

This PR took the longest to get right, and most of that was review rather than
code. Working through it produced several things worth recording:

- The estimator for grid node distance is `sqrt(x_range * y_range / N)`. Because
  `np.ptp` measures `n-1` intervals across `n` points, it is biased low by
  `sqrt((nx-1)(ny-1) / (nx * ny))`: about 0.2% at 500x500, but 10% at 10x10.
- Terminology was inconsistent between "spacing", "distance" and "resolution".
  We standardised on "distance", to match `weather-model-graphs`. That surfaced a
  genuine bug: `mesh_node_distance = grid_distance * ratio` means the ratio is
  mesh-to-grid, but the argument was named grid-to-mesh, the wrong way round. It
  was renamed before the CLI ever shipped in a release, so no deprecation path
  was needed.
- On my mentor's suggestion the hand-written assertions in the tests were removed
  in favour of the shared graph validator. Before deleting a check that had come
  from another contributor's PR, I corrupted a valid graph six different ways and
  confirmed the validator caught each one. It also turned out the manual check
  pinned edge features to exactly 3 dimensions, while the specification allows 3
  or 4, so removing it fixed a latent disagreement with the spec.

---

## Catching performance regressions in CI

Graph construction is the slow part of the pipeline, and nothing was watching it
for regressions. I opened
[issue #144](https://github.com/mllam/weather-model-graphs/issues/144) proposing a
CI check, and implemented it in
[PR #147](https://github.com/mllam/weather-model-graphs/pull/147), merged on
2026-08-17.

The design question was how to measure a slowdown on shared CI runners, where
absolute timings are meaningless because hardware varies between jobs. The answer
was to stop comparing against recorded numbers:

- Run the benchmark for the PR and for its base branch **back to back in the same
  job on the same runner**, so both sides see the same hardware and the noise
  largely cancels.
- Swap **only the library** between the two runs, keeping the benchmark harness
  fixed at the PR's version. Checking out the base branch would swap the harness
  too, and you would be measuring with two different rulers.
- Compare **relative percentages**, not seconds, and post the result as a sticky
  PR comment.

The check is informational rather than blocking, and reports peak memory
alongside runtime. The threshold is deliberately set very low (0.1%) so that it
can be raised against real observed noise rather than guessed at up front.

### Phase 2: repeated timings and the median

[PR #150](https://github.com/mllam/weather-model-graphs/pull/150) is the second
half. A single timing sample per grid size turned out to be noisy enough that
unchanged code could look slower, so it adds a `--repetitions` option and reports
the median instead. CI runs 5 repetitions.

Only the timed runs repeat. Peak memory is still measured once, for two reasons:
the run-to-run spread for a given input is under 0.01%, and `tracemalloc` makes
graph creation roughly 5x slower, so repeating it would have cost job time for no
information.

Separating the two exposed something I had not noticed. Because memory tracking
had been active during the timed runs, **every runtime we had reported was
inflated by it**. With `tracemalloc` kept out of the timed path, the largest grid
size drops from about 17.5s to about 3.6s. The benchmark had been partly
measuring its own instrumentation.

I also left the 0.1% threshold alone rather than setting it from local
measurements. Trying to find the noise floor on a laptop gave anywhere from 3% to
47% between two runs of identical code, depending on grid size and how warm the
machine was, which is not a fair proxy for a CI runner. #150 happens to be a clean
experiment for this: it changes nothing under `src/`, so both sides of its own A/B
comparison run identical library code, and whatever the benchmark comment reports
on that PR is close to pure runner noise at 5 repetitions on real hardware. That
is a much better basis for choosing the threshold than my machine, and the plan is
to set it from that number.

---

## Making triangular meshes actually equilateral

A triangular mesh is worth having because each interior node has 6 equidistant
neighbours instead of 8 neighbours at two different distances, which makes
message passing more isotropic. That argument only holds if the triangles really
are equilateral.

They were not. `networkx.triangular_lattice_graph` produces a unit-edge lattice,
but the lattice was then scaled to fill the domain **independently in x and y**.
Stretching one axis more than the other turns equilateral triangles into
isosceles ones, and the distortion depends on the shape of the domain:

| Domain | Longest/shortest edge, per-axis scaling | After the fix |
| :--- | ---: | ---: |
| 100 x 100 | 1.357 | 1.000000 |
| 200 x 60 | 1.902 | 1.000000 |
| 60 x 200 | 1.874 | 1.000000 |

Even on a square domain the longest edge was 36% longer than the shortest. On a
wide domain it was nearly double, so the "equidistant neighbours" property the
layout exists to provide was simply not there.

The fix is to scale by a **single factor in both directions**. Two behaviours
follow, depending on which arguments you give:

- With `mesh_node_spacing`, the edge length *is* the requested spacing, and the
  node counts are chosen to cover the domain. The outermost nodes may sit
  slightly outside it.
- With `nx`/`ny`, the counts are fixed and the lattice is scaled to the largest
  size that still fits inside the domain, so edge length follows from the counts.

The numbers above can be reproduced with `python verify_measurements.py`, which
needs only `networkx` and `numpy`.

### What the fix uncovered

Fixing the scaling exposed a second, quieter problem.

Multiscale graphs are built by generating several lattices at different
resolutions and merging them. The merge works **by coincident position**: a
coarse node and a fine node at the same coordinates become one node, and that is
what ties the levels together.

Because each level was scaled independently to fit the domain, coarse nodes
almost never landed exactly on fine ones. Nothing merged. `flat_multiscale` with
a triangular layout returned **three disconnected components, one per level**,
instead of a single connected multiscale graph. No test caught this, because the
tests checked node and edge counts, and the counts were all correct. The graph
was the right size and the wrong shape.

Anchoring every level to a shared origin, with per-level spacing derived from a
common base, fixed it. Coincident nodes went from 0 to the expected count and the
component count went from 3 to 1. The regression test now asserts connectivity
directly:

```python
def test_flat_multiscale_is_connected(self, xy_large):
    """Regression: independently scaled levels never merged, leaving one
    disconnected component per level instead of a multiscale graph."""
    components = wmg.create.create_all_graph_components(
        coords=xy_large,
        mesh_layout="triangular",
        mesh_layout_kwargs=dict(mesh_node_spacing=2.0),
        m2m_connectivity="flat_multiscale",
        ...
    )
    assert nx.number_weakly_connected_components(components["m2m"]) == 1
```

---

## Migrating to HeteroData

`neural-lam` represents a loaded graph as a dictionary of 11 keys. PyTorch
Geometric's `HeteroData` is a better fit: it is built for graphs with several node
and edge types, which is exactly what a grid/mesh graph is.

[PR #711](https://github.com/mllam/neural-lam/pull/711) loads flat graphs into
`HeteroData` and wires it into `BaseGraphModel`;
[#713](https://github.com/mllam/neural-lam/pull/713) extends it to hierarchical
graphs. Both are open with CI green.

A review from [@Sir-Sloth-The-Lazy](https://github.com/Sir-Sloth-The-Lazy) on #711
found two real bugs I had initially argued were not bugs, and was right on both
counts: the feature flag was unreachable from several
model classes, and storing a `HeteroData` object via `setattr` escapes
`nn.Module._apply`, so the graph stayed on the CPU after `.cuda()`. Both are
fixed, the second by rebuilding the view on access rather than storing it.

My mentor has since asked for a more ambitious version: drop the dual code path
entirely, remove the graph dictionary from `neural-lam` altogether, and hold graph
state in a dedicated module instead of assigning attributes onto the model. That
is the current state of this strand, and it is the main piece of unfinished work.

---

## Where things stand

**Merged and shipped:** the two-step mesh layout architecture (#81), the shared
graph writer (#123), and the CI benchmark regression check (#147). The first two
are in the released v0.4.0.

**Open and close:** #596 has all review feedback addressed, CI green, and is
waiting on final approval. #92 has an equilateral-triangle fix pushed and two
small review points outstanding.

**Open and further out:** #91 (prebuilt meshes), the `HeteroData` migration in
#711 and #713, which needs reworking toward the HeteroData-only design, and
[#150](https://github.com/mllam/weather-model-graphs/pull/150), which is CI green
and waiting on the threshold decision described above.

**Left for whoever picks this up next:** the `--mesh_layout` CLI flag in
[#661](https://github.com/mllam/neural-lam/issues/661), which is written and
waiting for #596 to land, and relaxing the `torch-geometric` floor
([#149](https://github.com/mllam/weather-model-graphs/issues/149)).

Not reached: the stretch work in the proposal on graph quality metrics,
density-adaptive meshes and adaptive mesh refinement. The layout abstraction from
#81 is the hook those would attach to.

---

## What I learned

Two of the bugs I ran into this summer had the same shape, and I did not notice
that until fairly late.

The disconnected-components bug got through a full test suite because every
number it checked was correct. The graph had the expected node count and the
expected edge count. It was still three separate pieces where it should have been
one, and the tests had no opinion about that at all. The benchmark turned out to
have a version of the same problem: memory tracking was running during the timed
section, so the runtimes we had been reporting included the cost of measuring
them. Taking `tracemalloc` out of the timed path dropped the largest grid size
from about 17.5 seconds to about 3.6. In both cases the thing that was supposed to
tell me something was wrong was itself the thing that was wrong, and it reported
success the whole time.

I was also straightforwardly wrong in review on more than one occasion. On #711 a
reviewer raised two problems that I initially argued were not problems. They both
were. The more serious one was that the way I was storing the graph object meant
it never moved to the GPU with the rest of the model, so training would have
silently used a stale copy left on the CPU. I would have shipped that. What stuck
with me is less the specific bug than what it cost to be talked out of it, and
that noticing when a reviewer restates a point is usually cheaper than defending
the first answer.

The part of the work I was weakest at coming in was the design itself. Writing
[#78](https://github.com/mllam/weather-model-graphs/issues/78) meant stating where
the boundary between two concepts actually sat, in a form other people could
disagree with, with no implementation yet to point at. My first attempt at that
argument was closed and had to be rewritten, and the discussion that followed
changed the design several more times. The version that shipped is not the version
I proposed, and it is better for that.

I came into this able to implement a design and came out able to propose one,
argue for it, and give up the parts of it that were wrong. Working on a codebase
that real forecasting work depends on is what taught me the difference between
code that runs and code other people can build on.

---

## Acknowledgements

Thanks to **Leif Denby** for detailed and patient review across both repositories
throughout the summer, and to **Hauke Schulz** and **Joel Oskarsson** for
mentorship and design discussion. Thanks also to the wider MLLAM community for
reviews and for the surrounding work this project depended on.

---

*The measurements quoted in this report can be reproduced from this repository:
`python verify_measurements.py`.*
