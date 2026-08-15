"""Choosing the grid from what a fetch costs, rather than the other way round.

:func:`fabriks.choose_cell_size` answers "what cell holds an object whole", which is the
question that keeps a collection *correct* -- an object wider than a cell is cut, and its cut
vertices are pinned. This module answers the other one: **what cell, and how many levels, make a
fetch cost about what I want it to.** They are different questions with different failure modes,
so this sits beside that function rather than replacing it, and yields to it where they disagree.

Two budgets, because two different things are being sized
---------------------------------------------------------
``cell_bytes`` sizes a **cell**, which is the smallest thing a reader fetches and the unit a
renderer draws. ``layer_bytes`` sizes a **level**, and decides where the octree stops: once a
whole level fits in one retrieval there is nothing left for a coarser one to save, so the tree
ends there and the renderer's root is a single request. That is why ``levels`` is a result here
and not an argument -- nothing about a fixed depth follows from the data, and a depth that
outruns the data costs a file per level and saves nothing.

Both are **blob bytes** -- ``len(positions) + len(indices)``, the quantity
:func:`fabriks.blob_sizes` computes and the one the format already exposes per cell as
``CellEntry.blob_bytes`` so that a plan can be budgeted before a single fetch. It is deliberately
not the Parquet file size: parts are written with zstd unconditionally, so a geometry file runs
roughly 0.4-0.7x its payload, and *larger* than it once a level gets small enough for the footer
to dominate. Budgeting on the payload is the number that travels with the format.

What is exact here and what is not
----------------------------------
Level 0 is arithmetic. Where geometry lands is `floor(p / cell_size)`, which is a binning
problem with no simplifier in it, so cell occupancy, face counts and per-cell bytes come out
within a few percent of a real build -- provided two corrections are kept:

- **Cutting inflates the face count.** A triangle crossing a cell plane is retriangulated into
  roughly three pieces, so each straddle costs *two extra* faces. Counting the cells a triangle
  touches and stopping there under-predicts by a fifth to a quarter.
- **Vertices per cell must be counted, not inferred.** ``V = F/2`` is an identity for a closed
  surface, and a cell holds an open patch with a boundary, so the real cost is nearer 22 bytes a
  face than the 15 that identity implies.

Coarse levels are **not** arithmetic, and this module does not pretend otherwise. A coarse level
is whatever the simplifier managed, and ``QUARTER`` is a target it frequently misses -- when a
cell is small relative to the objects, every cut vertex is pinned and the collapse stops short.
So per-cell bytes are reported as a band: they *grow* while the simplifier is falling short of
its target, then *shrink* at the top of the tree where the coarsest cells stop being saturated.
A single factor cannot span both regimes, which is why :data:`DEFAULT_COARSENING` is a pair and
why :class:`LayerEstimate` carries a low and a high rather than a number.

The honest summary: trust level 0, treat the rest as a bracket, and read
:meth:`GridPlan.describe` before spending an expensive write.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypedDict

import numpy as np
import numpy.typing as npt

from fabriks.build import choose_cell_size
from fabriks.errors import FormatError
from fabriks.frames import DEFAULT_ROW_GROUP_BYTES
from fabriks.manifest import CODEC_NONE
from fabriks.octree import MORTON_BITS
from fabriks.sources import MeshSource, coerce_objects

if TYPE_CHECKING:  # pragma: no cover - typing only
    from fabriks.sources import Mesh

#: How many bytes of geometry one cell should carry. A cell is what a renderer draws and the
#: smallest thing it fetches, so this is the number that decides whether a frame is forty small
#: requests or four large ones. Measured cells on typical scenes run 1-37 KB, which is the range
#: to reach for -- **not** :data:`fabriks.DEFAULT_ROW_GROUP_BYTES`, which is a *level*-scale
#: number: asking for a 512 KiB cell on most collections yields an octree of one cell.
DEFAULT_CELL_BYTES = 8 * 1024

#: How per-cell bytes change per level going coarse, as ``(low, high)``. Measured 0.65x-2.53x
#: across scenes, and the spread is not noise: bytes per cell *grow* while the simplifier misses
#: its ``QUARTER`` target, then *fall* once the coarsest cells stop being saturated. One factor
#: cannot describe both, so this is a band and the estimates it produces are a bracket.
DEFAULT_COARSENING = (0.4, 3.0)

#: What fraction of a level the next coarser one holds, as ``(low, high)``. This is a *layer*
#: figure and deliberately not derived from :data:`DEFAULT_COARSENING` and the cell counts: cells
#: fall roughly fourfold per level while per-cell bytes climb, and composing the pessimistic end
#: of both gives a layer that never shrinks, which is not a thing that happens. Measured directly
#: instead: layers came in at 0.08x-0.51x across scenes. The low end is well under ``QUARTER``
#: on purpose -- the target is a ceiling on what survives, not a floor, and at the top of a tree
#: where cells hold one small object apiece the collapse runs all the way down to
#: ``floor_faces`` and a level all but vanishes. The high end is what the stopping rule uses, so
#: it errs toward one level too many: an extra level is a small file, where one too few leaves a
#: renderer with no root to fetch.
DEFAULT_LAYER_SHRINK = (0.05, 0.55)

#: The smallest cell worth addressing, per component. Below a few voxels a cell partitions noise
#: rather than structure -- the same floor :func:`fabriks.choose_cell_size` applies.
_MIN_CELL = 8

#: Bytes a vertex and bytes a face under ``codec: NONE`` -- positions are three ``uint16``, and
#: an index is four bytes, so a face is three of them. See :mod:`fabriks.codecs.raw`.
_BYTES_PER_VERTEX = 6
_BYTES_PER_FACE = 12

#: Extra faces each straddled cell costs beyond the one already counted: a clipped triangle
#: comes back as roughly three. This is what turns a touched-cell count into a face count.
_RETRIANGULATION = 2

#: How far the level-0 model is trusted, as a fraction. Level 0 is arithmetic rather than a
#: guess, but "arithmetic" is not "to the byte": the clip is trimesh's rather than a formula, and
#: welding merges vertices across a fragment this cannot see. Measured within 12% of real builds
#: across three scenes and six cell sizes, on both the whole layer and the percentile the budget
#: is set on; 15% is the headroom that leaves.
_MODEL_TOLERANCE = 0.15

#: A constant the model is scaled by, because it under-counts by a consistent tenth and a
#: consistent bias is worth removing rather than hiding inside the tolerance. The bias is
#: one-directional for a reason: every unmodelled effect here *adds* geometry -- a clipped
#: triangle fans into slightly more than three pieces, and a fragment's boundary carries vertices
#: that the closed-surface arithmetic pairs off.
_CALIBRATION = 1.10

#: How many cells along one axis a single face is expanded into at most. Sixteen because that is
#: where the estimate stops improving: below it, scenes built from a few very large triangles
#: have their occupancy badly under-counted, and above it nothing moves. The cap exists so one
#: outsized face cannot turn the expansion into millions of rows.
_MAX_SPAN = 16

#: When to warn that a plan is in the regime where the writer's decimation is known to push a
#: vertex outside its cell: this many levels or more, on a cell this many times under the object
#: fit. Both thresholds are below the shallowest crash observed, so the warning fires first.
_RISKY_LEVELS = 5
_RISKY_FIT_RATIO = 4.0


class GridKwargs(TypedDict):
    """The two arguments a plan hands to :func:`fabriks.build_collection`."""

    cell_size: tuple[int, int, int]
    levels: int


@dataclass(frozen=True)
class LayerEstimate:
    """What one level is predicted to cost, and how much of that is a guess.

    ``cells`` comes from binning coordinates rather than from the simplifier, so it lands within
    a few percent at every level -- it runs slightly low, because a triangle is counted in the
    cell its centroid falls in while the cut can leave a sliver of it in a neighbour.

    Byte and face figures are arithmetic only at level 0, where ``exact`` is ``True`` and the
    ``_low`` / ``_high`` pair is the model's tolerance. Above it they are an extrapolation and
    the pair is a *band*: read it as "somewhere in here", not as "this ± a bit".
    """

    #: The level this describes, 0 being full detail.
    level: int
    #: Cells holding geometry at this level, from binning. Within a few percent, biased low.
    cells: int
    #: Faces across the whole level, after the cut inflation correction.
    faces_low: int
    faces_high: int
    #: Blob bytes across the whole level.
    bytes_low: int
    bytes_high: int
    #: Blob bytes for one cell, at the plan's percentile -- the number ``cell_bytes`` is set on.
    cell_bytes_low: int
    cell_bytes_high: int
    #: Whether the byte figures are arithmetic rather than an extrapolation. Level 0 only.
    exact: bool


@dataclass(frozen=True)
class GridPlan:
    """A grid chosen from a byte budget, with the per-level cost it is predicted to produce.

    Splat it straight into a build::

        plan = fabriks.plan_grid(objects, cell_bytes=8 * 1024)
        collection = fabriks.build_collection(objects, **plan.as_kwargs())

    It is worth printing :meth:`describe` first. The plan reports where it had to override the
    budget -- ``object_fit`` is the cell size below which objects start being cut, and a byte
    budget that asks for something smaller is refused rather than honoured, because a cut object
    is a coarse level that costs a file and saves nothing.
    """

    #: The level-0 cell, in voxels, in the same component order as the vertices.
    cell_size: tuple[int, int, int]
    #: How deep the octree goes -- derived from ``layer_bytes``, not chosen.
    levels: int
    #: One estimate per level, finest first.
    layers: tuple[LayerEstimate, ...]
    #: What :func:`fabriks.choose_cell_size` would have picked: the floor this yields to.
    object_fit: tuple[int, int, int]
    #: Faces after cutting over faces before it, at ``cell_size``. 1.0 means nothing was cut.
    inflation: float
    #: Why ``cell_size`` is not purely what the byte budget asked for, if it is not.
    overrides: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Refuse a plan that could not be built from."""
        if len(self.cell_size) != 3 or any(int(component) < 1 for component in self.cell_size):
            raise FormatError(
                f"`cell_size` is three whole numbers of at least 1 voxel, got {self.cell_size!r}."
            )
        if self.levels < 1:
            raise FormatError(f"An octree has at least one level, got {self.levels}.")

    def as_kwargs(self) -> GridKwargs:
        """The plan as keyword arguments for :func:`fabriks.build_collection`."""
        return {"cell_size": self.cell_size, "levels": self.levels}

    def describe(self) -> str:
        """The per-level table, as something to read before spending an expensive write.

        Estimated rows are marked, because the difference between "this is what level 0 will
        cost" and "this is roughly where level 3 lands" is the whole point of the bracket.
        """
        lines = [
            (
                f"cell_size {self.cell_size}  levels {self.levels}  "
                f"(object fit {self.object_fit}, cut inflation {self.inflation:.2f}x)"
            ),
            f"{'L':>2}  {'cells':>7}  {'faces':>17}  {'layer bytes':>21}  {'bytes/cell':>19}",
        ]
        for layer in self.layers:
            if layer.exact:
                # The tolerance is symmetric, so its midpoint is the model's own answer -- which
                # is the number to show, rather than either edge of the headroom around it.
                faces = f"{(layer.faces_low + layer.faces_high) // 2:,}"
                total = f"{(layer.bytes_low + layer.bytes_high) // 2:,}"
                per_cell = f"{(layer.cell_bytes_low + layer.cell_bytes_high) // 2:,}"
            else:
                faces = f"{layer.faces_low:,} .. {layer.faces_high:,}"
                total = f"{layer.bytes_low:,} .. {layer.bytes_high:,}"
                per_cell = f"{layer.cell_bytes_low:,} .. {layer.cell_bytes_high:,}"
            mark = "" if layer.exact else "  est."
            lines.append(
                f"{layer.level:>2}  {layer.cells:>7,}  {faces:>17}  {total:>21}  {per_cell:>19}{mark}"
            )
        lines.extend(f"note: {reason}" for reason in self.overrides)
        return "\n".join(lines)


def _unit_aspect(aspect: Sequence[int] | None) -> npt.NDArray[np.float64]:
    """The candidate cell's shape, normalized so its smallest component is 1.

    Unset it is cubic. That is deliberately *not*
    :func:`fabriks.choose_cell_size`'s ratios: those come from the shape of the objects, which is
    a different thing from the shape of the grid, and on real scenes they disagree with the
    source array's chunking they are standing in for. fabriks assigns no meaning to the three
    components, so isotropic-in-voxels is the neutral choice, and the answer that is actually
    worth having -- the chunk shape of the array the meshes were extracted from -- is knowledge
    only the caller has.
    """
    if aspect is None:
        return np.ones(3, dtype=np.float64)
    values = np.asarray(aspect, dtype=np.float64)
    if values.shape != (3,) or (values <= 0).any():
        raise FormatError(
            f"`aspect` is three positive numbers, one per component in the same order as the "
            f"vertices, got {aspect!r}."
        )
    return values / values.min()


def _linear_key(cells: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    """One integer per ``(i, j, k)`` triple, for grouping. Not a Morton code -- order is unused."""
    return (cells[:, 0] << (2 * MORTON_BITS)) | (cells[:, 1] << MORTON_BITS) | cells[:, 2]


@dataclass(frozen=True)
class _Binned:
    """Every face's cell at the finest candidate, plus what is needed to re-bin it coarser."""

    #: How many faces the collection holds, before any cutting.
    face_count: int
    #: Per face, the inclusive index range its three vertices span. ``(n_faces, 3)`` each.
    span_low: npt.NDArray[np.int64]
    span_high: npt.NDArray[np.int64]
    #: Each face's three vertex ids, made unique across objects. ``(n_faces, 3)``.
    vertex_ids: npt.NDArray[np.int64]
    #: The furthest coordinate any vertex reaches, per component -- the grid is anchored at 0.
    extent: npt.NDArray[np.float64]


def _bin_faces(meshes: Mapping[int, Mesh], base: npt.NDArray[np.float64]) -> _Binned:
    """Assign every face in the collection to a cell of the finest candidate grid, once.

    This is the only pass over the geometry. Every coarser candidate is a right-shift of what it
    produces, because cells nest exactly -- the same ``triple // 2**level`` regrouping the
    builder does when it merges eight cells into one.
    """
    span_low: list[npt.NDArray[np.int64]] = []
    span_high: list[npt.NDArray[np.int64]] = []
    vertex_ids: list[npt.NDArray[np.int64]] = []
    offset = 0
    faces_seen = 0
    extent = np.zeros(3, dtype=np.float64)

    for mesh in meshes.values():
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        if faces.size == 0:
            continue
        # The grid is anchored at the origin and cell indices are non-negative, so geometry in
        # the negative octant has no cell to land in. `morton_encode` says the same thing at
        # build time; saying it here means a plan is never quietly computed against a grid the
        # collection could not be written to.
        if vertices.min(initial=0.0) < 0.0:
            raise FormatError(
                "Vertices are non-negative -- the octree is anchored at the origin and a cell "
                "index cannot be. Shift the geometry into the positive octant first."
            )
        extent = np.maximum(extent, vertices.max(axis=0))
        corners = vertices[faces]  # (n, 3, 3): face, corner, component
        span_low.append(np.floor(corners.min(axis=1) / base).astype(np.int64))
        span_high.append(np.floor(corners.max(axis=1) / base).astype(np.int64))
        vertex_ids.append(faces + offset)
        offset += len(vertices)
        faces_seen += len(faces)

    if not span_low:
        raise FormatError("Every object in the collection is empty; there is nothing to size a grid against.")

    return _Binned(
        face_count=faces_seen,
        span_low=np.concatenate(span_low),
        span_high=np.concatenate(span_high),
        vertex_ids=np.concatenate(vertex_ids),
        extent=extent,
    )


@dataclass(frozen=True)
class _Rung:
    """One candidate cell size, and what a level built on it would hold."""

    cells: int
    faces: int
    inflation: float
    total_bytes: int
    cell_bytes: int  # at the requested percentile


def _measure(binned: _Binned, shift: int, percentile: float) -> _Rung:
    """What a level looks like at ``base * 2**shift``, from the single binning pass.

    Faces are corrected for cutting before anything is measured in bytes: a triangle crossing a
    cell plane comes back as roughly three, so a straddle costs two faces beyond the one the
    touched-cell count already accounts for. Vertices are counted rather than inferred from the
    face count -- a cell holds an open patch, whose boundary the closed-surface identity misses.
    """
    low = binned.span_low >> shift
    span = np.minimum((binned.span_high >> shift) - low + 1, _MAX_SPAN)
    incidences = np.prod(span, axis=1)
    touched = int(incidences.sum())

    # Expand each face into every cell it touches, rather than only the one holding its centroid.
    # The centroid is where a face would go if it were a point; a face that straddles a plane is
    # cut, and both sides are real geometry in both cells. Counting centroids alone under-reports
    # occupancy and per-cell load by a fifth on heavily cut collections.
    face_index = np.repeat(np.arange(len(incidences)), incidences)
    starts = np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum(incidences)[:-1]])
    within = np.arange(touched, dtype=np.int64) - np.repeat(starts, incidences)
    spans = span[face_index]
    steps = np.stack(
        [
            within % spans[:, 0],
            (within // spans[:, 0]) % spans[:, 1],
            within // (spans[:, 0] * spans[:, 1]),
        ],
        axis=1,
    )
    keys = _linear_key(low[face_index] + steps)

    unique_keys, inverse = np.unique(keys, return_inverse=True)
    faces_per_cell = np.bincount(inverse, minlength=len(unique_keys)).astype(np.float64)

    # Unique (cell, vertex) pairs, so a vertex shared by faces in the same cell is paid for once
    # and one shared across a cell plane is paid for in both -- which is what welding produces.
    pairs = np.unique(
        np.stack([np.repeat(keys, 3), binned.vertex_ids[face_index].reshape(-1)], axis=1), axis=0
    )
    _, vertex_counts = np.unique(pairs[:, 0], return_counts=True)
    vertices_per_cell = vertex_counts.astype(np.float64)

    # A clipped triangle comes back as roughly three, so a straddle costs two faces beyond the
    # incidence already counted. Spread over the incidences, that is how much each one grows.
    raw_faces = binned.face_count
    faces = raw_faces + _RETRIANGULATION * (touched - raw_faces)
    inflation = faces / raw_faces if raw_faces else 1.0
    per_incidence = faces / touched if touched else 1.0

    per_cell = (
        _BYTES_PER_VERTEX * vertices_per_cell + _BYTES_PER_FACE * faces_per_cell
    ) * (per_incidence * _CALIBRATION)

    return _Rung(
        cells=len(unique_keys),
        faces=int(faces),
        inflation=inflation,
        total_bytes=int(per_cell.sum()),
        cell_bytes=int(np.percentile(per_cell, percentile)),
    )


def plan_grid(
    objects: Mapping[int, MeshSource],
    *,
    cell_bytes: int = DEFAULT_CELL_BYTES,
    layer_bytes: int = DEFAULT_ROW_GROUP_BYTES,
    aspect: Sequence[int] | None = None,
    codec: str = CODEC_NONE,
    percentile: float = 95.0,
    coarsening: tuple[float, float] = DEFAULT_COARSENING,
    layer_shrink: tuple[float, float] = DEFAULT_LAYER_SHRINK,
    max_levels: int = 12,
) -> GridPlan:
    """Choose ``cell_size`` and ``levels`` from what a fetch should cost.

    Args:
        objects: The same ``{object_id: mesh}`` mapping :func:`fabriks.build_collection` takes.
            Nothing is built or written -- this reads the geometry and returns arithmetic.
        cell_bytes: How many blob bytes one cell should hold. The largest cell size whose
            ``percentile``-th cell fits this is chosen, because a bigger cell means fewer
            fetches as long as each fetch stays in budget. Reach for 4-64 KB; this is a *cell*
            budget, and :data:`fabriks.DEFAULT_ROW_GROUP_BYTES` is a level-scale number.
        layer_bytes: How many blob bytes a whole level may hold before the octree stops. Once a
            level fits in one retrieval a coarser one saves nothing, so this is what decides
            ``levels``. Defaults to the row-group budget, which is the format's own fetch unit.
        aspect: The shape to scale, e.g. the source array's chunk shape -- the value worth
            matching and the one nothing about the meshes can reveal. Cubic when unset.
        codec: The codec the collection will use. Byte figures are exact for ``NONE``; anything
            else compresses, so they become an upper bound and this says so in ``overrides``.
        percentile: Which cell ``cell_bytes`` describes. The default of 95 budgets the tail,
            which is what actually blows a fetch: cells routinely spread 7x from median to
            largest, so sizing on the average under-serves the cells that matter.
        coarsening: ``(low, high)`` bounds on how per-cell bytes change per level going coarse.
            See :data:`DEFAULT_COARSENING` for why it is a band and not a factor.
        layer_shrink: ``(low, high)`` bounds on what fraction of a level the next coarser one
            holds. Its high end is what decides ``levels``. See :data:`DEFAULT_LAYER_SHRINK`.
        max_levels: A backstop on the descent, in case the budget is never met.

    Returns:
        A :class:`GridPlan`. ``plan.as_kwargs()`` goes straight into
        :func:`fabriks.build_collection`; ``plan.describe()`` is worth reading first.

    Raises:
        FormatError: The mapping is empty, or ``aspect`` is not three positive numbers.

    Warns:
        UserWarning: The byte budget asked for something the geometry will not support -- a cell
            below the size objects fit in, or below what the Morton cap allows. The plan is
            still returned, with the reason in ``overrides``; nothing here is fatal.
    """
    if percentile < 0.0 or percentile > 100.0:
        raise FormatError(f"`percentile` selects a cell between 0 and 100, got {percentile}.")
    for name, band in (("coarsening", coarsening), ("layer_shrink", layer_shrink)):
        if band[0] <= 0.0 or band[1] < band[0]:
            raise FormatError(f"`{name}` is an ascending pair of positive factors, got {band!r}.")
    if max_levels < 1:
        raise FormatError(f"An octree has at least one level, got `max_levels` of {max_levels}.")

    meshes = coerce_objects(objects)
    if not meshes:
        raise FormatError("A collection holds at least one object; there is nothing to size a grid against.")

    object_fit = choose_cell_size(meshes)
    unit = _unit_aspect(aspect)

    # The ladder starts at the finest cell worth addressing and climbs in powers of two, so that
    # every rung is exactly two of the one below -- which is what lets one binning pass serve all
    # of them, and what the LOCKED boundary argument needs of the level-0 planes.
    base = unit * 2.0 ** max(0.0, np.ceil(np.log2(_MIN_CELL / unit.min())))
    binned = _bin_faces(meshes, base)

    def size_at(shift: int) -> tuple[int, int, int]:
        """The cell size of a rung, as whole voxels."""
        return tuple(int(component) for component in np.rint(base * (2**shift)))  # type: ignore[return-value]

    # Rungs are measured on demand and stop once one cell holds everything: there is no coarser
    # grid than that, and measuring 30 of them on a large mesh is real time for no answer.
    rungs: list[_Rung] = []

    def rung_at(shift: int) -> _Rung:
        """The rung at ``shift``, measuring the ones below it once each."""
        while len(rungs) <= shift:
            rungs.append(_measure(binned, len(rungs), percentile))
            if rungs[-1].cells <= 1:
                break
        return rungs[min(shift, len(rungs) - 1)]

    ceiling = max_levels + MORTON_BITS

    # The largest rung still inside the budget. Falling off the top means the whole collection
    # fits in one cell, which is a legitimate -- if degenerate -- answer.
    chosen = 0
    for shift in range(ceiling):
        rung = rung_at(shift)
        if rung.cell_bytes > cell_bytes:
            break
        chosen = shift
        if rung.cells <= 1:
            break

    overrides: list[str] = []

    # The Morton cap is the one hard constraint: level-0 indices are the largest in the tree, so
    # a cell too small makes a cell key unrepresentable. Resolved here rather than discovered by
    # `morton_encode` partway through an expensive build.
    raised_from = size_at(chosen)
    while chosen + 1 < ceiling and (
        np.ceil(binned.extent / np.asarray(size_at(chosen), dtype=np.float64)) >= (1 << MORTON_BITS)
    ).any():
        chosen += 1
    if size_at(chosen) != raised_from:
        overrides.append(
            f"cell_bytes of {cell_bytes:,} asked for {raised_from}, where a cell index would pass "
            f"the {MORTON_BITS}-bit Morton limit; raised to {size_at(chosen)}"
        )

    cell_size = size_at(chosen)

    # The object fit is reported, not enforced. Cutting objects is normal here -- the format
    # exists to cut them, and the repo's own fixtures use a cell smaller than their largest
    # object. What is harmful is a cell so small that every vertex ends up on a cut plane and
    # pinned, at which point the simplifier cannot reach QUARTER and a coarse level costs a file
    # while saving nothing. That is a warning rather than a clamp: clamping to
    # `choose_cell_size`'s answer, which deliberately overshoots at twice the 90th-percentile
    # object, would throw away the byte budget entirely and hand back a one-cell octree.
    if (np.asarray(cell_size, dtype=np.float64) < np.asarray(object_fit, dtype=np.float64)).any():
        overrides.append(
            f"cell {cell_size} is below the object fit {object_fit}, so objects will be cut and "
            f"their cut vertices pinned; expect the simplifier to fall short of QUARTER and the "
            f"coarse levels to be larger than estimated. Raise cell_bytes to trade fetch size for it"
        )
    level0 = rung_at(chosen)

    # `levels`: cells per level are exact -- they are the rungs above the one chosen. Per-cell
    # bytes are bracketed, and the stopping test uses the high end so the rule stays conservative.
    low, high = float(coarsening[0]), float(coarsening[1])
    shrink_low, shrink_high = float(layer_shrink[0]), float(layer_shrink[1])

    levels = max_levels
    for step in range(max_levels):
        if level0.total_bytes * (shrink_high**step) <= layer_bytes or rung_at(chosen + step).cells <= 1:
            levels = step + 1
            break

    layers: list[LayerEstimate] = []
    for step in range(levels):
        rung = rung_at(chosen + step)
        exact = step == 0
        cell_low = (1.0 - _MODEL_TOLERANCE) if exact else low**step
        cell_high = (1.0 + _MODEL_TOLERANCE) if exact else high**step
        layer_low = (1.0 - _MODEL_TOLERANCE) if exact else shrink_low**step
        layer_high = (1.0 + _MODEL_TOLERANCE) if exact else shrink_high**step
        layers.append(
            LayerEstimate(
                level=step,
                cells=rung.cells,
                faces_low=int(level0.faces * layer_low),
                faces_high=int(level0.faces * layer_high),
                bytes_low=int(level0.total_bytes * layer_low),
                bytes_high=int(level0.total_bytes * layer_high),
                cell_bytes_low=int(level0.cell_bytes * cell_low),
                cell_bytes_high=int(level0.cell_bytes * cell_high),
                exact=exact,
            )
        )

    if levels == 1:
        overrides.append(
            f"layer_bytes of {layer_bytes:,} already holds the whole of level 0, so there is "
            f"nothing a coarser level would save"
        )

    # A deep tree on cells well under the object fit is the regime where `build_collection` is
    # currently known to fail: decimation places a collapsed vertex at the quadric-optimal spot,
    # which for a vertex near -- but not on -- a cell face can land outside the cell, and
    # quantization is per cell, so `encode_positions` raises `PartitioningError`. Measured on one
    # scene at over a voxel outside. It is a writer bug rather than a planning one, but a plan is
    # the thing that would walk into it, so it is named here rather than discovered mid-build.
    if levels >= _RISKY_LEVELS and (
        np.asarray(cell_size, dtype=np.float64) * _RISKY_FIT_RATIO
        < np.asarray(object_fit, dtype=np.float64)
    ).any():
        overrides.append(
            f"{levels} levels on a cell of {cell_size}, well under the object fit {object_fit}, is "
            f"the regime where decimation is known to push a vertex outside its cell and the build "
            f"raises PartitioningError. Raise cell_bytes, or cap the depth with max_levels"
        )

    # These are the ones a caller should not have to go looking for: the plan is not what the
    # budget asked for. The codec note below is a caveat on the numbers, not an override, so it
    # is recorded without a warning.
    for reason in overrides:
        warnings.warn(reason, stacklevel=2)

    if codec != CODEC_NONE:
        overrides.append(
            f"codec {codec!r} compresses, so every byte figure here is an upper bound; they are "
            f"exact only for {CODEC_NONE!r}"
        )

    return GridPlan(
        cell_size=cell_size,
        levels=levels,
        layers=tuple(layers),
        object_fit=object_fit,
        inflation=level0.inflation,
        overrides=tuple(overrides),
    )


__all__ = [
    "DEFAULT_CELL_BYTES",
    "DEFAULT_COARSENING",
    "DEFAULT_LAYER_SHRINK",
    "GridKwargs",
    "GridPlan",
    "LayerEstimate",
    "plan_grid",
]
