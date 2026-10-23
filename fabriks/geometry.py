"""Cutting, snapping, welding and decimating -- everything that makes ``LOCKED`` true.

``boundary: LOCKED`` promises that vertices on cell face planes did not move during
decimation, so two neighbouring cells drawn at different levels meet without a crack. Nothing
downstream can check that promise, so it is kept here, by construction, in three steps:

1. Every object is clipped **once**, at the level-0 cell planes. Clipping puts vertices exactly
   on those planes, and coarser cell planes are a subset of the level-0 planes, so this one cut
   produces every boundary any level will ever need.
2. Coarser levels are decimated from that already-cut mesh with every on-plane vertex held
   **fixed** -- never moved, never collapsed away.
3. A coarse cell is assembled by **welding its eight children**, never by cutting again. The
   seams interior to it dissolve, so their vertices are free for the decimator to spend, and
   only vertices on *this* level's faces stay locked.

There is one residual case, and it is stated rather than hidden. **65535 is odd**, so a
coordinate at an odd multiple of the level-0 cell size -- ``z = 96`` with ``cell_size = 32``,
say -- is a level-0 cell face but lands exactly half a quantum off the level-1 lattice. A
vertex sitting on *both* a shared face plane and such a finer-only plane therefore differs
between the two levels by half the coarser quantum: ``extent / 131070``, about ``5e-4`` voxels
at level 1 for a 32-voxel cell. It is the edge-of-a-face case, a measure-zero set -- in the
reference fixture, 2 vertices in 250.

The alternative is worse, which is why this is the choice: snapping that coordinate onto the
coarse lattice would move the vertex off the level-0 plane it is shared across, and the two
level-0 neighbours -- the common case, and the one a renderer hits at every frame -- would
disagree by the same amount instead. A same-level crack is a worse bug than a cross-level one.
Removing it entirely needs an even quantization denominator with a far-face convention, which
is a format change, not a writer change.

``decimation: QUARTER`` targets ``(1/4)**L`` of the level-0 face count. The collapse is greedy
shortest-edge with a fixed-vertex set -- not quadric-optimal, which the format does not ask
for: it asks for a face-count ratio and an immobile boundary.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import trimesh

from fabriks.codecs import QUANT_MAX
from fabriks.sources import Mesh

# --------------------------------------------------------------------------- #
# Decimation with a fixed boundary
# --------------------------------------------------------------------------- #


def _find(parent: npt.NDArray[np.int64], index: int) -> int:
    """Union-find root, with path compression."""
    root = index
    while parent[root] != root:
        root = parent[root]
    while parent[index] != root:
        parent[index], index = root, parent[index]
    return root


def _roots(parent: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    """Every vertex's union-find root at once, by pointer jumping, compressing ``parent`` in place.

    Vectorised rather than a ``_find`` per vertex: the face count is taken over and over during a
    collapse, and a Python call per vertex per count is what made the collapse quadratic.
    """
    roots = parent
    while True:
        jumped = roots[roots]
        if (jumped == roots).all():
            break
        roots = jumped
    parent[:] = roots
    return roots.copy()


def decimate_fixed(
    vertices: npt.NDArray[np.float64],
    faces: npt.NDArray[np.int64],
    *,
    fixed: npt.NDArray[np.bool_],
    target_faces: int,
    check_interval: int | None = None,
    placement: str = "midpoint",
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64], float]:
    """Collapse shortest edges until ``target_faces`` remain, never moving a fixed vertex.

    Greedy and not quadric-optimal: ``QUARTER`` is a face-count ratio and ``LOCKED`` is boundary
    immobility, and neither asks for optimality. An edge with two fixed endpoints is never
    collapsed; an edge with one is collapsed *onto* the fixed vertex, which is what keeps a cell
    face plane exactly where the clip put it.

    This is the kernel behind ``simplification: GREEDY``, and ``MESHOPT`` (the default) and
    ``QUADRIC`` are both better shaped and better measured -- reach for this one where a heavily
    pinned boundary stops the others reaching a budget. Two things to know about it: the edge order is computed once from the initial lengths and never
    re-measured as vertices move, and no collapse is checked for validity, so an aggressive
    target on a flat-faced surface can consume it entirely. :mod:`fabriks.simplifiers` handles the
    second by relaxing the target until something survives.

    ``check_interval`` is how many collapses to make between face counts; ``None`` counts after
    every collapse below 256 faces, where the overshoot would be fatal, and above that after an
    eighth as many collapses as there are faces left to take. One collapse retires about two
    faces, but a merge between two grown clusters can retire many more, so the batch is sized
    well under the gap; it still shrinks geometrically, and the count is taken ``O(log F)``
    times rather than ``F/16``. ``placement`` is
    ``"midpoint"`` (move both endpoints to their average) or ``"onto_fixed"`` (never move a
    vertex at all, collapsing onto one endpoint -- lower quality, but every surviving vertex
    keeps its exact input position).

    Returns the surviving vertices, the remapped faces, and the largest distance any vertex
    moved -- which is this kernel's estimate of its own error.
    """
    if placement not in ("midpoint", "onto_fixed"):
        raise ValueError(f"`placement` is 'midpoint' or 'onto_fixed', got {placement!r}.")
    vertices = np.asarray(vertices, dtype=np.float64).copy()
    faces = np.asarray(faces, dtype=np.int64)
    fixed = np.asarray(fixed, dtype=bool)
    original = vertices.copy()

    if len(faces) <= target_faces:
        return vertices, faces, 0.0

    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)
    lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    order = np.argsort(lengths)

    parent = np.arange(len(vertices), dtype=np.int64)
    is_fixed = fixed.copy()

    def live_face_count() -> int:
        mapped = _roots(parent)[faces]
        keep = (mapped[:, 0] != mapped[:, 1]) & (mapped[:, 1] != mapped[:, 2]) & (mapped[:, 0] != mapped[:, 2])
        return int(keep.sum())

    # How often to stop and count what is left. Counting is O(V + F), so on a large mesh it is
    # amortised over a batch of collapses -- but a *small* mesh cannot afford the overshoot
    # that buys: one collapse can retire several faces, so a batch of 16 can take a 12-face
    # object from above its target to nothing at all. Below a few hundred faces the count is
    # cheap and the overshoot is fatal, so it is paid every collapse. Above, the batch is sized
    # to the gap still to close, which is what keeps the number of counts logarithmic.
    adaptive = check_interval is None and len(faces) > 256
    if check_interval is None:
        check_interval = max(1, (len(faces) - target_faces) // 8) if adaptive else 1
    check_interval = max(1, int(check_interval))

    collapses_since_check = 0
    for edge_index in order:
        a, b = int(edges[edge_index, 0]), int(edges[edge_index, 1])
        ra, rb = _find(parent, a), _find(parent, b)
        if ra == rb:
            continue
        if is_fixed[ra] and is_fixed[rb]:
            continue
        if is_fixed[ra]:
            parent[rb] = ra
        elif is_fixed[rb]:
            parent[ra] = rb
        elif placement == "midpoint":
            vertices[ra] = 0.5 * (vertices[ra] + vertices[rb])
            parent[rb] = ra
        else:  # "onto_fixed": collapse without moving anything
            parent[rb] = ra

        collapses_since_check += 1
        if collapses_since_check >= check_interval:
            collapses_since_check = 0
            live = live_face_count()
            if live <= target_faces:
                break
            if adaptive:
                check_interval = max(1, (live - target_faces) // 8)

    roots = _roots(parent)
    mapped = roots[faces]
    keep = (mapped[:, 0] != mapped[:, 1]) & (mapped[:, 1] != mapped[:, 2]) & (mapped[:, 0] != mapped[:, 2])
    mapped = mapped[keep]

    used = np.unique(mapped)
    remap = np.full(len(vertices), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    out_vertices = vertices[used]
    out_faces = remap[mapped]

    moved = np.linalg.norm(vertices[roots] - original, axis=1)
    return out_vertices, out_faces, float(moved.max()) if moved.size else 0.0


# --------------------------------------------------------------------------- #
# The boundary
# --------------------------------------------------------------------------- #


def snap_boundary(
    vertices: npt.NDArray[np.float64],
    cell_size: npt.NDArray[np.int64],
    coarse_extent: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    """Put every boundary vertex somewhere every level can represent exactly.

    The step that makes ``LOCKED`` true rather than merely declared, and the one that is easy
    to leave out. **65535 is odd**, so a level's quantization lattice is not the next level's:
    quantize the same point against a level-0 cell box and against the level-1 box that
    contains it and the two decode to positions up to half a coarse quantum apart. Along a
    plane where a fine cell abuts a coarse one, that difference is the crack ``LOCKED``
    forbids.

    Two rules, per axis, and only for vertices that lie on a level-0 cell face plane:

    - an **on-plane** coordinate is pinned to the plane exactly. It quantizes to 0 or 65535 at
      every level for which that plane is a cell face, so it survives untouched;
    - the remaining, **tangential** coordinates are snapped to the *coarsest* level's lattice.
      That lattice is a subset of every finer one -- a coarse point ``m*E/65535`` is the fine
      point ``2m*E'/65535`` in the first child and ``(2m-65535)*E'/65535`` in the second, both
      integral -- so a snapped coordinate is exactly representable at every level.

    Interior vertices are left alone: nothing across a cell face depends on them, and snapping
    them would throw away the precision the finest level exists to carry.

    **What counts as on-plane is a full coarse quantum wide, and that width is load-bearing.**
    A snap moves a coordinate by up to half a coarse quantum, so a vertex nearer to a plane
    than that -- but too far to be called on-plane -- would be snapped *across* it and end up
    outside the cell holding it, where quantization cannot represent it at all. Pinning
    everything within a whole quantum means nothing that could cross is ever snapped.

    Returns the adjusted vertices and a boolean mask of which ones are on a boundary.
    """
    vertices = np.asarray(vertices, dtype=np.float64).copy()
    quantum = np.asarray(coarse_extent, dtype=np.float64) / QUANT_MAX
    scaled = vertices / cell_size
    on_plane = np.abs(vertices - np.rint(scaled) * cell_size) <= quantum
    boundary: npt.NDArray[np.bool_] = np.asarray(on_plane.any(axis=1), dtype=bool)
    if not boundary.any():
        return vertices, boundary

    pinned = np.rint(scaled) * cell_size
    vertices[on_plane] = pinned[on_plane]

    step = np.asarray(coarse_extent, dtype=np.float64) / QUANT_MAX
    tangential = boundary[:, None] & ~on_plane
    snapped = np.rint(vertices / step) * step
    vertices[tangential] = snapped[tangential]
    return vertices, boundary


def on_planes(
    vertices: npt.NDArray[np.float64], extent: npt.NDArray[np.float64], tolerance: float = 1e-6
) -> npt.NDArray[np.bool_]:
    """Which vertices lie on a cell face plane of a grid with this cell extent.

    ``tolerance`` is **cell-relative, not in voxels**: the comparison happens after dividing by
    ``extent``, so 1.0 is a whole cell and one quantization step is ``1 / QUANT_MAX``. Handing
    it a distance in voxels multiplies it by the extent, which on a 256-voxel cell turns a
    quantum into half a voxel and collects vertices that merely pass near a plane.
    """
    scaled = vertices / extent
    return np.asarray((np.abs(scaled - np.rint(scaled)) < tolerance).any(axis=1), dtype=bool)


def border_vertices(faces: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    """The vertices on the mesh's open boundary -- those touching an edge with one triangle.

    **This is the curve that must not move**, and it is a sharper statement of it than
    :func:`on_planes`. A fragment is cut from its object with ``cap=False``, so the cut leaves
    an open boundary: exactly the curve the neighbouring cell's surface continues across. A
    vertex merely *lying on* a cell plane without being on that boundary is interior to this
    sheet, and nothing across the plane depends on where it goes.

    So ``on_planes`` is a conservative superset -- safe, and what the greedy collapse uses --
    while this is the precise condition, and what lets a simplifier that only ever *removes*
    vertices prove it kept ``LOCKED``.
    """
    faces = np.asarray(faces, dtype=np.int64)
    if not len(faces):
        return np.empty(0, dtype=np.int64)
    edges = np.sort(np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    border = unique[counts == 1]
    return np.unique(border) if len(border) else np.empty(0, dtype=np.int64)


# --------------------------------------------------------------------------- #
# Clipping to the level-0 grid
# --------------------------------------------------------------------------- #


def clip_to_cells(mesh: Mesh, cell_size: npt.NDArray[np.int64]) -> dict[tuple[int, int, int], Any]:
    """Cut a mesh at the level-0 cell planes, returning one fragment per occupied cell.

    Vertices produced by a cut lie exactly on the plane, which is what makes the boundary
    ``LOCKED`` at every level: coarser cell planes are a subset of these.

    ``cap=False``: a cell holds a piece of a *surface*, not a solid. The neighbouring cell
    continues the surface across the shared plane, so a cap would be a wall inside the object,
    doubling the triangles on every cell face and showing through anything transparent.

    **Recursive bisection, not one cut-out per cell.** Cutting each cell of the bounding box out
    of the whole mesh costs a full-mesh slice per cell -- empty ones included, which on a long
    diagonal object are most of them -- so the clip was ``cells x faces``. Halving the piece at
    the middle cell plane of its widest index range instead puts each face through about
    ``log2(cells)`` slices, and a half with no geometry is dropped before it costs anything. Both
    halves of a split come from the same parent and the same plane, so the seam between them is
    cut once rather than twice.
    """
    subject = mesh.as_trimesh()
    low, high = subject.bounds[0], subject.bounds[1]
    size = np.asarray(cell_size, dtype=np.float64)

    lower = np.floor(np.asarray(low) / cell_size).astype(np.int64)
    upper = np.floor((np.asarray(high) - 1e-9) / cell_size).astype(np.int64)

    fragments: dict[tuple[int, int, int], Any] = {}
    pending: list[tuple[Any, npt.NDArray[np.int64], npt.NDArray[np.int64]]] = [(subject, lower, upper)]
    while pending:
        piece, first, last = pending.pop()
        span = last - first
        if not span.any():
            fragments[(int(first[0]), int(first[1]), int(first[2]))] = piece
            continue

        axis = int(np.argmax(span))
        middle = int(first[axis] + (span[axis] + 1) // 2)
        plane_origin = np.zeros(3)
        plane_origin[axis] = float(middle * cell_size[axis])
        for sign in (-1.0, 1.0):
            normal = np.zeros(3)
            normal[axis] = sign
            half = trimesh.intersections.slice_mesh_plane(
                piece, plane_normal=normal, plane_origin=plane_origin, cap=False
            )
            if half is None or len(half.faces) == 0:
                continue
            half_first, half_last = first.copy(), last.copy()
            if sign < 0:
                half_last[axis] = middle - 1
            else:
                half_first[axis] = middle
            # Narrow to the cells whose *closed* box the half still touches. Closed, so a face
            # lying exactly in a cell plane reaches every cell the per-cell cut gave it to.
            half_low, half_high = half.bounds[0], half.bounds[1]
            half_first = np.maximum(half_first, np.ceil(half_low / size).astype(np.int64) - 1)
            half_last = np.minimum(half_last, np.floor(half_high / size).astype(np.int64))
            if (half_first <= half_last).all():
                pending.append((half, half_first, half_last))

    # In (i, j, k) order, as the per-cell loop produced them: the fragments of a coarse cell are
    # welded in this order, so it decides vertex order and with it what the simplifier sees.
    return dict(sorted(fragments.items()))


def concatenate_and_weld(
    pieces: Sequence[tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]], tolerance: int = 9
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    """Merge fragments into one mesh, fusing vertices that coincide exactly.

    The children's shared seams are bit-identical -- both sides were cut by the same plane
    and snapped by :func:`snap_boundary` -- so welding on rounded coordinates fuses them
    without a tolerance argument doing any real work.
    """
    if len(pieces) == 1:
        return pieces[0]

    offsets, all_vertices, all_faces = 0, [], []
    for vertices, faces in pieces:
        all_vertices.append(vertices)
        all_faces.append(faces + offsets)
        offsets += len(vertices)
    vertices = np.vstack(all_vertices)
    faces = np.vstack(all_faces)

    _, first, inverse = np.unique(
        np.round(vertices, tolerance), axis=0, return_index=True, return_inverse=True
    )
    order = np.argsort(first)
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    faces = rank[inverse.reshape(-1)][faces]
    keep = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])
    return vertices[first[order]], faces[keep]


def drop_degenerate(
    vertices: npt.NDArray[np.float64], faces: npt.NDArray[np.int64]
) -> npt.NDArray[np.int64]:
    """Drop faces that snapping fused into a line or a point.

    Snapping can bring two of a triangle's corners onto the same coordinate; a face that lost a
    corner is no longer a face, and a zero-area one contributes nothing but a degenerate
    normal.
    """
    keep = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])
    areas = np.linalg.norm(
        np.cross(
            vertices[faces[:, 1]] - vertices[faces[:, 0]],
            vertices[faces[:, 2]] - vertices[faces[:, 0]],
        ),
        axis=1,
    )
    return faces[keep & (areas > 0.0)]


__all__ = [
    "border_vertices",
    "clip_to_cells",
    "concatenate_and_weld",
    "decimate_fixed",
    "drop_degenerate",
    "on_planes",
    "snap_boundary",
]
