# Why fabriks does not store normals

The format carries positions and indices. It does not carry vertex normals, and a renderer
is expected to compute them itself — stated normatively in
[`fabriks/codecs/__init__.py`](fabriks/codecs/__init__.py) (the `normals` section, line 35):

> **normals** — Omitted, and therefore absent from both the `encoding` object and the shard
> columns. An omitted normal encoding means the renderer computes vertex normals itself.

That says *what*. This document says *why*, and — more usefully — what it costs, so that
the next person to ask does not have to re-derive the answer from `clip_to_cells` and the
simplifier protocol. Every structural claim below cites a line; every number below was
measured, and the method is given so it can be re-measured.

## The shape of the decision

Normals are absent from `_REQUIRED_ENCODING_KEYS` (`manifest.py:155`), so a manifest cannot
declare a normal encoding, and absent from the `geometry` schema (`frames.py:43`), so a
shard has nowhere to put one. On the way in they are discarded at `contrib/scene.py:243`,
where a loaded `trimesh.Trimesh` — which does carry `vertex_normals` — is narrowed to the
two arrays `Mesh` has room for (`sources.py:54`).

The whole tradeoff turns on one observation:

> **Stored normals would buy exactly two things a client cannot compute for itself:
> authored creases on objects that span more than one cell, and consistency across cell and
> LOD boundaries.**

Everything else about a normal is recoverable from positions plus winding. Storing it would
be storing something the client can regenerate exactly — redundancy, not information. So the
question is never "normals or no normals"; it is whether those two things are worth their
price. The sections below price them, and the first is narrower than it looks: a crease lives
in how vertices are *split*, not in the normal values, and positions plus indices carry
splitting perfectly well.

## What the omission buys

### Bytes, and more of them than the arithmetic suggests

For a closed manifold the naive estimate is mild: positions cost `6V`, indices `3F×4 = 24V`,
so octahedral 16-bit normals at 4 bytes should add about 13%. The measured figure is worse,
and it gets much worse as soon as compression is on.

Measured on a `subdivisions=5` icosphere across 32 level-0 cells of 64 voxels, comparing the
stored `positions` + `indices` blobs against an octahedral-16 normal stream computed from
the same geometry:

| `codec`/`compression` | geometry blobs | + oct16 normals |
| --- | --- | --- |
| `NONE`/`NONE` | 399,792 B | **+18.5%** |
| `NONE`/`ZSTD` | 193,713 B | **+37.5%** |
| `MESHOPT`/`NONE` | 211,568 B | **+35.0%** |

The reason the relative cost nearly doubles is that **an octahedral normal stream is very
nearly incompressible** — 74,144 B falls to 72,694 B under the format's own zstd, 98% of
raw, while positions and indices together fall to 48%. Normals are high-entropy where
spatially-sorted positions and a coherent triangle list are not. So the attribute that
looks cheapest in a raw buffer is the one that most resists every mechanism the format uses
to get small, and it costs most precisely in the configurations a bandwidth-conscious
deployment would choose.

One qualifier, in fairness: the `MESHOPT` row pairs meshopt-encoded geometry with a *naive*
oct16 normal stream. meshopt ships a dedicated octahedral filter for exactly this problem,
and a real implementation would use it and land lower. Treat 35% as an upper bound, and 18.5%
as the floor.

Against a format whose premise is per-cell byte budgets — `plan_grid(cell_bytes=…,
layer_bytes=…)` — that range is not accounting. It is fewer cells inside a fetch budget, or
a coarser level at the same bandwidth, on every frame.

### One buffer, and a spec that stays small

Under the default `codec: NONE` a blob is the renderer's buffer verbatim: the column is
uploaded, with no decoder in front of the geometry at all. One attribute is one buffer.

A second attribute is not just a second buffer. `manifest.py:369-375` derives a ZSTD blob's
uncompressed length from the row's counts — six bytes a vertex, four bytes an index — so
normals would need a third declared size relation. They would also need an
`encoding.normals` vocabulary, and a decision about whether the key is required or optional
that every third-party decoder in every language must then handle in both states.

The format already contains a warning about what that costs. `INDICES_UINT16`
(`manifest.py:115`) sits in the encoding vocabulary at `manifest.py:145`, is implemented
nowhere — both codecs hard-wire four-byte indices — and so a manifest declaring it validates
cleanly and then decodes to garbage. A vocabulary entry is a liability until something reads
it. Normals would be a much larger one.

### One source of truth for orientation

Winding. A stored normal is a second one, and the two can disagree — a mesh with inconsistent
winding and correct normals decodes differently depending on which the renderer believes.

## What it costs

### Authored creases, but only once an object spans cells

This one is easy to overstate, and the measurement is more interesting than the intuition.

A hard edge is conventionally encoded as *split* vertices: coincident positions carrying
different normals. Dropping the normal array at `contrib/scene.py:243` does not by itself lose
that, because the crease is really in the topology — two patches that do not share vertices —
and positions plus indices carry topology exactly. A renderer averaging face normals over the
split vertices recovers the hard edge on its own.

Decimation does not lose it either. The default backend runs `fast_simplification` with
`preserve_border=True`, and a split patch's border *is* the crease, so it is pinned by
construction.

What loses a crease is the octree. `clip_to_cells` cuts the surface at the cell planes,
introducing new vertices, and `concatenate_and_weld` then fuses coincident positions
unconditionally (`geometry.py:330`, `np.unique` over rounded coordinates) — when children are
merged into a coarse cell, and when `object_mesh` reassembles an object across cells. That
weld is position-only, so it cannot tell a crease from a coincidence.

Measured on a box whose six faces are welded patches split from one another — the way real
content encodes hard edges — as distinct vertex normals after a round trip, six being perfect:

| | level 0 | level 1 | level 2 |
| --- | --- | --- | --- |
| object inside one cell | **6** | **6** | **6** |
| same object across 56 cells | 82 | 37 | 26 |

So the format preserves authored creases perfectly, including through decimation, for any
object small enough to sit in a single cell — and destroys them for anything larger. Since an
object that fits in one cell is an object the octree was not needed for, the practical reading
is that creases do not survive on the geometry this format exists to serve.

Worth noting where the blame actually lies: the loss is caused by a **position-only weld**,
not by the absence of a normals column. A weld that declined to fuse across a sharp dihedral
angle would preserve creases without the format storing a single extra byte.

### Cross-cell shading seams

This is the cost specific to this format's shape rather than a general fact about normals,
and it is the one worth understanding before dismissing the topic.

`clip_to_cells` slices each object at the level-0 cell planes with `cap=False`
(`geometry.py:298`) — a cell holds a piece of a *surface*, not a solid. So a vertex lying on a
cell plane exists in both neighbouring cells, but each cell contains only the triangles on its
own side. A client computing vertex normals per cell therefore averages a **partial**
neighbourhood, and the two cells produce different normals for the same physical vertex.

`boundary: LOCKED` guarantees that fine and coarse cells meet without a *geometric* crack. It
says nothing about shading continuity, and shading continuity is what breaks here.

Measured as the angle between the two cells' normals for each vertex position appearing in
more than one cell:

| mesh | level | shared-vertex pairs | mean | p99 | max |
| --- | --- | --- | --- | --- | --- |
| icosphere | 0 | 7,104 | 2.2° | 9.2° | 12.2° |
| icosphere | 1 | 348 | 4.0° | 8.4° | 8.7° |
| torus | 0 | 3,180 | 4.1° | 11.9° | 15.8° |
| torus | 1 | 240 | 13.2° | 25.0° | 26.1° |
| subdivided box | 0 | 2,800 | 3.2° | 90.0° | **90.0°** |
| subdivided box | 1 | 144 | 0.6° | 17.6° | 20.5° |

Read that table as the whole argument in miniature. On smooth surfaces the seam is a couple of
degrees — present, visible under a sharp specular, tolerable.

The box row needs its condition stated precisely, because the headline 90° is narrow rather
than pervasive: the mean is 3.2°, and all 24 of the offending positions lie **both** on a cell
plane **and** on an edge of the box, where two perpendicular faces meet. That is the exact
coincidence — a cell plane intersecting an authored edge — and there one cell holds only the
triangles of one face while its neighbour holds only the other's, so the two "vertex normals"
are simply the two face normals, 90° apart. It is a small set of vertices, and it is a hard
lighting discontinuity on every one of them. Smooth surfaces cannot produce it; authored
geometry with sharp edges produces it wherever an edge crosses the grid.

The torus row makes the other point: it does not reliably improve at coarser levels. Mean
disagreement *rises* from 4.1° to 13.2° at level 1, because a decimated vertex has fewer faces
in its neighbourhood, so losing the ones across the cell plane costs proportionally more. That
is the level a viewer spends most of its time at when zoomed out.

Scope it precisely: this bites the streaming draw path — `read_cells` yielding `DecodedCell`
(`reader.py:217`), which is the loop the README shows. It does not bite `object_mesh`
(`reader.py:650`), which welds an object's pieces across cells before returning — but that is
a whole-object fetch at a single level, which is the opposite of what the octree is for.

### Shading pop across levels, and per-cell compute

Each level is an independent decimation, never a delta on a finer one, so recomputed normals
differ per level and shading shifts when the planner swaps a cell's level. Normals derived
once from level 0 would have kept it stable.

And the client pays O(F) cross products with scattered accumulation per cell per level, then
fills a normal buffer that cannot simply be uploaded from the column — which gives back part
of the zero-copy win the omission was partly justified by.

## Where the decision lands

**It is right for the data the format is shaped around.** Voxel coordinate space, a default
`resolution` of 512, `(z, y, x)` volumes passed through without transposition
(`README.md:165`), integer object ids standing in for segment labels — this is marching-cubes
isosurfaces over segmented volumes. That data has no authored creases to lose, has vertex
counts where 18–35% of payload is real money, and is correctly smooth-shaded. The measured
seam on smooth surfaces is 2–4°. The omission is not a shortcut there; it is the correct
call, twice over.

**It is weaker for `contrib`.** The GLB/OBJ/PLY readers invite authored content, and the
README calls GLB "the best round trip here". Authored content is exactly the content that has
creases and smoothing groups — the content whose creases survive only while an object fits in
one cell, and the only content that can produce the 90° seam at all. The round trip is
geometric, not appearance-preserving — which is consistent with materials being skipped
outright (`contrib/obj.py:61`), but is worth saying rather than leaving to be discovered.

## If this is ever revisited

The two sides are not equally easy, and a future attempt should start knowing which is which.

**The wire side is additively extensible.** Extra columns are allowed on purpose
(`frames.py:7`), `validate_columns` checks presence and never exclusivity, and
`Encoding.from_dict` drops keys it does not know — so a `normals` column and an
`encoding.normals` key are a backward-compatible extension, old readers ignoring both. The
one catch is that the builder's own frames are asserted to *equal* the declared schema
(`tests/test_frames.py:48`), so the writer needs a schema change rather than merely an extra
column.

**The ingest side is not.** An attribute would have to be threaded through `Mesh`
(`sources.py:54`), `scene_to_objects` (`contrib/scene.py:243`), `clip_to_cells` and
`concatenate_and_weld`, `Simplifier.simplify` and `Simplified`
(`simplifiers/protocol.py:60`, `:18`), `BlobCodec` (`codecs/protocol.py`), and `DecodedCell`
(`reader.py:217`). And the default backend declines `fast_simplification`'s collapse-report
array (`simplifiers/quadric.py:83` — "its collapses, which it is not here"), so there is no
vertex correspondence to remap an attribute through. Normals would end up recomputed per
level from the decimated geometry — which is what the client already does.

But the important conclusion is that **neither of the two losses actually requires that
project.** Both have a cheaper, targeted fix:

- **Creases** need no stored attribute at all. They are lost to a position-only weld
  (`geometry.py:330`); a weld that declined to fuse across a sharp dihedral angle would keep
  them, and would touch one function.
- **The seam** needs a normals column but no attribute plumbing: the writer sees the whole
  surface before `clip_to_cells` cuts it, so it could compute normals there, per level, and
  store them — fixing boundary consistency and LOD pop without threading anything through the
  simplifier, and without needing the collapse map that does not exist.

So a full attribute channel is the expensive way to buy both, and it is not the only way to
buy either. Anyone reopening this should price the two narrow fixes first — and the tables
above are the measurements to justify them against.

## The siblings

UVs, materials and vertex colours are omitted on the same reasoning, and `contrib/obj.py:61`
already says so for materials: fabriks writes surfaces. Normals are the interesting case only
because they are the one attribute a client can *mostly* reconstruct — which is what makes
the omission defensible, and what makes the two things it does lose easy to overlook.

## Reproducing the numbers

No fixtures ship for these; they were measured ad hoc and are cheap to redo.

- **Bytes.** Write one mesh with `fabriks.write_meshes(..., codec=…, compression=…)` into a
  `MemoryStore`, read `level0/part-00000.parquet` back with `fabriks.frames.parquet_to_table`,
  and sum `len()` over the `positions` and `indices` columns. Compute per-cell vertex normals,
  encode them octahedrally to two `int16`, and compare — against the raw stream for
  `compression: NONE`, and against `fabriks.codecs.compression.compress(blob, "ZSTD")` for
  `compression: ZSTD`, or the comparison is not like for like.
- **Seam.** For each cell from `collection.read_cells(...)` at one level, compute vertex
  normals from that cell's own vertices and faces alone, key them by rounded position, and
  take the angle between every pair of normals sharing a position across different cells.
- **Creases.** Build a box whose six faces are welded patches split from one another — take
  `trimesh.creation.box().subdivide()`, `submesh` the faces by face normal, `merge_vertices`
  each patch, and concatenate. Count distinct `vertex_normals` before, then round-trip it
  twice: once with a `cell_size` large enough to hold it, once small enough to split it. The
  contrast between the two is the whole finding — a fully split mesh (every triangle its own
  island) is the wrong test, since `preserve_border=True` then pins every vertex and no level
  decimates at all.
