"""A mesh file in, a mesh file out.

The other two directions: ``fabriks.contrib`` reads the formats a model usually arrives in and
writes them back out again. This one walks the whole loop and shows what each step cost:

1. **A model as it really comes** -- a two-object scene centred on the origin and two units
   across, which is what a ``.glb`` off any modelling tool looks like. Built as it stands, fabriks
   refuses it outright: the octree is anchored at the origin, and half of this model is below it.
2. **Reading it** -- ``read_glb`` fits the model into the voxel octant, records what it did, and
   writes the collection. The fit is the whole reason step 1 is not an error.
3. **What the fit bought** -- the same model read with no fit and with a shift but no scale, so
   the difference between a usable octree and a nominal one is a number rather than a claim.
4. **Reading the collection back** -- names, ids and the recorded fit, straight off the catalog.
5. **Writing it out again** -- as GLB and as OBJ, at level 0 and at a coarser level, with the
   round-trip error measured against the quantum that bounds it.

Run it::

    uv run python examples/04_formats.py
"""

from __future__ import annotations

import io
import warnings
from pathlib import Path

import numpy as np
import trimesh

import fabriks
from fabriks.contrib import VoxelFit, fit_of, names_of, read_glb, write_glb, write_obj

#: Where this example writes. Nothing else reads it.
OUTPUT = Path(__file__).parent / "out"
COLLECTION = "imported"
#: A cell small enough that a fitted model has something to partition into.
CELL_SIZE = (128, 128, 128)
LEVELS = 3


def demo_scene() -> trimesh.Scene:
    """A two-object scene shaped like a file someone would actually hand you.

    Centred on the origin, about two units across, and with one object placed by a **node
    transform** rather than by moved vertices -- which is the case an importer that reads
    ``scene.geometry`` directly gets silently wrong.

    One object is named ``"7"`` and one ``"head"``, because the two names travel differently: an
    integer-looking name is kept as the object's id, so it survives a round trip exactly, while
    ``"head"`` is assigned an id and carried in the catalog's ``name`` column.
    """
    scene = trimesh.Scene()
    scene.add_geometry(
        trimesh.creation.icosphere(radius=0.6, subdivisions=3),
        geom_name="head",
    )
    scene.add_geometry(
        trimesh.creation.box(extents=[1.0, 0.4, 0.3]),
        geom_name="7",
        transform=trimesh.transformations.translation_matrix([0.8, 0.0, 0.0]),
    )
    return scene


def megabytes(size: int) -> str:
    """Format a byte count the way a fetch budget is usually discussed."""
    return f"{size / 1_000_000:6.2f} MB"


def rule(title: str) -> None:
    """Print a section heading, so a long run stays readable."""
    print(f"\n{title}\n{'-' * len(title)}")


def quantum(collection: fabriks.Collection) -> float:
    """One level-0 position quantum, in the units the source file was written in.

    Positions are 16 bits against each cell's own box, so ``cell_size / 65535`` is the finest
    distance the format can represent -- and dividing by the fit's scale carries that number back
    into the source's units, which is where an error is worth quoting.
    """
    fit = fit_of(collection) or VoxelFit.identity()
    return max(collection.grid.cell_size) / fabriks.QUANT_MAX / fit.scale


def main() -> None:
    """Read a model in, look at what that cost, and write it back out."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    scene = demo_scene()
    blob = scene.export(file_type="glb")

    # ------ 1. what arrived ------ #
    rule("1. The model as it arrives")
    low, high = scene.bounds
    print(f"objects   {sorted(scene.geometry)}")
    print(f"bounds    {np.round(low, 3).tolist()} .. {np.round(high, 3).tolist()}")
    print(f"as GLB    {megabytes(len(blob))}")
    print(
        "\nHalf of this model sits below the origin, and fabriks addresses cells with"
        "\nnon-negative indices -- so `build_collection` would refuse it as it stands. It is"
        "\nalso two units across, and a cell is a whole number of voxels at least one across,"
        "\nso there is no cell smaller than the model to partition it with."
    )

    # ------ 2. reading it ------ #
    rule("2. Reading it")
    store = fabriks.DirectoryStore(OUTPUT / "store")
    imported = read_glb(blob, store, COLLECTION, cell_size=CELL_SIZE, levels=LEVELS)
    print(f"fit       scale x{imported.fit.scale:.1f}, then shift {np.round(imported.fit.translation, 1).tolist()}")
    print(f"names     {imported.names}")
    print(
        f"\n`read_glb` scaled the model so its longest side spans {fabriks.contrib.DEFAULT_RESOLUTION}"
        f" voxels and shifted it\nclear of the origin. `\"7\"` kept its number as its object id;"
        f" `\"head\"` was given the\nlowest free one and its name went into the object catalog."
    )

    # ------ 3. what the fit bought ------ #
    rule("3. What the fit bought")
    fitted = fabriks.open_collection(store, COLLECTION)

    # No fit at all: the model is where the file put it, which is across the origin.
    try:
        read_glb(blob, fabriks.MemoryStore(), "raw", fit=VoxelFit.identity(), levels=1)
    except ValueError as error:
        print(f"no fit at all      -> refused: {error}")

    # Shifted but not scaled: legal, and useless. The model is two voxels across.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        shifted_store = fabriks.MemoryStore()
        read_glb(
            blob,
            shifted_store,
            "shifted",
            fit=VoxelFit(scale=1.0, translation=(0.6, 0.6, 0.6)),
            cell_size=(1, 1, 1),
            levels=1,
        )
    shifted = fabriks.open_collection(shifted_store, "shifted")

    print(f"\n{'':12s}{'cell_size':>18s}{'levels':>8s}{'cells at L0':>14s}{'quantum':>12s}")
    for label, collection in (("shifted", shifted), ("fitted", fitted)):
        print(
            f"{label:12s}{collection.grid.cell_size!s:>18s}{collection.grid.levels:>8d}"
            f"{len(collection.cells_at(0)):>14d}{quantum(collection):>12.2e}"
        )
    print(
        "\nShifting alone is what a naive importer would do, and it is legal -- so nothing"
        "\nraises. But a cell is a whole number of voxels and the model is two voxels across, so"
        "\nthe grid has almost nowhere to cut: a handful of cells, one level, and a quantum twice"
        "\nas coarse even though each cell now covers the whole model. Scaling is the half of the"
        "\nfit that buys the resolution; shifting is only what makes the model representable."
    )

    # ------ 4. what the collection carries ------ #
    rule("4. What the collection carries")
    print(f"{'id':>6s}  {'name':<8s}{'vertices':>10s}{'faces':>8s}  cells")
    for object_id, entry in sorted(fitted.objects.items()):
        print(
            f"{object_id:>6d}  {entry.name or '-':<8s}{entry.vertex_count:>10d}"
            f"{entry.index_count // 3:>8d}  {len(entry.cells)}"
        )
    print(f"\nnames_of  {names_of(fitted)}")
    print(f"fit_of    {fit_of(fitted)}")
    print(
        "\nBoth ride on the object catalog as extra columns, which the format allows on purpose."
        "\nNothing in `REQUIRED_COLUMNS` changed, and `fabriks.verify` reads the collection as an"
        "\nordinary one."
    )

    # ------ 5. writing it back out ------ #
    rule("5. Writing it back out")
    bound = quantum(fitted)
    print(f"{'level':>6s}{'format':>8s}{'bytes':>12s}{'max bounds drift':>20s}")
    for level in range(LEVELS):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = write_glb(fitted, OUTPUT / f"level-{level}.glb", level=level)
        back = trimesh.load(io.BytesIO(out), file_type="glb")
        drift = float(np.max(np.abs(np.asarray(back.bounds) - np.asarray(scene.bounds))))
        print(f"{level:>6d}{'GLB':>8s}{len(out):>12,d}{drift:>20.2e}")

    obj = write_obj(fitted, OUTPUT / "level-0.obj", level=0)
    print(f"{0:>6d}{'OBJ':>8s}{len(obj):>12,d}{'':>20s}")
    print(
        f"\nOne level-0 quantum is {bound:.2e} in the source's own units, and level 0 lands"
        f"\ninside it: positions are 16 bits against each cell's box, so a round trip closes to"
        f"\nabout a quantum and never onto equality. The coarser levels drift further because"
        f"\nthey have also been decimated, which is the point of them rather than a loss."
    )
    print(f"\nWritten to {OUTPUT}")


if __name__ == "__main__":
    main()
