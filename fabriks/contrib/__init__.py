"""Reading and writing the mesh formats a collection is usually made from, and handed back to.

fabriks is a wire format, not an importer: :func:`fabriks.build_collection` takes geometry that is
already in memory, and :class:`fabriks.Collection` hands geometry back the same way. This package
is the layer either side of that -- point it at a ``.glb`` and get a collection, point it at a
collection and get a ``.obj``::

    from obstore.store import LocalStore
    from fabriks.contrib import read_glb, write_obj
    import fabriks

    imported = read_glb("bunny.glb", LocalStore("/data"), "bunny", levels=4)

    collection = fabriks.open_collection(LocalStore("/data"), "bunny")
    blob = write_obj(collection, level=2)

It needs **nothing installed that fabriks does not already require**. trimesh is a hard dependency
-- ``slice_mesh_plane`` is how a mesh is cut at the cell planes -- and its exchange layer reads and
writes every format below, so there is no extra here to forget.

What the adapters do beyond calling trimesh
-------------------------------------------
Three things, and each of them is a bug that looks like working code if it is skipped.

**They fit the model into the octant fabriks partitions.** A collection lives in a positive,
whole-number voxel space; a mesh file is usually centred on the origin and a couple of units
across. Built as it stands, the first is refused outright and the second collapses into a single
cell with no level of detail at all. So a read scales and shifts, records exactly what it did, and
a write undoes it -- see :mod:`fabriks.contrib.fit` and :mod:`fabriks.contrib.catalog`.

**They bake the scene graph.** ``scene.geometry`` holds the mesh a node *points at*, not where the
node puts it, so reading it directly stacks every object on the origin -- with the right vertex
counts and a plausible-looking result. See :mod:`fabriks.contrib.scene`.

**They keep identity.** OBJ groups and glTF nodes are named strings; a collection keys objects by
integer. A name that is already a non-negative integer keeps its value, so a collection written out
and read back comes home with the ids it left with; anything else is assigned the lowest free id
and its string is carried in the object catalog's ``name`` column.

What is supported
-----------------
=========  ==================  ===================================================================
Format     Identity            Notes
=========  ==================  ===================================================================
``GLB``    per object          One file; reads from bytes or a path. The best round trip here.
``GLTF``   per object          Several files -- see :mod:`fabriks.contrib.gltf` about buffers.
``OBJ``    per object          Identity rides on ``o`` markers; materials are skipped.
``PLY``    single object       No usable object names; a write merges. :mod:`fabriks.contrib.meshes`
``STL``    single object       Triangles and nothing else; a write merges.
``OFF``    single object       Vertices and faces and nothing else; a write merges.
=========  ==================  ===================================================================

**3MF and DAE are deliberately absent.** trimesh reads and writes neither without a package fabriks
does not depend on -- ``networkx`` for 3MF, ``pycollada`` for DAE -- and a function that is always
an ``ImportError`` is a worse contract than a function that is not there.

A round trip is not exact
-------------------------
Positions are stored as 16-bit values quantized **per cell**, so geometry comes back within about
one quantum of where it went in, never on it. At level 0 that quantum is ``cell_size / 65535``
voxels; coarser levels have also been decimated, which is a much larger difference and the point of
the format. Compare bounds and shape, not vertex arrays -- ``Collection.object_mesh`` welds the
pieces of an object back together, so neither vertex count nor vertex order survives either.

This package is ``fabriks.contrib`` and not ``fabriks``: ``import fabriks`` does not pull it in, so
the majority of callers who never touch a foreign format do not pay trimesh's exchange machinery at
import time. Reach for it by name -- ``from fabriks.contrib import read_glb``.
"""

from fabriks.contrib.catalog import fit_of, names_of
from fabriks.contrib.fit import DEFAULT_RESOLUTION, VoxelFit
from fabriks.contrib.gltf import read_glb, read_gltf, write_glb, write_gltf
from fabriks.contrib.meshes import read_off, read_ply, read_stl, write_off, write_ply, write_stl
from fabriks.contrib.obj import read_obj, write_obj
from fabriks.contrib.scene import Imported, ReadOptions, Source, WriteOptions

__all__ = [
    "DEFAULT_RESOLUTION",
    "Imported",
    "ReadOptions",
    "Source",
    "VoxelFit",
    "WriteOptions",
    "fit_of",
    "names_of",
    "read_glb",
    "read_gltf",
    "read_obj",
    "read_off",
    "read_ply",
    "read_stl",
    "write_glb",
    "write_gltf",
    "write_obj",
    "write_off",
    "write_ply",
    "write_stl",
]
