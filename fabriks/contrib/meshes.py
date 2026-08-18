"""PLY, STL and OFF -- the formats that hold a surface but not who it belongs to.

These three carry no usable notion of a named object, and pretending otherwise would be the
worst thing this package could do. STL is a bag of triangles with no grouping at all; PLY has
elements but no object names that survive a trimesh round trip; OFF is vertices and faces and
nothing else -- trimesh has no scene exporter for it at all (``ValueError: unsupported export
format: off``).

So these adapters are **single-object** by construction, and say so rather than looking like the
OBJ and glTF ones and quietly behaving differently:

- reading gives a collection with exactly one object;
- writing **merges** every selected object into one mesh first.

Use OBJ or GLB when identity matters. These are here because a viewer, a scanner or a mesher
often emits one of them and converting through a third tool to get an OBJ is a silly step.
"""

from __future__ import annotations

import os
from typing import Unpack

from fabriks.contrib import scene as _scene
from fabriks.contrib.scene import Imported, ReadOptions, Source, WriteOptions
from fabriks.reader import Collection
from fabriks.stores import FabriksStore

#: The file types trimesh knows these formats by.
FILE_TYPE_PLY = "ply"
FILE_TYPE_STL = "stl"
FILE_TYPE_OFF = "off"


def read_ply(
    source: Source, store: FabriksStore, prefix: str = "", **options: Unpack[ReadOptions]
) -> Imported:
    """Read a PLY file into a collection holding one object.

    See :class:`~fabriks.contrib.scene.ReadOptions` for the keyword arguments.
    """
    return _scene.read_source(source, store, prefix, FILE_TYPE_PLY, None, **options)


def write_ply(
    collection: Collection,
    target: str | os.PathLike[str] | None = None,
    **options: Unpack[WriteOptions],
) -> bytes:
    """Write a collection out as one PLY, merging every selected object into one mesh.

    See :class:`~fabriks.contrib.scene.WriteOptions` for the keyword arguments.
    """
    written = _scene.write_scene(collection, target, FILE_TYPE_PLY, merge=True, **options)
    return _scene.single_file(written, FILE_TYPE_PLY)


def read_stl(
    source: Source, store: FabriksStore, prefix: str = "", **options: Unpack[ReadOptions]
) -> Imported:
    """Read an STL file into a collection holding one object.

    See :class:`~fabriks.contrib.scene.ReadOptions` for the keyword arguments.
    """
    return _scene.read_source(source, store, prefix, FILE_TYPE_STL, None, **options)


def write_stl(
    collection: Collection,
    target: str | os.PathLike[str] | None = None,
    **options: Unpack[WriteOptions],
) -> bytes:
    """Write a collection out as one binary STL, merging every selected object into one mesh.

    See :class:`~fabriks.contrib.scene.WriteOptions` for the keyword arguments.
    """
    written = _scene.write_scene(collection, target, FILE_TYPE_STL, merge=True, **options)
    return _scene.single_file(written, FILE_TYPE_STL)


def read_off(
    source: Source, store: FabriksStore, prefix: str = "", **options: Unpack[ReadOptions]
) -> Imported:
    """Read an OFF file into a collection holding one object.

    See :class:`~fabriks.contrib.scene.ReadOptions` for the keyword arguments.
    """
    return _scene.read_source(source, store, prefix, FILE_TYPE_OFF, None, **options)


def write_off(
    collection: Collection,
    target: str | os.PathLike[str] | None = None,
    **options: Unpack[WriteOptions],
) -> bytes:
    """Write a collection out as one OFF, merging every selected object into one mesh.

    See :class:`~fabriks.contrib.scene.WriteOptions` for the keyword arguments.
    """
    written = _scene.write_scene(collection, target, FILE_TYPE_OFF, merge=True, **options)
    return _scene.single_file(written, FILE_TYPE_OFF)


__all__ = [
    "FILE_TYPE_OFF",
    "FILE_TYPE_PLY",
    "FILE_TYPE_STL",
    "read_off",
    "read_ply",
    "read_stl",
    "write_off",
    "write_ply",
    "write_stl",
]
