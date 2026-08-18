"""Wavefront OBJ, which keeps object identity in ``o`` markers.

OBJ is the format where the round trip is *nearly* free and then is not. trimesh's exporter
writes an ``o <name>`` line per geometry, and its loader will split on those lines -- but only
when asked, and the kwarg that asks has been spelled two ways. Loading without it merges every
object in the file into one mesh, and a merged mesh is not a failure a caller can see: the
geometry is all there, the vertex count is right, and only the identity is gone.
"""

from __future__ import annotations

import inspect
import os
from typing import Unpack

import trimesh
from trimesh.exchange.obj import load_obj

from fabriks.contrib import scene as _scene
from fabriks.contrib.scene import Imported, ReadOptions, Source, WriteOptions
from fabriks.errors import FormatError
from fabriks.reader import Collection
from fabriks.stores import FabriksStore

#: The file type trimesh knows this format by.
FILE_TYPE = "obj"


def _split_kwarg() -> str:
    """Which spelling of the split-into-objects kwarg the installed trimesh takes.

    ``split_objects`` in trimesh 5, ``split_object`` in 4, and fabriks's floor is ``>=4.0``, so
    the name is asked for rather than assumed. Probing beats raising the floor: every other
    format in this package works on both, and a floor bump would push an upgrade on callers who
    never read an OBJ.

    Neither name present is raised on rather than worked around, because the fallback -- load it
    anyway -- returns one merged object, which is indistinguishable from a file that genuinely
    held one.
    """
    parameters = inspect.signature(load_obj).parameters
    for name in ("split_objects", "split_object"):
        if name in parameters:
            return name
    raise FormatError(
        f"This trimesh ({trimesh.__version__}) takes neither `split_objects` nor `split_object` "
        f"on its OBJ loader, so an OBJ's `o` markers cannot be split into separate objects and "
        f"every object in the file would silently become one. Reading OBJ needs a trimesh whose "
        f"loader takes one of those."
    )


def read_obj(
    source: Source,
    store: FabriksStore,
    prefix: str = "",
    **options: Unpack[ReadOptions],
) -> Imported:
    """Read an OBJ file into a collection, one object per ``o`` marker.

    Materials are skipped: fabriks writes surfaces, and an ``.mtl`` beside the file would only
    be parsed to be discarded. Grouping by material is off for the same reason -- it would cut
    one object into several along a boundary the geometry does not have.

    See :class:`~fabriks.contrib.scene.ReadOptions` for the keyword arguments.
    """
    loader = {_split_kwarg(): True, "group_material": False, "skip_materials": True}
    return _scene.read_source(source, store, prefix, FILE_TYPE, loader, **options)


def write_obj(
    collection: Collection,
    target: str | os.PathLike[str] | None = None,
    **options: Unpack[WriteOptions],
) -> bytes:
    """Write a collection's objects out as one OBJ, each under its own ``o`` marker.

    See :class:`~fabriks.contrib.scene.WriteOptions` for the keyword arguments.
    """
    return _scene.single_file(
        _scene.write_scene(collection, target, FILE_TYPE, **options), FILE_TYPE
    )


__all__ = ["FILE_TYPE", "read_obj", "write_obj"]
