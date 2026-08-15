"""The store a collection is written into, and the little that fabriks asks of one.

A collection is **a tree, not a file**: a manifest, two catalogs and one directory per
octree level. That shape is why the store is a parameter rather than a path -- the same tree
has to land on a local disk during development and in an S3 prefix in production, and nothing
above this package should care which.

What fabriks asks for is three methods::

    store.put(path, data)     # bytes -> path, relative to the store's own root
    store.get(path)           # -> bytes, or anything with a .bytes() method
    store.list(prefix)        # -> the paths under a prefix

and one it will *use if it is there*::

    store.get_range(path, start=..., length=...)   # -> the bytes in that window

That is deliberately the shape obstore already has, so ``S3Store``, ``LocalStore``,
``GCSStore``, ``AzureStore`` and ``MemoryStore`` are all usable **as they are** -- there is no
adapter, and obstore is not a dependency of fabriks. :class:`DirectoryStore` is here so that a
plain filesystem path works with no dependencies at all.

``get_range`` is optional because it is the one method a hand-rolled store is likely to be
missing, and its absence must degrade rather than fail: :func:`get_range_bytes` falls back to
fetching the whole object and slicing it, which is exactly what fabriks did everywhere before
range reads existed. What it buys when present is the whole point of the format -- reading one
cell out of a level costs one row group rather than the level.

Paths are always ``/``-joined and relative to the store, never absolute and never
``..``-relative: a collection names files inside its own tree, and a writer that could escape
it would be a path-traversal surface in whatever is holding the credentials.

The package is split the way that contract is: :mod:`~fabriks.stores.protocol` states what a
store must offer, :mod:`~fabriks.stores.access` is how fabriks calls through it,
:mod:`~fabriks.stores.file` is the seekable view pyarrow needs, and
:mod:`~fabriks.stores.directory` and :mod:`~fabriks.stores.memory` are the two implementations
that ship.
"""

from fabriks.stores.access import (
    aget_bytes,
    aget_range_bytes,
    get_bytes,
    get_range_bytes,
    list_paths,
    put_bytes,
)
from fabriks.stores.directory import DirectoryStore
from fabriks.stores.file import StoreFile
from fabriks.stores.memory import MemoryStore
from fabriks.stores.protocol import (
    AsyncReadable,
    FabriksStore,
    RangeReadable,
    join,
    validate_relative,
)

__all__ = [
    "AsyncReadable",
    "DirectoryStore",
    "FabriksStore",
    "MemoryStore",
    "RangeReadable",
    "StoreFile",
    "aget_bytes",
    "aget_range_bytes",
    "get_bytes",
    "get_range_bytes",
    "join",
    "list_paths",
    "put_bytes",
    "validate_relative",
]
