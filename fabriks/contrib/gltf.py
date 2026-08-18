"""glTF and its binary container GLB, which keep object identity in node names.

The two are one format with two packagings, and the difference is the whole of what this module
has to care about. **GLB is one file** -- geometry, buffers and all -- so it reads from bytes as
happily as from a path. **glTF is a JSON document that points at its buffers**, usually sibling
``.bin`` files, so reading one from a bare blob can only work when those buffers are inlined as
``data:`` URIs. Handed a blob whose buffers are external, trimesh fails with ``TypeError:
'NoneType' object is not subscriptable`` from somewhere deep inside the loader, which tells a
caller nothing -- so the document is checked here first and the refusal says what to pass instead.

Because a glTF is several files, :func:`write_gltf` returns a mapping of filename to bytes rather
than one blob, and :func:`read_gltf` accepts that same mapping straight back. That is the round
trip that never touches a disk.

**No axis convention is applied here.** glTF is a Y-up format, and it would be easy to read that
as "so slot 1 is y" and permute. fabriks addresses components by position and declares nothing
about what they mean, and trimesh's writer emits no correction node either, so bytes in and bytes
out are coordinate-identical. A caller who owns that coordinate system says so with ``axes=``.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from typing import Unpack

from fabriks.contrib import scene as _scene
from fabriks.contrib.scene import Imported, ReadOptions, Source, WriteOptions
from fabriks.errors import FormatError
from fabriks.reader import Collection
from fabriks.stores import FabriksStore

#: The file type trimesh knows the JSON packaging by.
FILE_TYPE_GLTF = "gltf"
#: The file type trimesh knows the binary packaging by.
FILE_TYPE_GLB = "glb"


def read_glb(
    source: Source,
    store: FabriksStore,
    prefix: str = "",
    **options: Unpack[ReadOptions],
) -> Imported:
    """Read a GLB file into a collection, one object per named node.

    GLB is self-contained, so bytes and a path are equally good here.

    See :class:`~fabriks.contrib.scene.ReadOptions` for the keyword arguments.
    """
    return _scene.read_source(source, store, prefix, FILE_TYPE_GLB, None, **options)


def write_glb(
    collection: Collection,
    target: str | os.PathLike[str] | None = None,
    **options: Unpack[WriteOptions],
) -> bytes:
    """Write a collection's objects out as one GLB, each as its own named node.

    See :class:`~fabriks.contrib.scene.WriteOptions` for the keyword arguments.
    """
    return _scene.single_file(
        _scene.write_scene(collection, target, FILE_TYPE_GLB, **options), FILE_TYPE_GLB
    )


def read_gltf(
    source: Source | Mapping[str, bytes],
    store: FabriksStore,
    prefix: str = "",
    **options: Unpack[ReadOptions],
) -> Imported:
    """Read a glTF document into a collection, one object per named node.

    Three shapes of source work, and the difference between them is only ever about buffers:

    - a **path**, which lets trimesh resolve the sibling ``.bin`` files off disk;
    - the **mapping** :func:`write_gltf` returns, whose entries resolve each other;
    - **bytes**, which work only for a self-contained document -- one whose buffers are inlined
      as ``data:`` URIs. Anything else is refused here, by name, rather than deep in a loader.

    See :class:`~fabriks.contrib.scene.ReadOptions` for the keyword arguments.
    """
    if isinstance(source, Mapping):
        document, resolver = _unpack(source)
        return _scene.import_scene(
            _scene.load_scene(document, FILE_TYPE_GLTF, resolver=resolver), store, prefix, **options
        )

    if isinstance(source, (bytes, bytearray)):
        _refuse_external_buffers(bytes(source))
    return _scene.read_source(source, store, prefix, FILE_TYPE_GLTF, None, **options)


def write_gltf(
    collection: Collection,
    target: str | os.PathLike[str] | None = None,
    **options: Unpack[WriteOptions],
) -> dict[str, bytes]:
    """Write a collection's objects out as a glTF document and its buffers.

    The return is ``{filename: bytes}`` -- the ``.gltf`` plus one ``.bin`` per buffer -- because
    that is genuinely what this format is. Given a ``target``, the document is written there and
    its buffers land beside it. The mapping goes straight back into :func:`read_gltf`.

    See :class:`~fabriks.contrib.scene.WriteOptions` for the keyword arguments.
    """
    return _scene.write_scene(collection, target, FILE_TYPE_GLTF, **options)


def _unpack(source: Mapping[str, bytes]) -> tuple[bytes, dict[str, bytes]]:
    """Split a written glTF back into its document and the buffers that resolve against it."""
    names = [name for name in source if name.lower().endswith(".gltf")]
    if not names:
        raise FormatError(
            f"This mapping holds no `.gltf` document, only {sorted(source)}. `write_gltf` returns "
            f"the document alongside its buffers, and both halves are needed to read it back."
        )
    document = names[0]
    return bytes(source[document]), {
        name: bytes(body) for name, body in source.items() if name != document
    }


def _refuse_external_buffers(body: bytes) -> None:
    """Refuse a blob whose buffers live in files it has no way to reach."""
    try:
        document = json.loads(body)
    except ValueError as error:
        raise FormatError(f"This is not a readable glTF document: {error}.") from error

    external = [
        buffer["uri"]
        for buffer in document.get("buffers", [])
        if isinstance(buffer, dict)
        and isinstance(buffer.get("uri"), str)
        and not buffer["uri"].startswith("data:")
    ]
    if external:
        raise FormatError(
            f"This .gltf names an external buffer ({external[0]!r}), and a bytes blob has nothing "
            f"to resolve it against. Pass the file's path so its siblings can be found, pass the "
            f"`{{name: bytes}}` mapping `write_gltf` returns, or use `read_glb` -- GLB is one file "
            f"by construction."
        )


__all__ = [
    "FILE_TYPE_GLB",
    "FILE_TYPE_GLTF",
    "read_glb",
    "read_gltf",
    "write_glb",
    "write_gltf",
]
