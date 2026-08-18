"""The bridge: a ``trimesh.Scene`` on one side, a mesh collection on the other.

Every ``read_*`` and every ``write_*`` in this package is a thin call into this module with a
format name attached. The format-specific parts are which loader kwargs to pass and what the
bytes are called; everything that is actually hard is here and happens once.

Three things happen on the way in, and each of them is a bug if it does not:

**Node transforms are baked.** ``scene.geometry`` hands back the geometry a node *refers to*, not
where the node puts it -- a box placed ten units along x still reports bounds ``[-0.5, 0.5]``
there. So the scene graph is walked and each node's 4x4 is applied to its own copy of the mesh.
Reading ``scene.geometry`` directly would pile every object onto the origin, and the result looks
plausible enough to ship.

**The model is fitted into the voxel octant.** See :mod:`fabriks.contrib.fit` for why it has to
be, and :mod:`fabriks.contrib.catalog` for where the fit is then kept.

**Names become object ids.** A collection keys objects by integer; OBJ groups and glTF nodes are
strings. A name that is already a non-negative integer keeps its value, so a collection exported
and re-imported comes back with the ids it started with; anything else is assigned the lowest free
id and its string is written into the catalog, so nothing is lost either way.

On the way out an object is reassembled whole through ``Collection.object_mesh`` rather than cell
by cell, because a cell is an artifact of the octree and an object is what the source file had.
"""

from __future__ import annotations

import io
import os
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import IO, Any, TypedDict, Unpack, cast

import numpy as np
import trimesh

from fabriks.build import build_collection
from fabriks.contrib import catalog
from fabriks.contrib.fit import DEFAULT_RESOLUTION, VoxelFit
from fabriks.errors import FormatError
from fabriks.manifest import Decimation, Manifest
from fabriks.reader import Collection
from fabriks.simplifiers import Simplifier
from fabriks.sizing import plan_grid
from fabriks.sources import Mesh
from fabriks.stores import FabriksStore
from fabriks.writer import write_collection

#: What a caller may hand a reader as the file to read.
Source = str | os.PathLike[str] | bytes | bytearray | IO[bytes]


class ReadOptions(TypedDict, total=False):
    """The keyword arguments every ``read_*`` accepts, over and above source, store and prefix.

    ``fit``
        The transform to apply, or None to fit automatically at ``resolution``. Pass
        :meth:`VoxelFit.identity` for a model already in voxel coordinates.
    ``resolution``
        How many voxels the longest side of an auto-fitted model spans. Ignored when ``fit`` is
        given.
    ``cell_size`` / ``levels``
        The octree, straight through to :func:`fabriks.build_collection`. Left out, the grid comes
        from :func:`fabriks.plan_grid`, which budgets by bytes -- so a small model can come back
        with one level and no pyramid at all. Pass ``levels`` when coarse levels are the point.
    ``axes``
        What the three vertex slots mean, if the caller is in a position to say. Nothing is
        declared by default; see :mod:`fabriks.contrib`.
    ``simplifier`` / ``decimation`` / ``codec`` / ``compression``
        Straight through to :func:`fabriks.build_collection`.
    """

    fit: VoxelFit | None
    resolution: int
    cell_size: Sequence[int] | None
    levels: int | None
    axes: Sequence[str] | None
    simplifier: Simplifier | str | None
    decimation: Decimation | None
    codec: str | None
    compression: str | None


class WriteOptions(TypedDict, total=False):
    """The keyword arguments every ``write_*`` accepts, over and above the collection and target.

    ``level``
        Which level of detail to export. 0, the default, is full detail and the least lossy.
    ``objects``
        Which object ids to export, or None for all of them.
    ``fit``
        The transform to invert, or None to use the one the collection recorded at import. Pass
        :meth:`VoxelFit.identity` to export raw voxel coordinates.
    """

    level: int
    objects: Sequence[int] | None
    fit: VoxelFit | None


@dataclass(frozen=True)
class Imported:
    """What a ``read_*`` produced: the manifest, the fit it applied, and the names it kept.

    The fit and the names are also written into the collection's object catalog, so an export
    does not need this object -- it is what the caller wants *at the call site*, where the
    interesting question is usually "what did it do to my coordinates".
    """

    #: The manifest as written, which is the one :func:`fabriks.write_collection` returned.
    manifest: Manifest
    #: The transform applied to get the source into the voxel octant.
    fit: VoxelFit
    #: ``{object_id: name}`` for every object that had a name in the source.
    names: dict[int, str]

    def __len__(self) -> int:
        """How many objects were written."""
        return int((self.manifest.counts or {}).get("objects", len(self.names)))


# -- reading ----------------------------------------------------------------------- #


def load_scene(source: Source, file_type: str, **kwargs: Any) -> trimesh.Scene:  # noqa: ANN401 - loader kwargs differ per format
    """Load a file of a known type as a scene, whatever the caller handed over.

    A path is passed to trimesh as a path on purpose: that is what lets its resolver find the
    sibling files a multi-file format refers to. Bytes have no directory to resolve against, so a
    format that keeps its buffers outside the file can only be read from a path -- which is what
    the error says, rather than letting a missing buffer surface as a stack trace from inside a
    loader.
    """
    try:
        if isinstance(source, (bytes, bytearray)):
            loaded = trimesh.load(io.BytesIO(bytes(source)), file_type=file_type, **kwargs)
        elif isinstance(source, (str, os.PathLike)):
            loaded = trimesh.load(os.fspath(source), file_type=file_type, **kwargs)
        else:
            loaded = trimesh.load(source, file_type=file_type, **kwargs)
    except FileNotFoundError:
        raise
    except Exception as error:
        hint = (
            " A `.gltf` keeps its buffers in sibling `.bin` files, which can only be resolved from"
            " a path -- pass the file's path rather than its bytes."
            if file_type == "gltf" and not isinstance(source, (str, os.PathLike))
            else ""
        )
        raise FormatError(f"Could not read this as {file_type.upper()}: {error}.{hint}") from error

    return as_scene(loaded)


def as_scene(loaded: Any) -> trimesh.Scene:  # noqa: ANN401 - whatever a loader returned
    """Whatever a loader returned, as a scene, so there is one path below this point.

    A format that carries no object names -- STL has none at all, and PLY has none in practice --
    loads as a single mesh, which becomes a one-node scene here rather than a second code path.
    """
    if isinstance(loaded, trimesh.Scene):
        return loaded
    vertices = getattr(loaded, "vertices", None)
    faces = getattr(loaded, "faces", None)
    if vertices is None or faces is None:
        raise FormatError(
            f"This file loaded as {type(loaded).__name__}, which carries no vertices and faces. "
            f"fabriks partitions surfaces, so a point cloud or a curve has nothing to write."
        )
    scene = trimesh.Scene()
    scene.add_geometry(loaded, geom_name=str(getattr(loaded, "metadata", {}).get("name", "mesh")))
    return scene


def baked_geometries(scene: trimesh.Scene) -> dict[str, trimesh.Trimesh]:
    """Every geometry in the scene, with its node's transform applied.

    ``scene.geometry`` is deliberately *not* what this returns: it holds the untransformed mesh a
    node points at, so a scene whose objects are placed by node transforms would come back with
    every object stacked on the origin.
    """
    baked: dict[str, trimesh.Trimesh] = {}
    for node in scene.graph.nodes_geometry:
        transform, name = scene.graph[node]
        geometry = scene.geometry.get(name)
        if geometry is None or not hasattr(geometry, "faces"):
            continue
        placed = geometry.copy()
        placed.apply_transform(transform)
        # A geometry instanced under several nodes appears once per node; the node name keeps
        # them apart, because collapsing them would silently drop every copy but one.
        key = str(name)
        baked[key if key not in baked else str(node)] = cast("trimesh.Trimesh", placed)
    return baked


def assign_ids(names: Sequence[str]) -> dict[str, int]:
    """Map source names to object ids, keeping the ones that are already integers.

    A collection exported to a file and read back therefore comes home with the ids it left with,
    which is the only reason a round trip preserves identity rather than merely count.
    """
    taken: set[int] = set()
    integral: dict[str, int] = {}
    for name in names:
        if name.isdigit():
            value = int(name)
            if value not in taken:
                taken.add(value)
                integral[name] = value

    assigned: dict[str, int] = {}
    candidate = 0
    for name in names:
        if name in assigned:
            continue
        if name in integral:
            assigned[name] = integral[name]
            continue
        while candidate in taken:
            candidate += 1
        taken.add(candidate)
        assigned[name] = candidate
    return assigned


def scene_to_objects(
    scene: trimesh.Scene, *, fit: VoxelFit
) -> tuple[dict[int, Mesh], dict[int, str]]:
    """Bake a scene's node transforms, fit it into voxels, and key it by object id."""
    baked = baked_geometries(scene)
    if not baked:
        raise FormatError("This file holds no triangulated geometry, so there is nothing to write.")

    ids = assign_ids(list(baked))
    objects: dict[int, Mesh] = {}
    names: dict[int, str] = {}
    for name, geometry in baked.items():
        object_id = ids[name]
        objects[object_id] = Mesh(
            vertices=fit.apply(np.asarray(geometry.vertices, dtype=np.float64)),
            faces=np.asarray(geometry.faces, dtype=np.int64),
        )
        names[object_id] = name
    return objects, names


def fit_for(scene: trimesh.Scene, fit: VoxelFit | None, resolution: int) -> VoxelFit:
    """The fit to use: the caller's if they gave one, otherwise one derived from the bounds."""
    if fit is not None:
        return fit
    bounds = scene.bounds
    if bounds is None:
        return VoxelFit.identity()
    return VoxelFit.for_bounds(bounds[0], bounds[1], resolution=resolution)


def import_scene(
    scene: trimesh.Scene,
    store: FabriksStore,
    prefix: str = "",
    **options: Unpack[ReadOptions],
) -> Imported:
    """Fit, build, name and write a scene as a collection. Every ``read_*`` ends here."""
    fit = fit_for(scene, options.get("fit"), options.get("resolution", DEFAULT_RESOLUTION))
    objects, names = scene_to_objects(scene, fit=fit)

    cell_size = options.get("cell_size")
    levels = options.get("levels")
    if cell_size is None or levels is None:
        planned = plan_grid(objects).as_kwargs()
        cell_size = cell_size if cell_size is not None else planned["cell_size"]
        levels = levels if levels is not None else planned["levels"]
        if levels == 1 and options.get("levels") is None:
            warnings.warn(
                f"This model was sized to a single level: `plan_grid` budgets by bytes, and at "
                f"cell_size={tuple(cell_size)} the whole model fits in one layer. A collection "
                f"with one level is a valid collection but it carries no level of detail, which "
                f"is what the format is for -- pass `levels=` to ask for a pyramid, or "
                f"`resolution=` to fit the model into a larger voxel box first.",
                stacklevel=3,
            )

    built = build_collection(
        objects,
        cell_size=cell_size,
        levels=levels,
        axes=options.get("axes"),
        **_encoding_kwargs(options),
    )
    catalog.attach(built, names=names, fit=fit)
    manifest = write_collection(built, store, prefix)
    return Imported(manifest=manifest, fit=fit, names=names)


def _encoding_kwargs(options: ReadOptions) -> dict[str, Any]:
    """The build arguments that are simply passed through, minus the ones left unset."""
    from fabriks.manifest import CODEC_NONE, COMPRESSION_NONE

    return {
        "codec": options.get("codec") or CODEC_NONE,
        "compression": options.get("compression") or COMPRESSION_NONE,
        "simplifier": options.get("simplifier"),
        "decimation": options.get("decimation"),
    }


def read_source(
    source: Source,
    store: FabriksStore,
    prefix: str,
    file_type: str,
    loader: Mapping[str, Any] | None = None,
    **options: Unpack[ReadOptions],
) -> Imported:
    """Load a file and import it, which is the whole of what every ``read_*`` does."""
    return import_scene(load_scene(source, file_type, **(loader or {})), store, prefix, **options)


# -- writing ----------------------------------------------------------------------- #


def collection_to_scene(collection: Collection, **options: Unpack[WriteOptions]) -> trimesh.Scene:
    """Reassemble a collection at one level as a named scene in source coordinates.

    Each object is rebuilt whole through ``Collection.object_mesh`` rather than gathered cell by
    cell: a cell is how the octree stores a surface, and an object is what the source file had.

    An object with no geometry at the requested level -- one a coarse level decimated away
    entirely -- is skipped and reported through :mod:`warnings`, the same channel the builder uses
    when decimation falls short. A silently short export is the failure worth avoiding here.
    """
    level = options.get("level", 0)
    fit = options.get("fit")
    if fit is None:
        fit = catalog.fit_of(collection) or VoxelFit.identity()
    names = catalog.names_of(collection)

    wanted = options.get("objects")
    ids = [int(item) for item in wanted] if wanted is not None else sorted(collection.objects)

    scene = trimesh.Scene()
    missing: list[int] = []
    for object_id in ids:
        try:
            mesh = collection.object_mesh(object_id, level=level)
        except KeyError:
            missing.append(object_id)
            continue
        scene.add_geometry(
            trimesh.Trimesh(
                vertices=fit.invert(mesh.vertices),
                faces=np.asarray(mesh.faces, dtype=np.int64),
                process=False,
            ),
            geom_name=names.get(object_id, str(object_id)),
        )

    if missing:
        warnings.warn(
            f"{len(missing)} object(s) have no geometry at level {level} and were left out of this "
            f"export: {missing}. A coarse level drops an object that decimating would have removed "
            f"altogether -- export at a finer level, or pass `objects=` to say which ones matter.",
            stacklevel=3,
        )
    if not scene.geometry:
        raise FormatError(
            f"Nothing was exported: no requested object has geometry at level {level}. Level 0 is "
            f"full detail, and levels above it are progressively coarser."
        )
    return scene


def export_scene(
    scene: trimesh.Scene,
    file_type: str,
    target: str | os.PathLike[str] | None,
    *,
    merge: bool = False,
) -> dict[str, bytes]:
    """Serialize a scene, and write it beside ``target`` when there is one.

    Always a mapping of filename to bytes, because ``.gltf`` genuinely is several files -- the
    JSON plus a ``.bin`` per buffer. Single-file formats come back as a mapping of one, and the
    per-format wrappers unwrap it.

    ``merge`` is for the formats that carry no object names: PLY and STL write one mesh whatever
    they are handed, and OFF has no scene exporter at all (``ValueError: unsupported export
    format: off``). Concatenating first makes that explicit here rather than leaving it as a
    surprise inside trimesh.
    """
    exported = (
        _merged(scene).export(file_type=file_type) if merge else scene.export(file_type=file_type)
    )
    if isinstance(exported, dict):
        files = {str(name): _as_bytes(body) for name, body in exported.items()}
    else:
        files = {f"model.{file_type}": _as_bytes(exported)}

    if target is not None:
        _write_files(files, target, file_type)
    return files


def _merged(scene: trimesh.Scene) -> trimesh.Trimesh:
    """Every geometry in the scene as one mesh, for a format that cannot hold more than one."""
    geometries = list(scene.geometry.values())
    if len(geometries) == 1:
        return cast("trimesh.Trimesh", geometries[0])
    return cast("trimesh.Trimesh", trimesh.util.concatenate(geometries))


def _as_bytes(body: Any) -> bytes:  # noqa: ANN401 - an exporter returns str or bytes per format
    """An exporter's output as bytes, whichever of the two it handed back."""
    if isinstance(body, (bytes, bytearray)):
        return bytes(body)
    return str(body).encode()


def _write_files(files: Mapping[str, bytes], target: str | os.PathLike[str], file_type: str) -> None:
    """Write an export to disk: the model at ``target``, and any sibling buffers beside it."""
    from pathlib import Path

    path = Path(os.fspath(target))
    path.parent.mkdir(parents=True, exist_ok=True)
    principal = _principal(files, file_type)
    for name, body in files.items():
        (path if name == principal else path.parent / name).write_bytes(body)


def _principal(files: Mapping[str, bytes], file_type: str) -> str:
    """Which of an export's files is the model itself rather than one of its buffers."""
    for name in files:
        if name.lower().endswith(f".{file_type}"):
            return name
    return next(iter(files))


def write_scene(
    collection: Collection,
    target: str | os.PathLike[str] | None,
    file_type: str,
    *,
    merge: bool = False,
    **options: Unpack[WriteOptions],
) -> dict[str, bytes]:
    """Reassemble and serialize, which is the whole of what every ``write_*`` does."""
    scene = collection_to_scene(collection, **options)
    return export_scene(scene, file_type, target, merge=merge)


def single_file(files: Mapping[str, bytes], file_type: str) -> bytes:
    """The one blob a single-file format produced."""
    return files[_principal(files, file_type)]


__all__ = [
    "Imported",
    "ReadOptions",
    "Source",
    "WriteOptions",
    "as_scene",
    "assign_ids",
    "baked_geometries",
    "collection_to_scene",
    "export_scene",
    "fit_for",
    "import_scene",
    "load_scene",
    "read_source",
    "scene_to_objects",
    "single_file",
    "write_scene",
]
