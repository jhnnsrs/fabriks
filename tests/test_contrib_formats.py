"""One round trip per format, and what each format can actually promise.

The claim being tested is not "the bytes came back" but "the *objects* came back": the same ids,
under the same names, in the same places. OBJ and glTF can promise that; PLY, STL and OFF carry no
object names and cannot, so their tests assert the merge rather than pretending otherwise.

Tolerances are derived, never hardcoded. Positions are 16-bit quantized per cell, so one quantum
at level 0 is ``cell_size / QUANT_MAX`` voxels, and in source units that is divided by the fit's
scale. A constant tuned to today's fixture would stop meaning anything the moment the fixture
changed.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
import trimesh

import fabriks
from fabriks.contrib import (
    VoxelFit,
    fit_of,
    read_glb,
    read_gltf,
    read_obj,
    read_off,
    read_ply,
    read_stl,
    write_glb,
    write_gltf,
    write_obj,
    write_off,
    write_ply,
    write_stl,
)
from fabriks.contrib.scene import baked_geometries

CELL_SIZE = (128, 128, 128)
LEVELS = 3

#: The formats that keep one object one object, and the pair of functions that do it.
NAMED = {"obj": (read_obj, write_obj), "glb": (read_glb, write_glb)}
#: The formats that hold a surface but not who it belongs to.
MERGING = {"ply": (read_ply, write_ply), "stl": (read_stl, write_stl), "off": (read_off, write_off)}


@pytest.fixture()
def source():
    """A two-object scene: one name that is already an id, one that is not."""
    scene = trimesh.Scene()
    scene.add_geometry(trimesh.creation.icosphere(radius=0.6, subdivisions=3), geom_name="head")
    scene.add_geometry(
        trimesh.creation.box(extents=[1.0, 0.4, 0.3]),
        geom_name="7",
        transform=trimesh.transformations.translation_matrix([0.8, 0.0, 0.0]),
    )
    return scene


@pytest.fixture()
def written(source):
    """The scene read in through GLB, as an opened collection."""
    store = fabriks.MemoryStore()
    read_glb(source.export(file_type="glb"), store, "c", cell_size=CELL_SIZE, levels=LEVELS)
    return fabriks.open_collection(store, "c")


def quantum(collection):
    """One level-0 position quantum, in the source's own units.

    ``cell_size / QUANT_MAX`` is the finest distance the format can represent inside a cell; the
    fit's scale carries that back into the units the file was written in.
    """
    fit = fit_of(collection) or VoxelFit.identity()
    return max(collection.grid.cell_size) / fabriks.QUANT_MAX / fit.scale


def reload(blob, file_type):
    """Read exported bytes back into trimesh, splitting objects where the format has them."""
    kwargs = (
        {"split_objects": True, "group_material": False, "skip_materials": True}
        if file_type == "obj"
        else {}
    )
    loaded = trimesh.load(io.BytesIO(blob), file_type=file_type, **kwargs)
    return loaded if isinstance(loaded, trimesh.Scene) else trimesh.Scene(loaded)


@pytest.mark.parametrize("file_type", sorted(NAMED))
def test_two_objects_come_back_under_their_own_names(file_type, source, written):
    """The point of the whole package: identity survives a trip out to a file and home again.

    A merged export loses nothing visible -- the geometry is all there and the vertex count is
    right -- so only the names can tell the difference.
    """
    _, write = NAMED[file_type]

    exported = reload(write(written), file_type)

    assert sorted(exported.geometry) == ["7", "head"], (
        f"{file_type.upper()} came back as {sorted(exported.geometry)}, so object identity was "
        f"lost somewhere in the round trip"
    )


@pytest.mark.parametrize("file_type", sorted(NAMED))
def test_an_object_comes_back_where_it_started(file_type, source, written):
    """Bounds, not vertices: `object_mesh` welds an object's pieces, so neither count nor order
    of vertices survives by design. Where the surface *is* does.

    The reference is the *baked* source -- what the scene graph puts each object at -- because
    that is what was imported. Comparing against `scene.geometry` instead measures the node
    transform rather than the round trip.
    """
    _, write = NAMED[file_type]
    tolerance = 3 * quantum(written)

    exported = reload(write(written), file_type)

    baked = baked_geometries(source)
    for name, geometry in exported.geometry.items():
        drift = np.max(np.abs(np.asarray(geometry.bounds) - np.asarray(baked[name].bounds)))
        assert drift <= tolerance, (
            f"{file_type.upper()} object {name!r} came back {drift} from where it started, which "
            f"is more than the {tolerance} three quantization steps allow"
        )


@pytest.mark.parametrize("file_type", sorted(NAMED))
def test_a_collection_survives_being_written_out_and_read_back(file_type, written):
    """The full loop: a collection out to a file, back into a second collection, same ids.

    This is the test that the integer-name rule earns its keep -- without it the second
    collection would hold the same geometry under freshly invented ids.
    """
    read, write = NAMED[file_type]
    store = fabriks.MemoryStore()

    read(write(written), store, "again", cell_size=CELL_SIZE, levels=LEVELS)
    again = fabriks.open_collection(store, "again")

    assert sorted(again.objects) == sorted(written.objects), (
        f"ids went out as {sorted(written.objects)} and came back as {sorted(again.objects)}"
    )
    assert {entry.name for entry in again.objects.values()} == {"7", "head"}


@pytest.mark.parametrize("file_type", sorted(MERGING))
def test_a_format_without_names_comes_back_as_one_object(file_type, written):
    """PLY, STL and OFF hold a surface and not who it belongs to.

    Asserted rather than worked around: an adapter that looked like the OBJ one and quietly
    merged would be worse than one that says a merge is what this format does.
    """
    read, write = MERGING[file_type]
    store = fabriks.MemoryStore()

    read(write(written), store, "merged", cell_size=CELL_SIZE, levels=LEVELS)
    merged = fabriks.open_collection(store, "merged")

    assert len(merged.objects) == 1, (
        f"{file_type.upper()} carries no object names, so it must come back as one object, not "
        f"{len(merged.objects)}"
    )


@pytest.mark.parametrize("file_type", sorted(MERGING))
def test_a_merged_format_still_holds_the_whole_model(file_type, source, written):
    """Merging is allowed to lose identity. It is not allowed to lose geometry."""
    _, write = MERGING[file_type]
    tolerance = 3 * quantum(written)

    exported = reload(write(written), file_type)

    drift = np.max(np.abs(np.asarray(exported.bounds) - np.asarray(source.bounds)))
    assert drift <= tolerance, f"{file_type.upper()} lost part of the model: bounds moved {drift}"


def test_a_gltf_goes_out_and_comes_back_through_its_own_mapping(written):
    """A `.gltf` is a document plus its buffers, and both halves have to make the trip.

    `write_gltf` hands back exactly the mapping `read_gltf` takes, which is the round trip that
    never touches a disk.
    """
    store = fabriks.MemoryStore()
    blob = write_gltf(written)

    assert any(name.endswith(".gltf") for name in blob), f"no document in {sorted(blob)}"
    assert any(name.endswith(".bin") for name in blob), f"no buffers in {sorted(blob)}"

    read_gltf(blob, store, "c", cell_size=CELL_SIZE, levels=LEVELS)
    again = fabriks.open_collection(store, "c")

    assert {entry.name for entry in again.objects.values()} == {"7", "head"}


def test_a_gltf_blob_whose_buffers_are_elsewhere_says_what_to_pass_instead(written):
    """trimesh fails this case with `'NoneType' object is not subscriptable`, from deep inside.

    A bare blob has no directory to resolve a sibling `.bin` against, so the document is checked
    here first and the refusal names the three things that do work.
    """
    blob = write_gltf(written)
    document = next(name for name in blob if name.endswith(".gltf"))

    with pytest.raises(fabriks.FormatError, match="external buffer"):
        read_gltf(blob[document], fabriks.MemoryStore(), "c")


def test_a_gltf_read_from_a_path_finds_its_own_buffers(written, tmp_path):
    """A path is the one source shape that lets trimesh's resolver reach the sibling files."""
    write_gltf(written, tmp_path / "model.gltf")
    store = fabriks.MemoryStore()

    read_gltf(tmp_path / "model.gltf", store, "c", cell_size=CELL_SIZE, levels=LEVELS)

    assert {entry.name for entry in fabriks.open_collection(store, "c").objects.values()} == {
        "7",
        "head",
    }


def test_an_export_written_to_a_path_lands_there(written, tmp_path):
    """`target` is the convenience that saves a caller a `write_bytes`, and it must actually write."""
    blob = write_glb(written, tmp_path / "model.glb")

    assert (tmp_path / "model.glb").read_bytes() == blob


def test_a_coarse_level_is_coarser(written):
    """Exporting level 2 rather than level 0 should cost triangles, which is the whole point."""
    fine = reload(write_glb(written, level=0), "glb")
    coarse = reload(write_glb(written, level=2), "glb")

    assert sum(len(m.faces) for m in coarse.geometry.values()) < sum(
        len(m.faces) for m in fine.geometry.values()
    ), "level 2 should hold fewer triangles than level 0"
