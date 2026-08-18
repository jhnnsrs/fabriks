"""The layer between a mesh file and a collection: the fit, the identity, and the catalog.

Three of these are regression tests for failures that produce *plausible* output rather than an
error -- a model stacked on the origin because the scene graph was not baked, an octree of one
cell because the source was sized before it was fitted, and a merged mesh because an OBJ was
loaded without asking it to split. None of them raises anything on its own.

The round trips through actual formats are in `test_contrib_formats.py`.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import trimesh

import fabriks
from fabriks.contrib import DEFAULT_RESOLUTION, VoxelFit, fit_of, names_of, read_glb
from fabriks.contrib.obj import _split_kwarg
from fabriks.contrib.scene import assign_ids, baked_geometries, collection_to_scene


def two_objects(scale=1.0, centre=True):
    """A two-object scene, named one way that is an id and one way that is not."""
    scene = trimesh.Scene()
    sphere = trimesh.creation.icosphere(radius=0.6 * scale, subdivisions=2)
    box = trimesh.creation.box(extents=[1.0 * scale, 0.4 * scale, 0.3 * scale])
    scene.add_geometry(sphere, geom_name="head")
    scene.add_geometry(
        box,
        geom_name="7",
        transform=trimesh.transformations.translation_matrix([0.8 * scale, 0.0, 0.0]),
    )
    if not centre:
        scene.apply_translation([10.0 * scale, 10.0 * scale, 10.0 * scale])
    return scene


def dense_objects():
    """The same two objects with enough triangles to fill a real octree."""
    scene = trimesh.Scene()
    scene.add_geometry(trimesh.creation.icosphere(radius=0.6, subdivisions=3), geom_name="head")
    scene.add_geometry(
        trimesh.creation.box(extents=[1.0, 0.4, 0.3]),
        geom_name="7",
        transform=trimesh.transformations.translation_matrix([0.8, 0.0, 0.0]),
    )
    return scene


def imported(scene, **options: object):
    """Read a scene through GLB into a memory store, and open what came out."""
    store = fabriks.MemoryStore()
    result = read_glb(scene.export(file_type="glb"), store, "c", **options)
    return result, fabriks.open_collection(store, "c")


# -- the fit ------------------------------------------------------------------------ #


def test_a_model_centred_on_the_origin_is_fitted_rather_than_refused():
    """Most mesh files are centred on the origin, and the builder refuses negative cells.

    Without the fit this is not a wrong answer but a hard failure -- `build_collection` raises
    "Cell indices are non-negative" -- so an adapter that passed coordinates straight through
    could not read a typical `.glb` at all.
    """
    _, collection = imported(two_objects())

    minimums = [entry.bbox_min for entry in collection.cells.values()]

    assert min(min(corner) for corner in minimums) >= 0.0, (
        "the fit is what puts a centred model in the positive octant, and something is below it"
    )


def test_a_unit_scale_model_is_given_enough_voxels_to_partition():
    """`cell_size` is whole voxels, so a model one unit across has no cell smaller than itself.

    The fit is what buys the resolution to partition at all: fitted, a 0.002-unit model spans
    `resolution` voxels and a 128-voxel cell cuts it into many; unfitted, the same model is
    smaller than the smallest cell that can exist and the octree has nowhere to go.
    """
    _, collection = imported(two_objects(scale=0.002), cell_size=(128, 128, 128), levels=3)

    entries = collection.objects.values()
    low = [min(entry.bbox_min[axis] for entry in entries) for axis in range(3)]
    high = [max(entry.bbox_max[axis] for entry in entries) for axis in range(3)]
    span = max(high[axis] - low[axis] for axis in range(3))

    assert span == pytest.approx(DEFAULT_RESOLUTION, rel=0.01), (
        f"the fitted model spans {span} voxels along its longest axis, not the "
        f"{DEFAULT_RESOLUTION} it was fitted to"
    )
    assert len(collection.cells_at(0)) > 1, (
        f"level 0 holds {len(collection.cells_at(0))} cell(s); a model given real resolution "
        f"should partition into more than one"
    )


def test_a_unit_scale_model_read_without_the_fit_is_refused():
    """The counterpart: this is what `read_*` would do for every centred file without the fit.

    Not a subtly wrong octree but a hard failure, which is why fitting is the default rather
    than something a caller opts into.
    """
    with pytest.raises((fabriks.FormatError, ValueError)):
        imported(two_objects(scale=0.002), fit=VoxelFit.identity(), levels=3)


def test_the_fit_undoes_itself():
    """`apply` and `invert` are one transform read in two directions, and must agree."""
    fit = VoxelFit.for_bounds([-3.0, -1.0, 0.5], [1.0, 2.0, 4.5])
    points = np.array([[-3.0, -1.0, 0.5], [0.0, 0.0, 1.0], [1.0, 2.0, 4.5]])

    assert np.allclose(fit.invert(fit.apply(points)), points), "the fit does not round trip"
    assert np.all(fit.apply(points) >= -1e-9), "a fitted model should land in the positive octant"


def test_the_longest_side_sets_the_scale_for_every_component():
    """A per-component fit would stretch a model to fill its box and distort every render of it."""
    fit = VoxelFit.for_bounds([0.0, 0.0, 0.0], [4.0, 2.0, 1.0], resolution=DEFAULT_RESOLUTION)

    fitted = fit.apply(np.array([[4.0, 2.0, 1.0]]))[0]

    assert fitted[0] == pytest.approx(DEFAULT_RESOLUTION), "the longest side sets the scale"
    assert fitted[1] == pytest.approx(DEFAULT_RESOLUTION / 2), "proportions must be preserved"


def test_a_model_with_no_extent_does_not_divide_by_zero():
    """A flat or empty bound has no meaningful scale, and identity beats a ZeroDivisionError."""
    assert VoxelFit.for_bounds([1.0, 1.0, 1.0], [1.0, 1.0, 1.0]).is_identity


def test_an_identity_fit_leaves_a_model_where_it_was():
    """A caller whose model is already in voxel coordinates must be able to say so."""
    scene = two_objects(scale=200.0, centre=False)

    _, collection = imported(scene, fit=VoxelFit.identity(), cell_size=(64, 64, 64), levels=2)

    high = max(max(entry.bbox_max) for entry in collection.cells.values())
    assert high == pytest.approx(scene.bounds[1].max(), rel=0.01), (
        "an identity fit should leave the source's own coordinates alone"
    )


def test_a_model_sized_into_a_single_level_says_so():
    """A one-level collection is valid and carries no level of detail, which is the whole point.

    `plan_grid` budgets by bytes, so a small model legitimately comes back with no pyramid --
    and a caller who reached for a level-of-detail format deserves to hear that rather than
    discover it when a viewer has nothing to choose between.
    """
    with pytest.warns(UserWarning, match="single level"):
        imported(two_objects())


@pytest.mark.parametrize(
    ("options", "expected_levels"),
    [({}, 1), ({"cell_size": (128, 128, 128)}, 1), ({"levels": 3}, 3)],
)
def test_the_grid_is_filled_in_only_where_the_caller_left_it_open(options, expected_levels):
    """`cell_size` and `levels` are independent: giving one must not discard the other.

    Both come from `plan_grid` when neither is given, and either one alone has to survive while
    the other is planned around it.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, collection = imported(two_objects(), **options)

    assert collection.grid.levels == expected_levels
    assert all(component >= 1 for component in collection.grid.cell_size)


# -- the scene graph ---------------------------------------------------------------- #


def test_node_transforms_are_baked_rather_than_dropped():
    """`scene.geometry` holds the mesh a node points at, not where the node puts it.

    Reading it directly gives every object the right vertex count at the wrong place, stacked on
    the origin -- which looks like a working import right up until someone renders it.
    """
    scene = trimesh.Scene()
    scene.add_geometry(
        trimesh.creation.box(extents=[1.0, 1.0, 1.0]),
        geom_name="moved",
        transform=trimesh.transformations.translation_matrix([10.0, 0.0, 0.0]),
    )

    baked = baked_geometries(scene)["moved"]

    assert baked.bounds[0][0] == pytest.approx(9.5), (
        f"the node's transform was not applied: the box sits at {baked.bounds[0][0]}, not 9.5"
    )


def test_an_object_that_is_only_a_mesh_still_reads():
    """A format with no scene -- STL, PLY -- loads as one mesh, and must not need a second path."""
    store = fabriks.MemoryStore()
    mesh = trimesh.creation.icosphere(radius=1.0, subdivisions=2)

    result = read_glb(trimesh.Scene(mesh).export(file_type="glb"), store, "c", levels=2)

    assert len(result.names) == 1, "one mesh is one object"


# -- identity ----------------------------------------------------------------------- #


def test_an_integer_looking_name_keeps_its_number():
    """This is the whole reason a collection survives a trip out to a file and back.

    Written out, an object's id becomes its name; read back, that name has to become the id
    again or the round trip preserves only the count.
    """
    assert assign_ids(["7", "3"]) == {"7": 7, "3": 3}


def test_a_number_is_reserved_even_when_a_word_is_seen_first():
    """The integer names are collected before anything is assigned, and that order matters.

    Assign as you go and `"head"` takes 0 before `"0"` is ever reached, so an object comes back
    under an id that belonged to a different one -- the one silent way identity can be corrupted
    rather than merely lost.
    """
    assert assign_ids(["head", "torso", "0"]) == {"head": 1, "torso": 2, "0": 0}


def test_a_name_that_is_not_a_number_gets_the_lowest_free_id():
    """Ids have to come from somewhere, and they must not collide with the ones being kept."""
    assigned = assign_ids(["head", "7", "torso"])

    assert assigned["7"] == 7, "an integer name keeps its value"
    assert sorted([assigned["head"], assigned["torso"]]) == [0, 1], (
        f"names should take the lowest free ids, got {assigned}"
    )


def test_the_names_ride_on_the_object_catalog():
    """The names have to survive the write, or an export can only number its objects."""
    result, collection = imported(two_objects())

    assert "name" in collection.object_catalog.column_names, "the name column was not written"
    assert names_of(collection) == result.names, "the names did not survive the round trip"
    assert collection.objects[7].name == "7", "ObjectEntry should carry the name too"


def test_a_collection_written_without_the_column_still_reads(opened):
    """The column is optional, and every collection written before it exists lacks it.

    Reading one has to give None rather than raise -- this is the back-compatibility gate on the
    whole approach of carrying contrib's facts as extra catalog columns.
    """
    assert names_of(opened) == {}, "a collection with no names should report none, not guess"
    assert fit_of(opened) is None, "no recorded fit is a different answer from an identity fit"
    assert opened.objects[7].name is None, "an absent name column must read back as None"


def test_the_fit_travels_with_the_collection():
    """An export inverts the import's own numbers, so they have to be in the collection."""
    result, collection = imported(two_objects())

    assert fit_of(collection) == result.fit, "the recorded fit is not the one that was applied"


# -- export selection --------------------------------------------------------------- #


def test_an_export_can_be_narrowed_to_the_objects_that_matter():
    """A caller extracting one object should not pay for reassembling the rest."""
    _, collection = imported(two_objects())

    scene = collection_to_scene(collection, objects=[7])

    assert sorted(scene.geometry) == ["7"], f"asked for one object, got {sorted(scene.geometry)}"


def test_an_object_missing_from_a_coarse_level_is_reported_rather_than_dropped(opened):
    """A short export that says nothing is the failure worth avoiding here.

    `Collection.object_mesh` raises for an object a coarse level decimated away entirely, and the
    builder itself warns rather than fails in the same situation -- so this follows it.
    """
    with pytest.warns(UserWarning, match="no geometry at level"):
        scene = collection_to_scene(opened, level=0, objects=[7, 999])

    assert sorted(scene.geometry) == ["7"], "the objects that do exist should still be exported"


def test_an_export_with_nothing_in_it_is_refused(opened):
    """Handing back an empty file is worse than saying there was nothing to put in one."""
    with (
        pytest.raises(fabriks.FormatError, match="Nothing was exported"),
        pytest.warns(UserWarning),
    ):
        collection_to_scene(opened, objects=[404])


# -- the trimesh version probe ------------------------------------------------------- #


def test_the_obj_loader_takes_the_kwarg_this_package_splits_objects_with():
    """The kwarg is `split_objects` in trimesh 5 and `split_object` in 4, and fabriks allows both.

    A probe nothing exercises is a branch that rots: if trimesh renames it again, loading an OBJ
    silently returns one merged object, which is indistinguishable from a file that held one.
    """
    import inspect

    assert _split_kwarg() in inspect.signature(trimesh.exchange.obj.load_obj).parameters


# -- the collection is still a collection --------------------------------------------- #


def test_a_collection_written_by_contrib_verifies():
    """The extra columns must not make the collection anything but an ordinary one.

    Read at a grid that gives the octree something to do -- 30 cells across three levels -- so
    this checks the catalog contrib wrote rather than how a two-triangle level decimates.
    """
    _, collection = imported(dense_objects(), cell_size=(128, 128, 128), levels=3)

    report = fabriks.verify(collection, tier="geometry")

    assert report, f"a contrib-written collection failed verification:\n{report}"
