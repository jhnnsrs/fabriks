"""Choosing a grid from a byte budget, and whether the numbers it reports are true.

The interesting tests here are the last two. Everything before them checks that the planner is
internally consistent -- monotone, bounded, refusing what it should refuse -- which a wrong model
would pass just as easily. What makes the estimates a promise rather than decoration is building
the collection the plan describes and measuring it, which is what the closing tests do.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest
import trimesh

import fabriks
from fabriks.frames import blob_sizes


def plan_quietly(objects: dict[int, Any], **kwargs: Any) -> fabriks.GridPlan:  # noqa: ANN401
    """A plan without the advisory warnings, for tests not asserting on them."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fabriks.plan_grid(objects, **kwargs)


def build_quietly(objects: dict[int, Any], plan: fabriks.GridPlan) -> fabriks.MeshCollection:
    """The collection a plan describes. The QUARTER warning is asserted on elsewhere."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fabriks.build_collection(objects, **plan.as_kwargs())


@pytest.fixture(scope="module")
def surface() -> dict[int, Any]:
    """One large object cut into many cells, which the shared fixtures deliberately are not.

    ``conftest`` builds a scene of separate objects, where a cell holds whole objects and the
    cutting is incidental. A single surface spanning hundreds of voxels is the opposite regime --
    every cell holds a patch of one thing, and every cell face is a cut. The two disagree about
    how per-cell load moves up the levels, so a bracket calibrated on one alone would be too
    narrow, and the estimate tests below run against both on purpose.
    """
    return {1: trimesh.creation.icosphere(radius=400.0, subdivisions=4).apply_translation([420.0] * 3)}


@pytest.fixture(scope="module")
def scattered() -> dict[int, Any]:
    """Many small objects spread far apart -- the scene that exposed the writer's decimation bug.

    Spread along one axis on purpose: it makes the grid large relative to the objects, which is
    what lets a byte budget choose a cell far under the object fit and a tree deep enough for the
    failure to show up.
    """
    objects: dict[int, Any] = {}
    for index in range(32):
        centre = [120.0 + index * 90.0, 260.0 + 120.0 * np.sin(index * 0.7), 130.0 + 60.0 * np.cos(index * 0.5)]
        if index % 3 == 0:
            body = trimesh.creation.icosphere(radius=26.0, subdivisions=3)
        elif index % 3 == 1:
            body = trimesh.creation.box(extents=[46.0, 30.0, 22.0])
        else:
            body = trimesh.creation.capsule(radius=14.0, height=40.0, count=[24, 24])
        objects[1000 + index * 7] = body.apply_translation(centre)
    return objects


def test_a_plan_is_built_from_without_being_touched_up(objects):
    """The plan's whole claim is that its output goes straight into a build.

    Not trivially true: `cell_size` has to be three whole positive numbers and `levels` at least
    one, and both are derived here rather than supplied, so either could come back malformed.
    """
    plan = plan_quietly(objects)
    collection = build_quietly(objects, plan)

    assert collection.grid.cell_size == plan.cell_size, "the built grid is not the planned one"
    assert collection.grid.levels == plan.levels
    assert len(collection.shards) == plan.levels, "every level from 0 gets a shard"


def test_the_cell_is_a_power_of_two_on_every_component(objects):
    """Coarser cells have to be exactly two finer ones, or the LOCKED boundary argument fails.

    A level-0 plane is only a superset of every coarser level's plane set when the cell doubles
    exactly, so a cell size of, say, 96 would silently break the no-cracks guarantee.
    """
    for budget in (2 * 1024, 8 * 1024, 64 * 1024):
        plan = plan_quietly(objects, cell_bytes=budget)
        for component in plan.cell_size:
            assert component & (component - 1) == 0, f"{plan.cell_size} is not a power of two"


def test_a_bigger_cell_budget_never_asks_for_a_smaller_cell(objects):
    """Monotonicity is the property that makes the knob a knob.

    A planner that picked the *closest* rung rather than the largest one under the budget would
    fail this: it could jump down a rung as the budget rises past a midpoint.
    """
    sizes = [np.prod(plan_quietly(objects, cell_bytes=b).cell_size) for b in (1024, 4096, 16384, 65536)]
    assert sizes == sorted(sizes), f"cell volume is not monotone in cell_bytes: {sizes}"


def test_a_bigger_layer_budget_never_asks_for_more_levels(objects):
    """The tree stops once a level fits one retrieval, so a larger retrieval means a shallower tree.

    Worth checking rather than assuming: `levels` also floors at one cell and caps at
    `max_levels`, and either could break the ordering if the stopping rule were written the
    other way round.
    """
    depths = [plan_quietly(objects, layer_bytes=b).levels for b in (8 * 1024, 64 * 1024, 1024 * 1024)]
    assert depths == sorted(depths, reverse=True), f"levels is not monotone in layer_bytes: {depths}"


def test_a_layer_budget_that_already_holds_everything_gives_one_level(objects):
    """A single level is the right answer, not a degenerate one to be avoided.

    When level 0 already fits in one fetch there is nothing a coarser level could save, and
    emitting one anyway would cost a file per level for no reduction in what a renderer moves.
    """
    with pytest.warns(UserWarning, match="nothing a coarser level would save"):
        plan = fabriks.plan_grid(objects, layer_bytes=1 << 30)

    assert plan.levels == 1
    assert len(plan.layers) == 1


def test_a_cell_below_the_object_fit_is_warned_about_and_still_returned(objects):
    """Cutting objects is normal here, so this is advice rather than a refusal.

    The format exists to cut objects -- the suite's own fixtures use a cell smaller than their
    largest object. What the warning is for is the consequence: cut vertices are pinned, so the
    simplifier falls short of QUARTER and the coarse levels come out larger than estimated.
    Clamping instead would throw the byte budget away and hand back a one-cell octree.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        plan = fabriks.plan_grid(objects, cell_bytes=512)

    assert any("below the object fit" in str(entry.message) for entry in caught), (
        f"expected an object-fit warning, got {[str(entry.message) for entry in caught]}"
    )

    fit = fabriks.choose_cell_size({key: fabriks.coerce_mesh(value) for key, value in objects.items()})
    assert plan.object_fit == fit, "the plan reports a different fit than choose_cell_size"
    assert any(a < b for a, b in zip(plan.cell_size, fit)), "nothing was actually below the fit"
    assert plan.levels >= 1


def test_a_cell_index_never_passes_the_morton_limit(objects):
    """The 17-bit cap is resolved here rather than discovered partway through an expensive build.

    Level-0 indices are the largest in the tree, so a small cell on a distant object is what
    pushes a key past what the format can address. `morton_encode` refuses it, and refusing it
    after the clipping has run is the expensive way to find out.
    """
    far = dict(objects)
    far[99] = trimesh.creation.box(extents=[10.0, 10.0, 10.0]).apply_translation([4.0e6, 10.0, 10.0])

    plan = plan_quietly(far, cell_bytes=1024)
    reach = [int(np.ceil(4.01e6 / component)) for component in plan.cell_size]
    assert max(reach) < (1 << 16), f"cell {plan.cell_size} leaves indices up to {max(reach)}"
    fabriks.morton_encode_one((reach[0], 1, 1))  # raises if the cap is breached


def test_an_aspect_anchor_is_scaled_rather_than_replaced(objects):
    """`aspect` exists so a caller can hand over the source array's chunk shape.

    That shape is the one thing about the grid no amount of looking at the meshes can reveal, so
    the planner scales it rather than choosing three independent numbers -- the ratios have to
    survive.
    """
    plan = plan_quietly(objects, aspect=(1, 2, 4), cell_bytes=8 * 1024)
    x, y, z = plan.cell_size
    assert (y, z) == (2 * x, 4 * x), f"aspect (1, 2, 4) came back as {plan.cell_size}"


def test_an_empty_collection_is_refused_the_way_a_build_refuses_one(objects):
    """Same error type and the same moment as `build_collection`, so the two agree.

    A planner that returned some default grid for no objects would push the failure into the
    build, which is the expensive place to discover it.
    """
    del objects
    with pytest.raises(fabriks.FormatError, match="at least one object"):
        fabriks.plan_grid({})


def test_a_plan_deep_on_cells_under_the_object_fit_warns_that_the_build_may_raise(scattered):
    """The one regime where a plan's own output is known to break `build_collection`.

    Decimation places a collapsed vertex at the quadric-optimal position, and for a vertex near
    but not on a cell face that position can fall outside the cell -- measured at 1.15 voxels
    out, against a clip-and-snap step that is exact to 0.0. Quantization is per cell, so
    `encode_positions` raises. It is a writer bug rather than a planning one, but the plan is
    what walks into it, so the warning has to come from here.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fabriks.plan_grid(scattered, cell_bytes=6 * 1024, layer_bytes=64 * 1024)

    assert any("PartitioningError" in str(entry.message) for entry in caught), (
        f"expected the writer-bug warning, got {[str(entry.message) for entry in caught]}"
    )


@pytest.mark.parametrize(("cell_bytes", "layer_bytes"), [(16 * 1024, 128 * 1024), (32 * 1024, 512 * 1024)])
def test_a_plan_that_does_not_warn_builds_without_raising(scattered, cell_bytes, layer_bytes):
    """The warning above has to be the *only* way into the failure, or it is not worth having.

    This is the scene that first exposed the writer bug, so it is the one that has to demonstrate
    a clean build when the planner says nothing -- a quiet plan that still raises would make the
    warning meaningless.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        plan = fabriks.plan_grid(scattered, cell_bytes=cell_bytes, layer_bytes=layer_bytes)

    assert not any("PartitioningError" in str(entry.message) for entry in caught)
    build_quietly(scattered, plan)  # raises PartitioningError if the regime was misjudged


def test_geometry_in_the_negative_octant_is_refused_rather_than_planned_around(objects):
    """A plan against a grid the collection could never be written to is worse than no plan.

    The octree is anchored at the origin and cell indices are non-negative, so a vertex below it
    has no cell. `morton_encode` says so at build time; saying it here stops a plausible-looking
    table being computed for a grid that cannot exist.
    """
    shifted = dict(objects)
    shifted[42] = trimesh.creation.box(extents=[10.0, 10.0, 10.0]).apply_translation([-50.0, 10.0, 10.0])

    with pytest.raises(fabriks.FormatError, match="positive octant"):
        fabriks.plan_grid(shifted)


@pytest.mark.parametrize("band", [(0.0, 1.0), (2.0, 1.0)])
def test_a_band_that_is_not_an_ascending_pair_of_factors_is_refused(objects, band):
    """Both bands are ordered pairs, and a reversed one would silently inverve every bracket."""
    with pytest.raises(fabriks.FormatError, match="ascending pair"):
        fabriks.plan_grid(objects, coarsening=band)


def test_describe_prints_a_row_per_level_and_marks_the_estimates(surface):
    """The table is the deliverable for a human, and the est. marks are the honest part of it.

    Level 0 is arithmetic and everything above it is an extrapolation; a table that did not say
    which was which would invite the coarse numbers to be read as promises.
    """
    plan = plan_quietly(surface, cell_bytes=8 * 1024, layer_bytes=32 * 1024)
    lines = plan.describe().splitlines()

    assert plan.levels > 1, "this test needs a tree with coarse levels in it"
    assert sum(1 for line in lines if "est." in line) == plan.levels - 1
    for layer in plan.layers:
        assert any(line.startswith(f"{layer.level:>2}  ") for line in lines)


@pytest.mark.parametrize("cell_bytes", [2 * 1024, 8 * 1024, 32 * 1024])
def test_level_zero_lands_within_the_tolerance_it_claims(objects, cell_bytes):
    """The contract that makes level 0 worth reporting as arithmetic rather than a guess.

    Nothing about binning coordinates is guaranteed to match what trimesh's clipper and the
    welding actually produce -- this is the test that says the model tracks them.
    """
    plan = plan_quietly(objects, cell_bytes=cell_bytes)
    collection = build_quietly(objects, plan)

    sizes = np.asarray(blob_sizes(collection.shards[0][1]))
    layer = plan.layers[0]

    assert layer.exact, "level 0 should be reported as arithmetic"
    assert layer.bytes_low <= sizes.sum() <= layer.bytes_high, (
        f"level 0 measured {sizes.sum():,} bytes, predicted {layer.bytes_low:,}..{layer.bytes_high:,}"
    )
    assert layer.cell_bytes_low <= np.percentile(sizes, 95) <= layer.cell_bytes_high, (
        f"level 0 p95 measured {np.percentile(sizes, 95):,.0f}, predicted "
        f"{layer.cell_bytes_low:,}..{layer.cell_bytes_high:,}"
    )
    assert 0.8 <= layer.cells / len(sizes) <= 1.2, (
        f"level 0 predicted {layer.cells} cells against {len(sizes)} built"
    )


def test_every_level_lands_inside_its_bracket_on_a_single_cut_surface(surface):
    """The bracket has to hold in the regime it was widened for, not just the easy one.

    A scene of separate objects and one large cut surface move differently up the levels -- on
    one, per-cell bytes fall at the top of the tree as the coarsest cells stop being saturated;
    on the other they climb the whole way, because the simplifier keeps missing QUARTER. The
    band spans both, so it is only tested by running both.
    """
    plan = plan_quietly(surface, cell_bytes=8 * 1024, layer_bytes=64 * 1024)
    collection = build_quietly(surface, plan)

    for level, shard in collection.shards:
        sizes = np.asarray(blob_sizes(shard))
        layer = plan.layers[level]
        assert layer.bytes_low <= sizes.sum() <= layer.bytes_high, (
            f"level {level} measured {sizes.sum():,} bytes, predicted "
            f"{layer.bytes_low:,}..{layer.bytes_high:,}"
        )
        assert layer.cell_bytes_low <= np.percentile(sizes, 95) <= layer.cell_bytes_high, (
            f"level {level} p95 measured {np.percentile(sizes, 95):,.0f}, predicted "
            f"{layer.cell_bytes_low:,}..{layer.cell_bytes_high:,}"
        )
