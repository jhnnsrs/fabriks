"""Writing: the layout, the part splitting, and the ordering that makes a prefix trustworthy."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
import trimesh

import fabriks
from fabriks.stores import DirectoryStore
from tests.conftest import CELL_SIZE


class RecordingStore(fabriks.MemoryStore):
    """A store that remembers the order it was written in."""

    def __init__(self) -> None:
        """Start empty, with an empty log."""
        super().__init__()
        self.order: list[str] = []

    def put(self, path: str, data: object) -> None:
        """Record the write, then perform it."""
        super().put(path, data)
        self.order.append(path)


def test_the_manifest_is_written_last(collection: fabriks.MeshCollection):
    """The completion protocol in one assertion.

    A prefix has no atomic "upload finished" flag, so the manifest landing last is the only
    thing separating "this collection is half written" from "this collection is corrupt". A
    writer that lands it first builds a prefix that registers cleanly and fails later, on a
    reader.
    """
    store = RecordingStore()

    fabriks.write_collection(collection, store, "c")

    assert store.order[-1] == "c/fabriks.json"
    assert len(store.order) == len(set(store.order)), "nothing was written twice"


def test_everything_the_manifest_names_exists_before_the_manifest_does(collection: fabriks.MeshCollection):
    """The manifest is a promise about the tree, so the tree has to be there first."""
    store = RecordingStore()

    fabriks.write_collection(collection, store, "c")

    files = json.loads(store.objects["c/fabriks.json"])["files"]
    named = [files["cells"]["path"], files["objects"]["path"]]
    named += [entry["path"] for entries in files["levels"].values() for entry in entries]

    written_before_manifest = set(store.order[:-1])
    for path in named:
        assert f"c/{path}" in written_before_manifest, f"the manifest names {path}, which was not written before it"


def test_an_interrupted_write_leaves_a_prefix_that_is_refused(collection: fabriks.MeshCollection):
    """Precisely the shape a killed writer leaves, and it must not read as a collection."""
    store = fabriks.MemoryStore()
    fabriks.write_collection(collection, store, "c")
    del store.objects["c/fabriks.json"]  # the last write never happened

    with pytest.raises(fabriks.UnfinishedCollectionError, match="manifest last"):
        fabriks.open_collection(store, "c")


def test_the_written_manifest_names_the_parts_that_actually_landed(collection: fabriks.MeshCollection):
    """A reader that cannot list a prefix -- an HTTP store -- still has to find every level."""
    store = fabriks.MemoryStore()

    manifest = fabriks.write_collection(collection, store, "c")

    for level in range(manifest.grid.levels):
        entries = manifest.level_files(level)
        assert entries, f"level {level} claims no parts"
        for entry in entries:
            assert f"c/{entry.path}" in store.objects
            assert entry.size == len(store.objects[f"c/{entry.path}"]), (
                "the recorded length is what lets a reader seek to a Parquet footer without "
                "being able to stat the object, so a wrong one is worse than an absent one"
            )
            assert entry.row_groups and entry.row_groups >= 1


def test_a_large_level_is_split_across_parts_and_still_reads_as_one(collection: fabriks.MeshCollection):
    """A level is a directory precisely so this can happen without the layout changing shape."""
    store = fabriks.MemoryStore()

    # A budget far below one cell's blobs, so every cell becomes its own part.
    manifest = fabriks.write_collection(collection, store, "c", max_part_bytes=1)

    paths = [entry.path for entry in manifest.level_files(0) or []]
    assert len(paths) > 1, "the level should have been split"
    assert paths == sorted(paths), "parts are numbered in order"

    opened = fabriks.open_collection(store, "c")
    assert opened.geometry(0).num_rows == collection.shards[0][1].num_rows, "the parts read back as one level"


def test_the_levels_written_are_contiguous_from_zero(collection: fabriks.MeshCollection):
    """A gap is geometry a planner never asks for.

    A planner descends from the coarsest level it can see, so anything under a missing level is
    absent from the render with nothing raised anywhere. A genuinely empty level is written out
    rather than skipped.
    """
    store = fabriks.MemoryStore()

    manifest = fabriks.write_collection(collection, store, "c")

    levels = sorted(int(level) for level in manifest.files["levels"])
    assert levels == list(range(manifest.grid.levels))


def test_a_cell_never_spans_more_than_its_declared_cell_size_allows(collection: fabriks.MeshCollection):
    """A cell's geometry lives inside its address box, which is what makes ``cell_size`` checkable.

    Positions are quantized to exactly that box, so an observed extent larger than
    ``cell_size * 2**level`` proves the declared size is not the one the octree was cut with.
    One-sided: it catches too-small, the direction that silently misplaces geometry, and cannot
    catch too-large, since a sparse cell is legitimately smaller than its box.
    """
    store = fabriks.MemoryStore()
    fabriks.write_collection(collection, store, "c")
    opened = fabriks.open_collection(store, "c")

    for entry in opened.cells.values():
        allowed = opened.grid.cell_extent(entry.level)
        observed = [high - low for low, high in zip(entry.bbox_min, entry.bbox_max)]
        for axis, (span, limit) in enumerate(zip(observed, allowed)):
            assert span <= limit + 1e-6, (
                f"cell {entry.cell} at level {entry.level} spans {span} on axis {axis}, past the {limit} its "
                f"declared cell_size allows"
            )


def test_the_declared_index_width_is_the_one_the_blobs_use():
    """fabriks declares ``UINT32`` because it writes uint32, not because it is the safe default.

    Checked against the bytes rather than against the declaration: under ``codec: NONE`` the
    blob is a flat little-endian uint32 triangle list, so its length pins the width exactly.
    A collection whose declaration and blobs disagreed would decode to garbage with nothing
    raised anywhere.
    """
    objects = {1: trimesh.creation.icosphere(radius=20.0).apply_translation([64.0, 64.0, 32.0])}

    raw = fabriks.build_collection(
        objects, cell_size=(128, 128, 64), levels=1, codec=fabriks.CODEC_NONE
    )

    assert raw.encoding.indices == fabriks.INDICES_UINT32
    shard = raw.shards[0][1]
    for row in range(shard.num_rows):
        blob = shard.column("indices")[row].as_py()
        assert len(blob) == 4 * shard.column("index_count")[row].as_py(), "four bytes per index is UINT32"


def test_a_frame_missing_a_required_column_is_refused_before_any_write():
    """The earliest point the mistake is catchable and the only point it is cheap."""
    pa = pytest.importorskip("pyarrow")
    broken = pa.table({"level": [0]})

    with pytest.raises(fabriks.FormatError, match="missing"):
        fabriks.validate_columns(broken, "cell_catalog")


def test_write_meshes_builds_and_writes_in_one_call(objects: dict, tmp_path: Path):
    """The convenience form, and the one most callers will use."""
    store = DirectoryStore(tmp_path)

    manifest = fabriks.write_meshes(objects, store, "c", cell_size=CELL_SIZE, levels=2)

    assert manifest.spec_version == fabriks.SPEC_VERSION
    assert manifest.grid.levels == 2
    assert (tmp_path / "c" / "fabriks.json").is_file()


def test_the_async_writer_produces_the_same_tree(collection: fabriks.MeshCollection):
    """Mirrors the sync path rather than reimplementing it, so the ordering cannot drift."""
    synchronous = fabriks.MemoryStore()
    fabriks.write_collection(collection, synchronous, "c")

    asynchronous = fabriks.MemoryStore()
    asyncio.run(fabriks.awrite_collection(collection, asynchronous, "c"))

    assert synchronous.objects.keys() == asynchronous.objects.keys()
    assert synchronous.objects["c/fabriks.json"] == asynchronous.objects["c/fabriks.json"]


def test_writing_twice_replaces_rather_than_accumulates(collection: fabriks.MeshCollection):
    """A collection is immutable per version, but a retried write must not leave debris."""
    store = fabriks.MemoryStore()

    fabriks.write_collection(collection, store, "c")
    first = dict(store.objects)
    fabriks.write_collection(collection, store, "c")

    assert store.objects.keys() == first.keys()
