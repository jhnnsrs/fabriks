"""The extra object-catalog columns an import writes, so an export can undo it.

Two facts have to survive a write for a round trip to close, and neither has a home in the
format's required columns: what each object was **called** in the file it came from, and what
:class:`~fabriks.contrib.fit.VoxelFit` was applied to get it into the voxel octant.

Both go into the object catalog as extra columns, which is what :mod:`fabriks.frames` says extra
columns are for -- *"a writer may carry a denormalized attribute copy alongside"* -- and why
``validate_columns`` asks whether the required columns are present rather than whether any others
are. Nothing in the format changes: ``REQUIRED_COLUMNS`` and the Arrow schemas are untouched, a
collection written without these columns reads exactly as it did before, and ``fabriks.verify``
passes either way.

The fit is one fact about the whole collection written identically on every row. That is a
denormalization rather than a mistake, and it is the same shape the cell catalog already has with
``part`` / ``row_group`` / ``blob_bytes``: nullable, filled by a writer, read back defensively.
Giving the fit a column keeps it inside the collection instead of in a sidecar file the format
does not describe, or in a manifest field the spec does not define.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from fabriks.contrib.fit import VoxelFit
from fabriks.errors import FormatError

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pyarrow as pa

#: The object's name in the file it was read from.
NAME_COLUMN = "name"
#: The uniform scale an import applied, repeated on every row.
FIT_SCALE_COLUMN = "fit_scale"
#: The translation an import applied, one column per component, repeated on every row.
FIT_TRANSLATION_COLUMNS = ("fit_translation_x", "fit_translation_y", "fit_translation_z")


def _table(collection: Any) -> pa.Table:  # noqa: ANN401 - a built or an opened collection
    """The object catalog off either a built ``MeshCollection`` or an opened ``Collection``."""
    return collection.object_catalog


def _object_ids(table: pa.Table) -> list[int]:
    """The catalog's object ids, in the row order the table has them in.

    Row order, not sorted order: every column added here is positional, so a sort would line the
    names up against the wrong objects. ``object_id`` is non-null in the format's schema, and a
    null one is refused rather than skipped -- skipping would shift every row after it.
    """
    ids: list[int] = []
    for row, value in enumerate(table.column("object_id").to_pylist()):
        if value is None:
            raise FormatError(
                f"Row {row} of this object catalog holds null in `object_id`, which the format "
                f"declares non-null. It was not written by fabriks."
            )
        ids.append(int(value))
    return ids


def attach(collection: Any, *, names: Mapping[int, str], fit: VoxelFit) -> None:  # noqa: ANN401
    """Add contrib's columns to a built collection's object catalog, in place.

    Call it between :func:`fabriks.build_collection` and :func:`fabriks.write_collection`:
    ``MeshCollection`` is a mutable dataclass and ``write_collection`` only rewrites the
    manifest's ``files``, so a column added here is what lands in the Parquet.

    An object with no name gets null rather than a stand-in, because a stand-in would read back
    as a name the source file never had.
    """
    import pyarrow as pa

    table = _table(collection)
    ids = _object_ids(table)

    columns: list[tuple[pa.Field[Any], pa.Array[Any]]] = [
        (
            pa.field(NAME_COLUMN, pa.string(), nullable=True),
            pa.array([names.get(object_id) for object_id in ids], type=pa.string()),
        ),
        (
            pa.field(FIT_SCALE_COLUMN, pa.float64(), nullable=True),
            pa.array([fit.scale] * len(ids), type=pa.float64()),
        ),
    ]
    for axis, component in zip(FIT_TRANSLATION_COLUMNS, fit.translation):
        columns.append(
            (
                pa.field(axis, pa.float64(), nullable=True),
                pa.array([float(component)] * len(ids), type=pa.float64()),
            )
        )

    for field, values in columns:
        if field.name in table.column_names:
            table = table.drop_columns([field.name])
        table = table.append_column(field, values)
    collection.object_catalog = table


def names_of(collection: Any) -> dict[int, str]:  # noqa: ANN401 - an opened collection
    """The source names the collection carries, empty when it carries none.

    Empty is the honest answer for a collection written by anything but contrib: the column is
    optional, and its absence means the names were never recorded rather than that they were
    empty strings.
    """
    table = _table(collection)
    if NAME_COLUMN not in table.column_names:
        return {}
    names = table.column(NAME_COLUMN).to_pylist()
    return {
        object_id: str(name)
        for object_id, name in zip(_object_ids(table), names)
        if name is not None
    }


def fit_of(collection: Any) -> VoxelFit | None:  # noqa: ANN401 - an opened collection
    """The fit recorded at import, or None for a collection contrib did not write.

    None and :meth:`~fabriks.contrib.fit.VoxelFit.identity` are deliberately different answers.
    None means nobody recorded a fit, so an export cannot claim to be returning source
    coordinates; the identity means someone recorded that no fit was needed.
    """
    table = _table(collection)
    wanted = (FIT_SCALE_COLUMN, *FIT_TRANSLATION_COLUMNS)
    if table.num_rows == 0 or any(column not in table.column_names for column in wanted):
        return None

    numbers: list[float] = []
    for column in wanted:
        value = table.column(column)[0].as_py()
        if value is None:
            return None
        numbers.append(float(value))
    scale, *shift = numbers
    return VoxelFit(scale=scale, translation=(shift[0], shift[1], shift[2]))


__all__ = [
    "FIT_SCALE_COLUMN",
    "FIT_TRANSLATION_COLUMNS",
    "NAME_COLUMN",
    "attach",
    "fit_of",
    "names_of",
]
