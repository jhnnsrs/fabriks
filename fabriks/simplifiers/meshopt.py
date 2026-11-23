"""``simplification: MESHOPT`` -- meshoptimizer's edge collapse with a per-vertex lock. The default.

The one simplifier that ships with the constraint the format actually states: ``vertex_lock``
pins exactly the vertices in ``fixed`` and nothing else. :class:`~fabriks.simplifiers.quadric.QuadricSimplifier`
can only say ``preserve_border``, which pins the whole topological boundary -- inherited level-0
seams included -- and so stalls short of the budget on a heavily cut object.

meshoptimizer is the ``meshopt`` extra of fabriks, and it is reached through the binding's private loader,
so it is resolved on the first simplify rather than on ``import fabriks``: a reader that never
builds a collection never touches it.
"""

from __future__ import annotations

import ctypes
import functools
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from fabriks.simplifiers.greedy import GreedyEdgeCollapse
from fabriks.simplifiers.measure import measure_deviation, pinned_held
from fabriks.simplifiers.protocol import Simplified, Simplifier

#: The ``simplification`` value that selects this backend, and the format's default.
SIMPLIFICATION_MESHOPT = "MESHOPT"

#: ``meshopt_SimplifyErrorAbsolute``: errors in position units rather than relative to the extent.
_ERROR_ABSOLUTE = 1 << 2
#: A target error no surface reaches, so the face budget alone decides when to stop.
_UNBOUNDED_ERROR = 3.0e38


@functools.cache
def _simplify_with_attributes() -> Any:  # noqa: ANN401
    """``meshopt_simplifyWithAttributes``, with the prototype the binding forgets to declare.

    The Python wrapper calls this symbol without ``argtypes``, so ctypes refuses the float
    ``target_error`` and would pass every ``size_t`` as a 32-bit int. ``lib[...]`` hands back a
    fresh function pointer rather than the cached attribute, so declaring it here changes nothing
    for anyone else using the library. The prototype is meshoptimizer 0.23's, which is what the
    ``>=0.2.30a0`` binding bundles.
    """
    from meshoptimizer._loader import lib  # type: ignore

    function = lib["meshopt_simplifyWithAttributes"]
    size = ctypes.c_size_t
    floats = ctypes.POINTER(ctypes.c_float)
    function.argtypes = [
        ctypes.POINTER(ctypes.c_uint),  # destination
        ctypes.POINTER(ctypes.c_uint),  # indices
        size,  # index_count
        floats,  # vertex_positions
        size,  # vertex_count
        size,  # vertex_positions_stride
        floats,  # vertex_attributes
        size,  # vertex_attributes_stride
        floats,  # attribute_weights
        size,  # attribute_count
        ctypes.POINTER(ctypes.c_ubyte),  # vertex_lock
        size,  # target_index_count
        ctypes.c_float,  # target_error
        ctypes.c_uint,  # options
        floats,  # result_error
    ]
    function.restype = size
    return function


@dataclass(frozen=True)
class MeshoptSimplifier:
    """Edge collapse via meshoptimizer, holding exactly the ``fixed`` mask still.

    meshoptimizer only ever *chooses which vertices survive* -- it never repositions one -- so a
    locked vertex comes back at bit-identical float64 coordinates: the library sees float32
    positions to decide collapses, but the result is indices into the caller's own array.

    ``verify_border`` re-checks the mask afterwards and hands the piece to ``fallback`` if a
    pinned position went missing. It should never fire; it is what turns the library's promise
    into the format's.
    """

    verify_border: bool = True
    fallback: Simplifier = field(default_factory=GreedyEdgeCollapse)
    name: str = SIMPLIFICATION_MESHOPT
    #: The lock is the mask, so the mask is what limits this backend.
    uses_fixed_mask: bool = True

    def simplify(
        self,
        vertices: npt.NDArray[np.float64],
        faces: npt.NDArray[np.int64],
        *,
        fixed: npt.NDArray[np.bool_],
        target_faces: int,
    ) -> Simplified:
        """Collapse to the face budget without collapsing a locked vertex."""
        source_vertices = np.asarray(vertices, dtype=np.float64)
        source_faces = np.asarray(faces, dtype=np.int64)
        if len(source_faces) <= target_faces:
            return Simplified(source_vertices, source_faces, 0.0, True, self.name)

        indices = np.ascontiguousarray(source_faces.reshape(-1), dtype=np.uint32)
        positions = np.ascontiguousarray(source_vertices, dtype=np.float32)
        lock = np.ascontiguousarray(np.asarray(fixed, dtype=bool), dtype=np.uint8)
        # One attribute at weight zero: the entry point wants an attribute stream, and a zero
        # weight makes the metric the plain positional one.
        attributes = np.zeros(len(positions), dtype=np.float32)
        weights = np.zeros(1, dtype=np.float32)
        destination = np.empty(len(indices), dtype=np.uint32)
        result_error = ctypes.c_float(0.0)

        floats = ctypes.POINTER(ctypes.c_float)
        kept_count = int(
            _simplify_with_attributes()(
                destination.ctypes.data_as(ctypes.POINTER(ctypes.c_uint)),
                indices.ctypes.data_as(ctypes.POINTER(ctypes.c_uint)),
                len(indices),
                positions.ctypes.data_as(floats),
                len(positions),
                positions.strides[0],
                attributes.ctypes.data_as(floats),
                attributes.strides[0],
                weights.ctypes.data_as(floats),
                1,
                lock.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
                3 * max(1, int(target_faces)),
                _UNBOUNDED_ERROR,
                _ERROR_ABSOLUTE,
                ctypes.byref(result_error),
            )
        )
        mapped = destination[:kept_count].astype(np.int64).reshape(-1, 3)
        if not len(mapped):
            return Simplified(source_vertices[:0], mapped, 0.0, False, self.name)

        # Compact against the float64 originals: the survivors keep their exact input positions.
        used, compact = np.unique(mapped, return_inverse=True)
        kept_vertices = source_vertices[used]
        kept_faces = compact.reshape(-1, 3).astype(np.int64)

        if self.verify_border and not pinned_held(
            source_vertices, kept_vertices, fixed
        ):
            return self.fallback.simplify(
                source_vertices, source_faces, fixed=fixed, target_faces=target_faces
            )

        return Simplified(
            vertices=kept_vertices,
            faces=kept_faces,
            # The library's own figure is a quadric estimate; the vertex-distance bound is the
            # one every backend reports, and it over-reports, which is the safe direction.
            error=measure_deviation(source_vertices, kept_vertices),
            reached_target=len(kept_faces) <= target_faces,
            backend=self.name,
        )


__all__ = ["SIMPLIFICATION_MESHOPT", "MeshoptSimplifier"]
