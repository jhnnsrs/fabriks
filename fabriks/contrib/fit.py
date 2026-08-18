"""The transform that puts a source model into the octant fabriks partitions.

A collection lives in a **positive, integer-voxel** space, and a mesh file usually does not. Two
facts about the builder force this module to exist:

- ``cell`` indices are non-negative, so geometry below the origin is refused outright rather than
  wrapped or clamped -- a ``.glb`` centred on the origin, which is most of them, cannot be built
  as it stands;
- ``cell_size`` is three whole numbers of at least one voxel, so a model one unit across has no
  cell smaller than itself. It becomes a single cell with no octree above it, and every level of
  detail the format exists to provide is gone.

So an import scales the model up and shifts it into the positive octant, and an export undoes
exactly that. The transform is a **similarity** -- one scale for all three components -- because
the alternative distorts: a per-component fit stretches a model to fill its voxel box, and every
render of it afterwards is wrong in a way nothing downstream can detect.

The fit travels with the collection in the object catalog (see :mod:`fabriks.contrib.catalog`),
so an export inverts the import's own numbers rather than guessing at them.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

#: How many voxels the longest side of a fitted model spans, when nothing says otherwise.
#: Large enough that the octree has somewhere to go, small enough that a cell stays a sensible
#: number of bytes -- ``fabriks.plan_grid`` is what actually sizes the cells within it.
DEFAULT_RESOLUTION = 512


@dataclass(frozen=True)
class VoxelFit:
    """The similarity transform that put a source model into fabriks's positive voxel octant.

    ``voxels = source * scale + translation``, and :meth:`invert` is that read backwards. The
    default is the identity, which is what a caller whose model is *already* in voxels wants.
    """

    #: The uniform scale applied to every component. One scale, never three -- see the module.
    scale: float = 1.0
    #: The shift applied after scaling, which is what lands the model in the positive octant.
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @classmethod
    def identity(cls) -> VoxelFit:
        """The fit that changes nothing, for a model already in voxel coordinates."""
        return cls()

    @classmethod
    def for_bounds(
        cls,
        low: npt.ArrayLike,
        high: npt.ArrayLike,
        *,
        resolution: int = DEFAULT_RESOLUTION,
    ) -> VoxelFit:
        """The fit that lands ``[low, high]`` at the origin corner, ``resolution`` voxels long.

        The longest side sets the scale, so the shortest one keeps its proportion instead of
        being stretched to match. A degenerate bound -- a flat or empty model, where the longest
        side is zero -- yields the identity rather than a division by zero: there is no
        meaningful scale for a model with no extent, and refusing here would only move the
        failure somewhere less clear.
        """
        origin = np.asarray(low, dtype=np.float64)
        extent = np.asarray(high, dtype=np.float64) - origin
        longest = float(np.max(extent)) if extent.size else 0.0
        if not np.isfinite(longest) or longest <= 0.0:
            return cls.identity()

        scale = float(resolution) / longest
        shift = -origin * scale
        return cls(scale=scale, translation=(float(shift[0]), float(shift[1]), float(shift[2])))

    @property
    def is_identity(self) -> bool:
        """Whether this fit leaves coordinates alone."""
        return self.scale == 1.0 and self.translation == (0.0, 0.0, 0.0)

    def apply(self, vertices: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Take ``(n, 3)`` source coordinates into voxel coordinates."""
        return np.asarray(vertices, dtype=np.float64) * self.scale + np.asarray(
            self.translation, dtype=np.float64
        )

    def invert(self, vertices: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Take ``(n, 3)`` voxel coordinates back to source coordinates."""
        if self.scale == 0.0:  # only reachable if a catalog carried one; do not divide by it
            return np.asarray(vertices, dtype=np.float64)
        return (
            np.asarray(vertices, dtype=np.float64) - np.asarray(self.translation, dtype=np.float64)
        ) / self.scale

    def to_dict(self) -> dict[str, Any]:
        """The fit as plain numbers, for a catalog row or a printout."""
        return {"scale": self.scale, "translation": list(self.translation)}

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> VoxelFit:
        """Read back what :meth:`to_dict` wrote, treating nothing at all as the identity."""
        if not raw:
            return cls.identity()
        shift = tuple(float(component) for component in raw.get("translation", (0.0, 0.0, 0.0)))
        if len(shift) != 3:
            return cls.identity()
        return cls(scale=float(raw.get("scale", 1.0)), translation=(shift[0], shift[1], shift[2]))


__all__ = ["DEFAULT_RESOLUTION", "VoxelFit"]
