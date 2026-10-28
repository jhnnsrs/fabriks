"""How a coarse level is made, and how to choose or replace the algorithm that makes it.

A coarser level is the same surfaces with fewer triangles, and the writer's whole obligation is
two-fold: hit a face budget (``decimation``), and leave the cut boundary exactly where it was
(``boundary: LOCKED``), so a fine cell drawn beside a coarse one meets it without a crack.

Everything else about *how* is a choice, so the backend is named the way a codec is -- a value
out of a small vocabulary, resolved to an implementation -- rather than a hardcoded loop::

    fabriks.build_collection(objects, cell_size=...)                      # MESHOPT if installed, else QUADRIC
    fabriks.build_collection(objects, cell_size=..., simplifier="GREEDY")
    fabriks.build_collection(objects, cell_size=..., simplifier=fabriks.GreedyEdgeCollapse(placement="onto_fixed"))

Three backends ship:

``MESHOPT`` (:class:`~fabriks.simplifiers.meshopt.MeshoptSimplifier`)
    The default where it is installed (``fabriks[meshopt]``; it builds from source), backed by
    `meshoptimizer <https://github.com/zeux/meshoptimizer>`_'s
    ``simplifyWithAttributes``. Its ``vertex_lock`` pins **exactly** the ``fixed`` mask -- this
    level's cell planes -- and it never repositions a vertex, so every survivor keeps its exact
    input position. That is the format's constraint stated directly, where ``QUADRIC`` can only
    pin the whole topological boundary, inherited level-0 seams included.

``QUADRIC`` (:class:`~fabriks.simplifiers.quadric.QuadricSimplifier`)
    Backed by `fast-simplification
    <https://github.com/pyvista/fast-simplification>`_ -- Sven Forstmann's quadric collapse.
    It is run with ``preserve_border=True``, which pins every vertex on the open boundary at
    **exactly** its input position while leaving interior vertices free to move to the position
    the quadric says is best. That split is precisely what the format wants: the cut curve is
    the only thing a neighbouring cell shares, so it is the only thing that must not move, and
    holding the interior still as well would cost shape quality for nothing.

``GREEDY`` (:class:`~fabriks.simplifiers.greedy.GreedyEdgeCollapse`)
    Shortest-edge collapse in pure numpy, for the cases where a heavily pinned boundary stops
    the quadric collapse reaching a budget: this one will collapse an edge *onto* a locked
    vertex. Lower quality, and its error estimate is how far it moved a vertex, which on a
    collapsing object is about that object's radius.

The package is split the way the stores and codecs are: :mod:`~fabriks.simplifiers.protocol`
states what a backend is, :mod:`~fabriks.simplifiers.quadric` and
:mod:`~fabriks.simplifiers.greedy` and :mod:`~fabriks.simplifiers.meshopt` are the three implementations,
:mod:`~fabriks.simplifiers.measure` is the error bound neither library provides, and
:mod:`~fabriks.simplifiers.registry` is the name-to-backend table and the retry policy the
builder drives them through.
"""

from fabriks.simplifiers.greedy import SIMPLIFICATION_GREEDY, GreedyEdgeCollapse
from fabriks.simplifiers.measure import boundary_held, measure_deviation, pinned_held
from fabriks.simplifiers.meshopt import SIMPLIFICATION_MESHOPT, MeshoptSimplifier
from fabriks.simplifiers.protocol import Simplified, Simplifier
from fabriks.simplifiers.quadric import SIMPLIFICATION_QUADRIC, QuadricSimplifier
from fabriks.simplifiers.registry import (
    default_simplification,
    resolve_simplifier,
    simplifier_for,
    simplify_to_target,
)

__all__ = [
    "SIMPLIFICATION_GREEDY",
    "SIMPLIFICATION_MESHOPT",
    "SIMPLIFICATION_QUADRIC",
    "GreedyEdgeCollapse",
    "MeshoptSimplifier",
    "QuadricSimplifier",
    "Simplified",
    "Simplifier",
    "boundary_held",
    "default_simplification",
    "measure_deviation",
    "pinned_held",
    "resolve_simplifier",
    "simplifier_for",
    "simplify_to_target",
]
