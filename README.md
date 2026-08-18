# fabriks

**A level-of-detail (LOD) wire format for 3D meshes.** `fabriks` partitions surface collections into an octree of Parquet files, allowing viewers to fetch only the required spatial detail per frame without running a dedicated server.

It is strictly a *serializer*: zero network dependencies, client code, or storage opinions. Hand it an abstract store, and it writes the same layout to disk, S3, GCS, or memory.

---

**Key Properties**

* **Octree Partitioning**: Level 0 is full detail. Each coarser level combines 8 finer cells into 1 cell with a quarter of the face count.
* **Dual Catalogs**: `catalog/cells.parquet` (spatial index read once at mount) and `catalog/objects.parquet` (inverted index mapping object IDs to cell keys).
* **Seamless Boundaries**: Vertices on cell faces stay pinned during decimation—fine cells meet coarse neighbors without visual cracks.
* **Range-Query Optimized**: Cells map to individual Parquet row groups sorted in Morton order for efficient byte-span fetches.

---

**Layout & Partitioning**

```text
my-collection/
  fabriks.json                  <- Root manifest (written last; atomic completion signal & checksums)
  catalog/cells.parquet        <- Spatial index (maps level & cell_key to row group byte locators)
  catalog/objects.parquet      <- Identity index (maps object IDs to cell keys for isolation/extraction)
  level0/part-00000.parquet    <- Level 0 full-detail geometry
  level1/part-00000.parquet    <- Decimated coarse-level geometry (L=1, L=2, ...)

```

* **Spatial Octree**: Space is partitioned into uniform 3D cells (`cell_size`). Each parent cell at level `L >= 1` merges 8 child cells (2×2×2) from level `L - 1`.
* **Seam Locking**: Vertices on cell boundary planes stay pinned during decimation so fine and coarse cells tile seamlessly without visual gaps.
* **Morton Ordering**: Geometry row groups are sorted along a Z-order curve (Morton space) for spatially compact, range-query-friendly byte fetches.
* **Signable Paths**: Every directory and file name is letters, digits, `-` and `.` — no `=`, no `~`, nothing else outside RFC 3986's unreserved set. A key spelled this way percent-encodes to itself, so SigV4's canonical request is the same string whichever SDK, proxy or hand-rolled presigner builds it, and a presigned fetch cannot fail with `SignatureDoesNotMatch` over an encoding disagreement.

---

**Install**

```bash
pip install fabriks              # Core writer/reader (trimesh, fast-simplification, scipy, shapely)
pip install 'fabriks[obstore]'   # + S3 / GCS / Azure support via obstore
pip install 'fabriks[meshopt]'   # + MESHOPT blob decoding

```

---

**Sizing & Writing**

```python
import trimesh
from obstore.store import LocalStore
import fabriks

objects = {
    7: trimesh.creation.icosphere(radius=18.0).apply_translation([200, 160, 60]),
    3: trimesh.creation.box(extents=[40, 24, 16]).apply_translation([90, 70, 40]),
}

# Derive grid dimensions from byte targets (optional)
grid_plan = fabriks.plan_grid(objects, cell_bytes=16 * 1024, layer_bytes=128 * 1024)

# Write meshes directly to a store
manifest = fabriks.write_meshes(
    objects,
    store=LocalStore("/data"),
    prefix="my-collection",
    **grid_plan.as_kwargs(),
)

```

---

**Reading & Planning**

```python
import fabriks
from obstore.store import LocalStore

collection = fabriks.open_collection(LocalStore("/data"), "my-collection")

# Camera or voxel error budget planning
camera = fabriks.Camera.perspective((0, 0, 500), fov_y=0.8, viewport_height=1080)
plan = collection.plan(camera=camera, pixel_budget=1.0)

# Synchronous batch fetch
for cell in collection.read_cells([(e.level, e.cell) for e in plan]):
    draw(cell.vertices, cell.faces)

# Reassemble single object across cells
mesh = collection.object_mesh(7)
collection.release()

```

**Async Reading**

```python
collection = await fabriks.aopen_collection(S3Store(...), "my-collection")
plan = collection.plan(camera=camera)
cells = await collection.aread_cells([(e.level, e.cell) for e in plan], concurrency=16)

```

---

**Foreign Formats**

`fabriks.contrib` reads the formats a model usually arrives in and writes them back out. 

```python
from fabriks.contrib import read_glb, write_obj
import fabriks

imported = read_glb("bunny.glb", store, "bunny", levels=4)

collection = fabriks.open_collection(store, "bunny")
blob = write_obj(collection, level=2)          # or write to a path: write_obj(collection, "out.obj")

```

| Format | Identity | Notes |
| --- | --- | --- |
| `GLB` | per object | One file; reads from bytes or a path. The best round trip here. |
| `GLTF` | per object | Several files — `write_gltf` returns `{name: bytes}`, which `read_gltf` takes back. |
| `OBJ` | per object | Identity rides on `o` markers; materials are skipped. |
| `PLY` / `STL` / `OFF` | single object | No object names in the format; a write merges every object into one mesh. |

*3MF and DAE are deliberately absent* — trimesh reads neither without `networkx` / `pycollada`, and a function that is always an `ImportError` is a worse contract than one that is not there.

**The fit.** A collection lives in a positive, whole-number voxel space; a mesh file is usually centred on the origin and a couple of units across, which fabriks refuses outright and could not partition anyway. So `read_*` scales the model so its longest side spans `resolution` voxels (512 by default) and shifts it clear of the origin. The transform and each object's source name ride on the object catalog as extra columns — which the format allows on purpose — so `write_*` inverts the import's own numbers and a round trip lands back on the source's coordinates. Pass `fit=VoxelFit.identity()` for a model already in voxel coordinates.

**Identity.** An object named with a plain integer keeps that number as its id, so a collection written out and read back comes home with the ids it left with. Any other name is assigned the lowest free id and its string is kept in the catalog's `name` column, readable as `collection.objects[id].name`.

**Round trips are not exact.** Positions are 16-bit quantized per cell, so geometry comes back within about one quantum (`cell_size / 65535`) of where it went in, never on it. Compare bounds and shape, not vertex arrays — `object_mesh` welds an object's pieces back together, so neither vertex count nor order survives either.

---

**Configuration**

| Protocol | Options | Notes |
| --- | --- | --- |
| **Stores** | `LocalStore`, `S3Store`, `DirectoryStore`, `MemoryStore` | Implements `put`, `get`, `list`, and optional `get_range`. |
| **Simplifiers** | `"QUADRIC"` (default), `"GREEDY"` | `QUADRIC` uses `fast-simplification` with `preserve_border=True`. Custom simplifiers implement `fabriks.Simplifier`. |
| **Codecs** | `codec`: `NONE`, `MESHOPT`<br>

<br>`compression`: `NONE`, `ZSTD` | Default is `NONE`/`NONE` (raw little-endian arrays for zero-copy GPU upload). |

---

**Verification**

```python
report = fabriks.verify(collection, tier="geometry")  # "structure" | "blobs" | "geometry"
if not report:
    print(report)

```

---

**Coordinates & Conventions**

`fabriks` addresses dimensions strictly by array slot `(0, 1, 2)`. Inputs from `(z, y, x)` volumes pass directly via `cell_size=(z_size, y_size, x_size)` without transposition. Axis labels in column names are slot identifiers, not physical coordinate claims.