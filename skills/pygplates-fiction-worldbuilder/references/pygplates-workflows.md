# PyGPlates Workflows

## Setup and Runtime
- Prefer `conda` for pyGPlates 1.x installs (binary extension friendly).
- `pip` wheels are supported; Linux minimal environments may need `libGL.so.1` installed.
- Remove old pre-1.0 pyGPlates paths from `PYTHONPATH` to avoid import conflicts.

## Workflow A: Rigid reconstruction over many times
Use this for coastlines, terranes, isochrons, and other regular features.

```python
import pygplates

rotation_model = pygplates.RotationModel("rotations.rot")
reconstruct_model = pygplates.ReconstructModel("features.gpmlz", rotation_model)

for t in range(200, -1, -10):
    snapshot = reconstruct_model.reconstruct_snapshot(t)
    snapshot.export_reconstructed_geometries(f"reconstructed_{t}Ma.shp")
```

Notes:
- Reuse `ReconstructModel` for performance across many times.
- `reconstruct(...)` is fine for one-off calls.
- Export format depends on output filename extension.

## Workflow B: Reverse reconstruction
Use when input geometry is already paleo-positioned and you need to store present-day geometry.

```python
pygplates.reverse_reconstruct("features.gpml", "rotations.rot", 10)
```

Notes:
- This modifies feature geometries in-place (or rewrites files when filenames are passed).
- `reconstruction_time` also sets geometry import time metadata.

## Workflow C: Assign plate IDs by partitioning
Use for raw points/lines/polygons without `reconstructionPlateId`.

```python
features = pygplates.partition_into_plates(
    "static_polygons.gpml",
    "rotations.rot",
    "features_to_partition.gpml",
    partition_method=pygplates.PartitionMethod.most_overlapping_plate,
)
```

Notes:
- Default copied property is reconstruction plate ID.
- Features to partition are treated as geometry at the target reconstruction time (not auto-reconstructed first).
- Sorting of partitioning plates affects overlap resolution when polygons overlap.

## Workflow D: Resolve topologies and deforming networks
Use for dynamic boundaries, deforming regions, and section-level output.

```python
resolved_topologies = []
resolved_sections = []
pygplates.resolve_topologies(
    "plate_polygons_and_networks.gpml",
    "rotations.rot",
    resolved_topologies,
    50,
    resolved_sections,
)
```

Notes:
- Defaults output boundary and network topologies.
- `ResolveTopologyParameters` controls strain clamping/smoothing and rift settings.
- Topological networks exported to OGR GMT/ESRI shapefile may not reload in GPlates as networks.

## Workflow E: Topological point tracking through time
Use for advecting seed points and retrieving velocity/strain/scalars through geologic history.

```python
topo_model = pygplates.TopologicalModel("topologies.gpml", "rotations.rot")
points = [pygplates.PointOnSphere(lat, lon) for lat, lon in [(0, 0), (20, 140)]]
span = topo_model.reconstruct_geometry(points, initial_time=100)
pts_50 = span.get_geometry_points(50)
vel_50 = span.get_velocities(50)
```

Notes:
- Topological reconstruction currently operates on point geometries.
- Default `time_increment` is 1 Myr; larger increments reduce fidelity.
- Outside topology coverage, fallback behavior depends on `reconstruction_plate_id`.

## Workflow F: Velocities and boundary statistics
- Snapshot point velocities in static polygons: `ReconstructSnapshot.get_point_velocities(...)`.
- Generic velocities from finite rotations: `pygplates.calculate_velocities(...)`.
- Plate-boundary metrics (convergence/divergence/obliquity): `TopologicalSnapshot.calculate_plate_boundary_statistics(...)`.

## Workflow G: File I/O and interoperability
- Use `FeatureCollection` to read/write GPML, GPMLZ, rotation formats, shapefile, GeoJSON, GeoPackage, GMT.
- Reconstruction exports support `.shp`, `.geojson`/`.json`, `.gmt`, `.xy`.
- Shapefile attribute retention has limitations when combining multiple input collections.

## Performance Checklist
- Reuse model objects (`RotationModel`, `ReconstructModel`, `TopologicalModel`).
- Avoid repeatedly creating temporary models inside tight loops.
- Cache derived outputs by time step where user scrubbing/animation is expected.
