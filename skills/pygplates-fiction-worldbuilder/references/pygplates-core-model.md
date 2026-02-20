# PyGPlates Core Model

## Mental Model
PyGPlates is a Python API over GPlates functionality. The core model is:
- `Feature` and `FeatureCollection`: geology objects and containers.
- `RotationModel`: time-dependent plate rotations.
- Reconstruction layer: rigidly move features through time.
- Topology layer: resolve dynamic plate boundaries/networks and optionally deform points through time.

Think in this order:
1. Load/construct features and rotations.
2. Choose rigid or topology-driven reconstruction.
3. Query outputs at times of interest.
4. Export to interoperable formats.

## Main Object Families

### Data and identity
- `pygplates.Feature`
- `pygplates.FeatureCollection`
- `pygplates.FeatureType`, `pygplates.PropertyName`, `pygplates.ScalarType`

### Reconstruction (rigid)
- `pygplates.ReconstructModel`: cached history for repeated time queries.
- `pygplates.ReconstructSnapshot`: one reconstruction time snapshot.
- `pygplates.reconstruct(...)`: convenience one-shot reconstruction.
- `pygplates.reverse_reconstruct(...)`: reverse from paleo geometry back to present-day storage geometry.

### Rotation
- `pygplates.RotationModel`
- `pygplates.FiniteRotation`
- `pygplates.ReconstructionTree` (advanced)

### Topology and deformation
- `pygplates.TopologicalModel`
- `pygplates.TopologicalSnapshot`
- `pygplates.resolve_topologies(...)`
- `pygplates.ResolveTopologyParameters`
- `pygplates.ReconstructedGeometryTimeSpan`

### Plate partitioning and assignment
- `pygplates.partition_into_plates(...)`
- `pygplates.PlatePartitioner`

### Geometry and vectors
- `pygplates.PointOnSphere`, `PolylineOnSphere`, `PolygonOnSphere`, `MultiPointOnSphere`
- `pygplates.Vector3D`, `pygplates.LocalCartesian`

## Time Semantics
- Geological time is in Ma before present.
- Total rotation: present day (0 Ma) to target time.
- Stage rotation: between two non-zero times.
- Anchor plate controls frame of reference in reconstructions.

## Rigid vs Topological Reconstruction
- Rigid reconstruction uses feature properties (especially `reconstructionPlateId`) plus rotations.
- Topological reconstruction uses resolved topologies and can include deformation.
- Topological point reconstruction is incremental over time slots and supports deactivation behavior.

## Product Mapping for Non-Experts
- Expose conceptual objects: "plates", "boundaries", "landmasses", "events".
- Keep plate IDs internal; show names/labels in UI.
- Provide simple presets:
  - "Rigid world" -> `ReconstructModel` only.
  - "Deforming world" -> `TopologicalModel` with default parameters.
- Keep advanced controls optional:
  - anchor plate ID
  - strain clamping/smoothing
  - rift profile parameters
