# Tutorial Notebooks Map

Use this to quickly route a task to the most relevant notebook patterns from the `pygplates-tutorials` repository.

## Core onboarding
- `pygplates-getting-started.ipynb`
  - Basic import, simple reconstruct, and plotting scaffolding.
- `pygplates-fileio.ipynb`
  - Feature collection loading, filtering, merging, and writing.

## Plate assignment and reconstruction basics
- `pygplates-assign-plateids.ipynb`
  - `partition_into_plates`, `FeatureCollection`, and reconstruction after assignment.
- `pygplates-reconstruct-gmt-file.ipynb`
  - Importing GMT content, partitioning, then reconstruction.
- `plate-tectonics-over-time.ipynb`
  - Time-loop reconstruction examples for teaching/exploration.

## Rotations, flowlines, motion paths
- `pygplates-Working-with-Rotation-Poles.ipynb`
  - Finite and stage pole workflows with `RotationModel`.
- `pygplates-Flowlines.ipynb`
  - Construct and reconstruct flowline features.
- `pygplates-Motion-Paths.ipynb`
  - Motion path feature construction and time/rate interpretation.

## Topologies and deformation
- `pygplates-topologies.ipynb`
  - Resolve topological polygons and query outputs.
- `track-point-through-topologies.ipynb`
  - Point tracking via topologies and partitioning.
- `pygplates-plate-boundary-convergence-and-deforming-region-strain-rates.ipynb`
  - Boundary convergence and deforming-region strain-rate analysis.

## Velocities and kinematics
- `velocity-basics.ipynb`
  - `calculate_velocities`, delta-time settings, units.
- `velocity-fields.ipynb`
  - Velocity domain generation, partitioning, and vector visualization.

## Domain-specific examples
- `pygplates-reconstruct-to-birth-time.ipynb`
  - Birth-time reconstruction patterns.
- `hotspot-tracking-movie-maker.ipynb`
  - End-to-end hotspot tracking and animation workflow.
- `time-latitude-plots.ipynb`
  - Time-latitude analysis outputs.

## Useful library scripts in tutorial repo
- `libs/topology_plotting.py`
  - Plate velocity extraction and topology plotting helper functions.
- `libs/velocity_utils.py`
  - Velocity-domain helper functions.
- `libs/subduction_convergence.py`
  - Advanced subduction convergence calculations over time.

## Practical use in app development
- Start with notebook patterns to validate scientific behavior first.
- Extract only the API calls and data contracts into production code.
- Replace plotting-heavy tutorial logic with your app's rendering layer.
