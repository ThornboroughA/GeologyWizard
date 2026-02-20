# GPlates Data Model and Formats

## Feature-Centric Data Model
GPlates models geology as features with typed properties.

Minimum practical requirements for reconstruction:
- Geometry
- `reconstructionPlateId`
- Valid time (`gml:validTime`) for time-limited features

Common geometry properties:
- `position`: point-like features
- `centerLineOf`: line-like features
- `outlineOf`: polygon-like features

Important distinction:
- `reconstructionPlateId` drives reconstruction.
- `reconstructedPlateId` is internal snapshot bookkeeping.

## Time Model
- Valid time is `[begin, end]` in Ma.
- `begin` can be "Distant Past".
- `end` can be "Distant Future".

## Plate-Boundary Semantics
Many boundary features use:
- `leftPlate`
- `rightPlate`
- `conjugatePlateId`

`left` and `right` depend on digitization direction of boundary geometry. Always preserve and explain this in editing tools.

Subduction boundaries commonly add:
- `subductingSlab` ("left"/"right")

Rift/ridge and island arc commonly use:
- `isActive`

## GPML and Conversion Caveats
- GPML is GPlates' XML/GML-based native format.
- GPML can preserve richer tectonic semantics than generic GIS formats.
- Shapefile conversion typically loses semantic richness unless mapped carefully.
- For shapefile input, geometry comes from `.shp` and attributes from `.dbf`; all sidecar files must be present.

## FeatureCollection File Support (PyGPlates)
Read/write support includes:
- `.gpml`, `.gpmlz`, `.gpml.gz`
- `.rot`, `.grot`, `.dat`, `.pla`
- `.shp`, `.geojson`, `.json`, `.gpkg`, `.gmt`
- `.xy` is write-only in FeatureCollection context

## Reconstruction/Topology Export Support
Functions like `reconstruct(...)` and `resolve_topologies(...)` export to:
- `.shp`
- `.geojson` / `.json`
- `.gmt`
- `.xy`

Practical caveat:
- Resolved topological networks exported to some GIS formats may not reload as networks in GPlates.

## Product Implications for Fictional Worldbuilding
- Preserve a high-fidelity internal format (GPML-like model) even if UI imports CSV/GIS layers.
- Generate and maintain stable feature IDs for editing history.
- Store user-friendly labels separately from tectonic identifiers.
- Add validation for required reconstruction fields before simulation/export.
