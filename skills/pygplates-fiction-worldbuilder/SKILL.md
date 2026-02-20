---
name: pygplates-fiction-worldbuilder
description: Build and iterate user-facing apps that wrap PyGPlates and GPlates data for fictional or educational geologic history creation. Use when tasks involve PyGPlates API design, reconstruction/topology workflows, feature and rotation data modeling, plate-ID assignment, velocity or strain queries, data import/export, or translating expert tectonics workflows into non-expert UX.
---

# Pygplates Fiction Worldbuilder

## Purpose
Use this skill to design and implement a product layer above PyGPlates where non-experts can create and explore fictional tectonic history with safe defaults.

## Reference Loading
Load only what is needed:
- For system concepts and object relationships: `references/pygplates-core-model.md`.
- For API call patterns and implementation choices: `references/pygplates-workflows.md`.
- For feature/property/file semantics and conversion constraints: `references/gplates-data-model-and-formats.md`.
- For worked examples and notebook routing: `references/tutorial-notebooks-map.md`.

## Workflow
1. Classify the request:
- `rigid reconstruction`: regular features moved by plate ID and rotation model.
- `topology/deformation`: resolved boundaries/networks, strain, topological point tracking.
- `plate assignment`: infer or update plate IDs from partitioning plates.
- `data I/O`: import/export/convert GPML, rotations, shapefile, GeoJSON, GMT.
- `product UX mapping`: simplify expert settings into guided user controls.
2. Select API depth:
- Prefer `ReconstructModel` and `TopologicalModel` for repeated time queries.
- Use one-shot functions (`reconstruct`, `resolve_topologies`, `partition_into_plates`) for batch utilities.
3. Preserve data invariants:
- Ensure reconstructable features have geometry, `reconstructionPlateId`, and valid time when needed.
- Keep rotations and feature times in the same temporal frame (Ma before present).
- Treat topology-driven reconstruction as point-based unless you explicitly build a mesh/sampling strategy.
4. Build user-facing defaults:
- Default anchor plate from model unless user requests otherwise.
- Default topological `time_increment=1` Myr for stability.
- Expose named plates in UI and map to internal plate IDs.
- Keep advanced deformation parameters hidden behind an "expert" panel.
5. Validate behavior:
- Verify a known point/path reconstructs correctly at multiple times.
- Check features outside plate coverage and report fallback behavior.
- Confirm exports load in GPlates when interoperability is required.

## Product Guardrails
- Use `reverse_reconstruct` only when you intentionally rewrite stored geometries in-place.
- Reuse `RotationModel`/`ReconstructModel`/`TopologicalModel` instances across time loops for performance.
- When partitioning with overlapping polygons, set deterministic sorting and document chosen policy.
- Explain left/right plate semantics whenever editing subduction or transform boundaries.

## Output Requirements
- Provide explicit assumptions about time range, anchor plate, and model coverage.
- Surface API-level caveats that affect user-facing correctness.
- Prefer minimal reproducible examples over abstract guidance.
