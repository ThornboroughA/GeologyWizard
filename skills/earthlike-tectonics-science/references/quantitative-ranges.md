# Earthlike Quantitative Ranges

Use this file for plausibility guardrails, not absolute laws. Prefer warnings outside these ranges unless the model violates conservation/geometry constraints.

## 1. Kinematics and Timescales

| Metric | Earthlike guidance | Use in simulator |
| --- | --- | --- |
| Plate velocity (full rate) | Usually a few cm/yr; values can span roughly sub-cm/yr to about 10+ cm/yr | Use as a soft band; flag persistent extreme values as warnings |
| Seafloor spreading rate (full) | Slow to fast ridges observed; order-of-magnitude range is tens to >100 mm/yr | Classify ridge style (slow/intermediate/fast) for morphology proxies |
| Oceanic crust lifetime | Most oceanic crust is geologically young; oldest preserved seafloor is far younger than most continental crust (order 100-300 Myr) | Warn if oceanic domains persist unrealistically long without subduction |
| Supercontinent cycle interval | Broadly debated, often discussed on several-hundred-Myr cadence | Treat as long-range tendency, not strict periodic target |

## 2. Crust and Lithosphere Structure

| Metric | Earthlike guidance | Use in simulator |
| --- | --- | --- |
| Oceanic crust thickness | Commonly near 6-7 km | Use as default for new oceanic crust unless modeled otherwise |
| Continental crust thickness | Commonly ~30-50 km; thickened orogens can exceed this | Use regional thickening in collisional belts; avoid globally uniform values |
| Subduction seismicity depth | Earthquakes can track slabs from shallow to deep upper mantle (~700 km max) | Use as plausibility check when synthesizing slab depth fields |

## 3. Process Duration Heuristics

| Process | Earthlike heuristic duration | Use in simulator |
| --- | --- | --- |
| Rift initiation to breakup | Often multi-stage and prolonged (roughly tens of Myr) | Require persistence before converting continental rift to oceanic spreading |
| Orogenic phase persistence | Commonly tens to >100 Myr across full lifecycle | Avoid single-step mountain creation/removal |
| Arc migration/reorganization | Usually progressive over many Myr | Smooth shifts in arc/trench geometry over timesteps |

## 4. Practical Threshold Policy

- Use `hard fail` only for impossible kinematics or broken continuity:
  - Plate interior not moving rigidly with own rotation model.
  - Boundary segment simultaneously tagged with contradictory primary types.
  - Non-deterministic outputs for identical seed/config.
- Use `warning` for unusual but potentially plausible states:
  - Very high plate speeds with limited duration.
  - Long-lived stagnant ocean basin without convergent consumption.
  - Excessive boundary-type flipping within short intervals.
- Use `info` for low-confidence inference:
  - Plume-related intraplate volcanism attribution.
  - Specific supercontinent periodicity claims.

## 5. Uncertainty Guidance

- Prefer ranges and probability language over single-point values.
- Distinguish observationally constrained behavior from conceptual model choices.
- Document every override that intentionally departs from Earthlike defaults.
