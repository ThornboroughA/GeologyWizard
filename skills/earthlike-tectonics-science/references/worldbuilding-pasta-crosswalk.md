# Worldbuilding Pasta Crosswalk

Map concepts from the Worldbuilding Pasta plate-tectonics post to scientific interpretations usable in a simulation engine.

Primary post:
- [An Apple Pie from Scratch Part Va: Plate Tectonics](https://worldbuildingpasta.blogspot.com/2020/01/an-apple-pie-from-scratch-part-va.html)

## 1. Concept Mapping

| Blog concept | Scientific analog | Simulation implication |
| --- | --- | --- |
| Patterns of plate motion | Relative plate kinematics and force-balance outcomes | Model with persistent Euler-rotation histories and boundary constraints |
| Flat-slab subduction | Low-angle slab geometry segments at convergent margins | Reduce arc volcanism locally, shift deformation inland, and preserve along-strike variability |
| Island arc accretion | Terrane/arc docking at continental margins | Add stepwise crustal growth, sutures, and inherited heterogeneity |
| Subduction jumping | Trench relocation/polarity reorganization | Require explicit trigger conditions and multi-step transition states |
| Subduction invasion | Progressive encroachment of subduction into new lithosphere domains | Enforce time persistence and region-dependent resistance |
| Triple-junction ocean plate formation | Ridge-ridge-ridge junction reorganization and microplate creation | Keep topology consistent at junction nodes through time |
| Simulating plate tectonics (high-level workflow) | Scenario-based reconstruction with process-informed constraints | Translate to deterministic state machine plus diagnostics |

## 2. Reliability Rules

- Treat the post as a design-friendly synthesis, not as a sole scientific authority.
- Resolve every process-level numeric parameter against `source-canon.md` before implementation.
- Prefer explicit uncertainty labels when adopting conceptual mechanisms not tightly constrained in literature.

## 3. Integration Guidance for This Repository

- Convert each selected blog motif into:
  - a boundary/lifecycle rule,
  - a required precondition,
  - a persistence minimum,
  - an expected terrain-coupling signal,
  - and a diagnostic check.
- Keep all transitions deterministic and seed-reproducible.
- Keep non-expert controls simple; expose advanced parameters only in expert mode.
