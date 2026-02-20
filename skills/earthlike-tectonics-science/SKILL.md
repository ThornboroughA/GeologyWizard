---
name: earthlike-tectonics-science
description: Scientific guardrails and reference workflow for Earth-like tectonic world simulation. Use when designing or reviewing plate kinematics, boundary evolution, supercontinent cycles, subduction/rifting/orogeny/accretion/volcanism logic, or when adding plausibility checks and diagnostics for deterministic worldbuilding tools.
---

# Earthlike Tectonics Science

## Overview

Use this skill to ground tectonic simulation decisions in Earth-system process knowledge while keeping outputs deterministic, reproducible, and explicit about uncertainty.

## Reference Loading

Load only what is needed:
- For process mechanics and event persistence expectations, read `references/process-primer.md`.
- For numeric guardrails and rough Earth-like bounds, read `references/quantitative-ranges.md`.
- For implementation-time validation and diagnostics behavior, read `references/plausibility-checks.md`.
- For canonical textbooks, papers, and agency references, read `references/source-canon.md`.
- For mapping the Worldbuilding Pasta post to scientific interpretations, read `references/worldbuilding-pasta-crosswalk.md`.

## Workflow

1. Define assumptions.
- Assume Earth-like gravity, heat budget, and material behavior unless the request overrides it.
- State active simulation mode (`fast_plausible` or `hybrid_rigor`).
- State whether results are strict constraints or exploratory guidance.
2. Map features to process classes.
- Map requested behavior into plate kinematics, boundary semantics, event synthesis, and terrain coupling.
- Keep climate out of scope.
3. Apply sanity ranges.
- Use `references/quantitative-ranges.md` to set fail/warn bands.
- Keep hard failures for geometric impossibilities and deterministic invariants.
- Keep soft warnings for low-confidence but still plausible edge cases.
4. Run plausibility checks.
- Use `references/plausibility-checks.md` to evaluate continuity, boundary consistency, lifecycle realism, and coverage diagnostics.
- Report every failed check explicitly.
5. Produce implementation guidance.
- Recommend deterministic defaults and provenance fields.
- Include uncertainty text where evidence is weak or parameter ranges are broad.
6. Cite reference basis.
- Tie recommendations to specific entries in `references/source-canon.md`.
- Call out when guidance is heuristic from `references/worldbuilding-pasta-crosswalk.md`.

## Non-Negotiables

- Preserve deterministic behavior for fixed seed and config.
- Do not suppress uncertainty or coverage gaps; surface them in diagnostics.
- Prefer physically informed approximations over decorative noise.
- Preserve non-expert usability by hiding advanced controls unless explicitly requested.

## Output Contract

When using this skill, return:
- Explicit assumptions (planet baseline, mode, timespan).
- Process-level rationale (why this tectonic behavior is plausible).
- Check outcomes (fail/warn/pass with thresholds).
- Remaining uncertainty and recommended next checks.
