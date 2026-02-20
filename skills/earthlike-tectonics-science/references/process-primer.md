# Earthlike Tectonic Process Primer

Use this file to map requested simulator behavior to process-level geology.

## 1. Earthlike Baseline

- Treat plate tectonics as a coupled mantle-lithosphere system where plate motions, boundary geometry, and slab evolution co-control topography and geologic events.
- Treat plate boundaries as persistent entities with identity and lifecycle, not one-frame labels.
- Treat crustal architecture as path-dependent: present-day elevation potential depends on cumulative tectonic history.

## 2. Plate Lifecycle

### 2.1 Birth and Early Growth

- Form new oceanic lithosphere at divergent boundaries (mid-ocean ridges and some back-arc settings).
- Start young oceanic lithosphere buoyant and thin; increase density and thickness with age.
- Encode ridge segmentation and transform offsets to avoid unrealistic perfectly straight boundaries.

### 2.2 Maturation

- Increase negative buoyancy with plate age to support eventual subduction potential.
- Preserve coherent rigid-plate interior motion between boundaries.
- Carry inherited structures (fracture zones, sutures, terrane contacts) forward as future weakness candidates.

### 2.3 Destruction and Recycling

- Remove oceanic lithosphere primarily through subduction.
- Track slab rollback, trench migration, and variable slab dip as first-order controls on arc position and back-arc behavior.
- End convergent segments through collision, polarity reversal, or subduction jump only when preconditions are satisfied.

## 3. Boundary Process Families

### 3.1 Divergent Systems

- Represent continental rifting as staged: extension, thinning, localized magmatism, and eventual breakup/passive margin development.
- Allow failed rifts (aulacogens) when extension does not reach oceanic spreading.
- Use ridge push as secondary forcing compared with slab-driven dynamics.

### 3.2 Convergent Systems

- Distinguish ocean-ocean, ocean-continent, and continent-continent convergence.
- In ocean-ocean settings, emphasize island arcs, forearc basins, and possible arc accretion.
- In ocean-continent settings, emphasize continental arcs, crustal thickening, foreland loading, and episodic uplift.
- In continent-continent collision, enforce subduction shutdown on involved continental margin(s), crustal shortening, thickening, and long-lived orogeny.

### 3.3 Transform Systems

- Model transforms as shear boundaries that transfer displacement between ridge and trench segments.
- Avoid persistent net convergence/divergence across pure transform segments unless a boundary-type transition is occurring.

## 4. Orogeny and Accretion

- Treat orogenies as multi-phase systems: convergence, peak shortening/thickening, and post-orogenic reorganization.
- Represent accretionary growth through terrane docking, arc-continent collision, and sediment addition at convergent margins.
- Keep sutures and inherited belts active as long-lived controls on later deformation and basin placement.

## 5. Volcanism Context

- Link arc volcanism to subduction and volatile transfer.
- Link ridge volcanism to decompression melting during seafloor spreading.
- Link intraplate volcanic provinces to mantle anomalies/plume hypotheses with explicit uncertainty when used.
- Keep volcanism spatially coupled to tectonic setting; avoid isolated random volcanic belts without mechanism.

## 6. Supercontinent and Wilson-Cycle Behavior

- Treat ocean basin opening and closing as Wilson-cycle behavior.
- Treat supercontinent assembly/dispersal as emergent long-timescale behavior with broad periodic uncertainty.
- Preserve memory effects: old sutures and cratonic boundaries commonly guide later rifting and reactivation.

## 7. Terrain-Coupling Expectations

- Raise uplift potential near active convergent belts, collisional zones, and some rift shoulders.
- Raise subsidence potential in passive margins, back-arc/forearc settings, and flexural foreland basins.
- Keep terrain effects temporally lagged relative to tectonic triggers where appropriate; avoid instant mountain creation/deletion.

## 8. Deterministic Modeling Guidance

- Keep all stochastic choices seed-driven and logged in provenance.
- Keep process transitions state-based and thresholded, never random one-off toggles.
- Keep diagnostic traces for boundary history so failures can be reproduced and explained.
