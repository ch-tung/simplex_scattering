# form_factor.py

This module provides robust numerical routines for computing the **scattering form factor** \( F(\mathbf{q}) \) and intensity \( I(\mathbf{q}) = |F(\mathbf{q})|^2 \) of **arbitrary polyhedral shapes** represented as **tetrahedral meshes**.

It combines an **analytic formula** for orthogonal tetrahedra (from Tianjuan Yang et al.) with a novel **adaptive subdivision pipeline** to avoid numerical singularities.

---

## Contents

- `orthogonal_tetrahedra_formfactor(...)`: Analytic form factor for a tetrahedron.
- `scattering_function_tetrahedron(...)`: Scattering from a tetrahedral mesh using the analytic method.

- `adaptive_formfactor(...)`: Recursive, subdivision-based evaluation to avoid singularities.
- `scattering_function_adaptive(...)`: Mesh-level interface for adaptive evaluation.
- `subdivide_tetrahedron(...)`: Perturbative 8-cell subdivision of a tetrahedron.

---

## Methodology

### Analytic Scattering of a Single Tetrahedron

Based on:
> **Tianjuan Yang et al.**, *J. Appl. Cryst.* 55, 1432–1440 (2022)  
> DOI: [10.1107/S160057672201130X](https://doi.org/10.1107/S160057672201130X)

Given three edge vectors \( \mathbf{V}_1, \mathbf{V}_2, \mathbf{V}_3 \) of a tetrahedron, the form factor is expressed in closed form by integrating the complex exponential over the tetrahedral volume:

\[
F(\mathbf{q}) = \int_{\text{tetrahedron}} \rho(\mathbf{r}) \, e^{i \mathbf{q} \cdot \mathbf{r}} \, d\mathbf{r}
\]

---

### Adaptive Subdivision Pipeline

To resolve numerical instabilities, we recursively **subdivide any tetrahedron** where the analytic formula becomes unreliable.

#### Subdivision Pipeline Overview

1. Identify singularity via projection checks on q·Vi

2. Subdivide tetrahedron into 8 cells by computing
   midpoints of all 6 edges, with random jitter to
   avoid symmetric projection degeneracy.

3. Recursively evaluate each sub-tetrahedron with the
   same analytic method, up to a maximum depth.

4. Accumulate results:
   \[
   F_{\text{total}} = \sum_{i=1}^{N_{\text{sub}}} F_i(\mathbf{q})
   \]

