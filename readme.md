# form_factor.py

This module provides robust numerical routines for computing the **scattering form factor** $F(\mathbf{q})$ and intensity $I(\mathbf{q}) = |F(\mathbf{q})|^2$ of **arbitrary polyhedral shapes** represented as **tetrahedral meshes**.

It combines an **analytic formula** for orthogonal tetrahedra (from Tianjuan Yang et al.) with an **adaptive subdivision pipeline** to avoid numerical singularities.

---

## Contents

- `orthogonal_tetrahedra_formfactor(...)`: Analytic form factor for a tetrahedron.
- `scattering_function_tetrahedron(...)`: Scattering from a tetrahedral mesh using the analytic method.

- `jittered_formfactor(...)`: Random perturbation to the vertices of tetrahedra exhibiting singularities
- `adaptive_formfactor(...)`: Recursive, subdivision-based evaluation to avoid singularities.
- `subdivide_tetrahedron(...)`: Perturbative 8-cell subdivision of a tetrahedron.
- `scattering_function_adaptive(...)`: Mesh-level interface for adaptive evaluation.

---

## Methodology

### Analytic Scattering of a Single Tetrahedron

Based on:
> **Tianjuan Yang et al.**, *J. Appl. Cryst.* 55, 1432–1440 (2022)  
> DOI: [10.1107/S160057672201130X](https://doi.org/10.1107/S160057672201130X)

Given three edge vectors $\mathbf{V}_1, \mathbf{V}_2, \mathbf{V}_3$ of a tetrahedron, the form factor is expressed in closed form by integrating the complex exponential over the tetrahedral volume:

$$
F(\mathbf{q}) = \int_{\text{tetrahedron}} \rho(\mathbf{r}) \, e^{i \mathbf{q} \cdot \mathbf{r}} \, d\mathbf{r}
$$

---

## Numerical Regularization

### Recursive Subdivision
The adaptive_formfactor(...) function subdivides problematic tetrahedra into smaller ones and recomputes the form factor recursively. This is more robust in pathological cases but introduces additional computational overhead.

### Vertex Jitter (Non-Recursive)
The jittered_formfactor(...) function instead applies a one-time small random perturbation to the vertices of tetrahedra exhibiting singularities. This avoids recursive subdivision, offering faster performance with acceptable error for most cases.

To choose between these two modes, change the accumulation line in your scattering loop:
```python
form_factors += adaptive_formfactor(V1[i], V2[i], V3[i], qx, qy, qz, r0[i], DENSITY, det_T)
form_factors += jittered_formfactor(V1[i], V2[i], V3[i], qx, qy, qz, r0[i], DENSITY, det_T)
```

#### Important Note
This method introduces a form of numerical regularization, the perturbed geometry is no longer an exact representation of the original polyhedral shape. However, the error appears negligible in practice based on comparison of 2D scattering spectra before and after numerical regularization.