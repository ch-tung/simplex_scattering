import numpy as np
import time

# ------------------------------------------------------------------------------
# Part 1: Analytic Scattering Form Factor for Orthogonal Tetrahedra
# Based on: Tianjuan Yang et al., J. Appl. Cryst. 55, 1432–1440 (2022)
# DOI: https://doi.org/10.1107/S160057672201130X
#
# This implementation computes the scattering amplitude F(q) of a tetrahedron
# defined by three orthogonal edge vectors (V1, V2, V3) using a closed-form
# analytical expression. The model assumes piecewise constant scattering length
# density and a sharp polyhedral boundary.
# ------------------------------------------------------------------------------

def orthogonal_tetrahedra_formfactor(V1, V2, V3, qx, qy, qz, DENSITY):
    """
    Calculate the form factor (scattering amplitude) of orthogonal tetrahedra.
    https://doi.org/10.1107/S160057672201130X
    
    Parameters:
    V1, V2, V3 (numpy.ndarray): Vertices of the tetrahedron.
    qx, qy, qz (numpy.ndarray): Components of the wave vector.
    DENSITY (float): Density of the material.
    
    Returns:
    numpy.ndarray: Scattering amplitude.
    
    Originaal MATLAB code description:
    % Calculate the form factor (scattering amplitude) of orthogonal tetrahedra.
    % Code written by Tianjuan Yang.
    """
    T = np.array([V1, V2, V3]).T
    Q1 = qx * V1[0] + qy * V1[1] + qz * V1[2]
    Q2 = qx * V2[0] + qy * V2[1] + qz * V2[2]
    Q3 = qx * V3[0] + qy * V3[1] + qz * V3[2]
    det_T = np.linalg.det(T)
    
    scattering_amplitude = DENSITY * det_T * (1j * np.exp(1j * Q3) / (Q3 * (Q3 - Q2) * (Q3 - Q1)) +
                                              1j * np.exp(1j * Q2) / (Q2 * (Q2 - Q1) * (Q2 - Q3)) +
                                              1j * np.exp(1j * Q1) / (Q1 * (Q1 - Q2) * (Q1 - Q3)) -
                                              1j / (Q1 * Q2 * Q3))
    
    # Calculation at singularities
    row2 = np.where((np.abs(Q1) <= 1e-9) & (np.abs(Q2) >= 1e-9) & (np.abs(Q3) >= 1e-9) & (np.abs(Q2 - Q3) >= 1e-9))
    scattering_amplitude[row2] = DENSITY * det_T * (1j * np.exp(1j * Q2[row2]) / (Q2[row2]**2 * (Q2[row2] - Q3[row2])) +
                                                    1j * np.exp(1j * Q3[row2]) / (Q3[row2]**2 * (Q3[row2] - Q2[row2])) +
                                                    1j * (Q2[row2] + Q3[row2] + 1j * Q2[row2] * Q3[row2]) / (Q2[row2]**2 * Q3[row2]**2))
    
    row3 = np.where((np.abs(Q2) <= 1e-9) & (np.abs(Q1) >= 1e-9) & (np.abs(Q3) >= 1e-9) & (np.abs(Q1 - Q3) >= 1e-9))
    scattering_amplitude[row3] = DENSITY * det_T * (1j * np.exp(1j * Q1[row3]) / (Q1[row3]**2 * (Q1[row3] - Q3[row3])) +
                                                    1j * np.exp(1j * Q3[row3]) / (Q3[row3]**2 * (Q3[row3] - Q1[row3])) +
                                                    1j * (Q1[row3] + Q3[row3] + 1j * Q1[row3] * Q3[row3]) / (Q1[row3]**2 * Q3[row3]**2))
    
    row4 = np.where((np.abs(Q3) <= 1e-9) & (np.abs(Q1) >= 1e-9) & (np.abs(Q2) >= 1e-9) & (np.abs(Q1 - Q2) >= 1e-9))
    scattering_amplitude[row4] = DENSITY * det_T * (1j * np.exp(1j * Q1[row4]) / (Q1[row4]**2 * (Q1[row4] - Q2[row4])) +
                                                    1j * np.exp(1j * Q2[row4]) / (Q2[row4]**2 * (Q2[row4] - Q1[row4])) +
                                                    1j * (Q1[row4] + Q2[row4] + 1j * Q1[row4] * Q2[row4]) / (Q1[row4]**2 * Q2[row4]**2))
    
    row5 = np.where((np.abs(Q1 - Q3) < 1e-9) & (np.abs(Q1) >= 1e-9) & (np.abs(Q1 - Q2) >= 1e-9) & (np.abs(Q3) >= 1e-9))
    scattering_amplitude[row5] = DENSITY * det_T * (1j * np.exp(1j * Q2[row5]) / (Q2[row5] * (Q2[row5] - Q1[row5])**2) +
                                                    np.exp(1j * Q1[row5]) * (1j * Q2[row5] - 2j * Q1[row5] - Q1[row5]**2 + Q1[row5] * Q2[row5]) / ((Q2[row5] - Q1[row5])**2 * Q1[row5]**2) -
                                                    1j / (Q1[row5]**2 * Q2[row5]))
    
    row6 = np.where((np.abs(Q1 - Q2) < 1e-9) & (np.abs(Q1) >= 1e-9) & (np.abs(Q2 - Q3) >= 1e-9) & (np.abs(Q2) >= 1e-9))
    scattering_amplitude[row6] = DENSITY * det_T * (1j * np.exp(1j * Q3[row6]) / (Q3[row6] * (Q3[row6] - Q1[row6])**2) +
                                                    np.exp(1j * Q1[row6]) * (1j * Q3[row6] - 2j * Q1[row6] - Q1[row6]**2 + Q1[row6] * Q3[row6]) / ((Q3[row6] - Q1[row6])**2 * Q1[row6]**2) -
                                                    1j / (Q1[row6]**2 * Q3[row6]))
    
    row7 = np.where((np.abs(Q2 - Q3) < 1e-9) & (np.abs(Q2) >= 1e-9) & (np.abs(Q1 - Q3) >= 1e-9) & (np.abs(Q3) >= 1e-9))
    scattering_amplitude[row7] = DENSITY * det_T * (1j * np.exp(1j * Q1[row7]) / (Q1[row7] * (Q1[row7] - Q2[row7])**2) +
                                                    np.exp(1j * Q2[row7]) * (1j * Q1[row7] - 2j * Q2[row7] - Q2[row7]**2 + Q2[row7] * Q1[row7]) / ((Q1[row7] - Q2[row7])**2 * Q2[row7]**2) -
                                                    1j / (Q2[row7]**2 * Q1[row7]))
    
    row8 = np.where((np.abs(Q1 - Q2) <= 1e-5) & (np.abs(Q1 - Q3) <= 1e-5) & (np.abs(Q2 - Q3) <= 1e-5) & (np.abs(Q1) >= 1e-9) & (np.abs(Q2) >= 1e-9) & (np.abs(Q3) >= 1e-9))
    scattering_amplitude[row8] = DENSITY * det_T * (np.exp(1j * Q1[row8]) * (2j + 2 * Q1[row8] - 1j * Q1[row8]**2) / (2 * Q1[row8]**3) - 1j / Q1[row8]**3)
    
    row9 = np.where(Q1**2 + Q2**2 + Q3**2 < 1e-3)
    scattering_amplitude[row9] = DENSITY * det_T / 6 + 1j * det_T * (Q1[row9] + Q2[row9] + Q3[row9])
    
    return scattering_amplitude

def scattering_function_tetrahedron(verts, tetrahedra, q_grid):
    """
    Evaluate the scattering function of the whole mesh over a grid of wave vectors for tetrahedra.
    
    Parameters:
    vertices (numpy.ndarray): Array of shape (n, 3) containing the vertices of the mesh.
    tetrahedra (numpy.ndarray): Array of shape (m, 4) containing the indices of the vertices forming each tetrahedron.
    q_grid (list): List of 3 numpy arrays representing the grid of wave vectors.
    
    Returns:
    numpy.ndarray: The scattering function values over the grid.
    """
    print("Flattening q_grid...")
    start_time = time.time()
    q_grid_flat = np.array([q_grid[0].flatten(), q_grid[1].flatten(), q_grid[2].flatten()])
    print(f"Time taken: {time.time() - start_time} seconds")
    
    print("Calculating tetrahedron volumes...")
    start_time = time.time()
    r0 = verts[tetrahedra[:, 0]]
    r1 = verts[tetrahedra[:, 1]]
    r2 = verts[tetrahedra[:, 2]]
    r3 = verts[tetrahedra[:, 3]]
    print(f"Time taken: {time.time() - start_time} seconds")
    
    print("Calculating orthogonal tetrahedra form factor...")
    start_time = time.time()
    V1 = r1 - r0
    V2 = r2 - r0
    V3 = r3 - r0
    qx, qy, qz = q_grid_flat
    
    form_factors = np.zeros((tetrahedra.shape[0], q_grid_flat.shape[1]), dtype=complex)
    for i in range(tetrahedra.shape[0]):
        form_factors[i] = orthogonal_tetrahedra_formfactor(V1[i], V2[i], V3[i], qx, qy, qz, 1) * np.exp(1j * (qx * r0[i, 0] + qy * r0[i, 1] + qz * r0[i, 2]))
    print(f"Time taken: {time.time() - start_time} seconds")
    
    F_q = np.sum(form_factors, axis=0)
    
    print("Calculating scattering function I_q...")
    start_time = time.time()
    I_q = np.abs(F_q) ** 2
    print(f"Time taken: {time.time() - start_time} seconds")
    
    print("Scattering function calculation complete.")
    return I_q

# ------------------------------------------------------------------------------
# Part 2: Adaptive Subdivision Strategy for Robust Scattering Calculation
#
# This section augments the analytic method by introducing an adaptive subdivision
# strategy to mitigate numerical singularities. When a tetrahedron exhibits a
# potentially unstable projection configuration (e.g., q·V1 ≈ q·V2), it is
# subdivided into smaller tetrahedra and recomputed recursively.
#
# Method:
# - Detect ill-defined configurations by checking small Q_i or Q_i - Q_j.
# - If singular and recursion depth < max_depth, subdivide tetrahedron into 8 parts.
# - Perturb edge midpoints with random jitter to avoid maintaining coplanar edges.
# - Recursively apply the same analytic method to all sub-tetrahedra.
#
# Benefits:
# - Stabilizes the scattering amplitude evaluation across q-space.
# - Ensures convergence near singularities while preserving physical consistency.
#
# Tradeoffs:
# - Increased computational cost due to recursive evaluation.
# ------------------------------------------------------------------------------
def subdivide_tetrahedron(v0, v1, v2, v3, jitter=1e-3):
    """
    Subdivide a tetrahedron into 8 smaller tetrahedra.
    Midpoints of edges are slightly perturbed to avoid singularities caused
    by degenerate configurations (e.g. coplanarity or collinearity in projection).

    Parameters:
    - v0, v1, v2, v3 : np.ndarray (3,)
        Vertex coordinates of the original tetrahedron.
    - jitter : float
        Magnitude of random perturbation added to midpoint to break symmetry.

    Returns:
    - List of 8 new tetrahedra (each as list of 4 vertices).
    """

    def perturb(a, b):
        midpoint = 0.5 * (a + b)
        noise = np.random.normal(scale=jitter, size=3)
        return midpoint + noise  # break symmetry to reduce projection degeneracy

    # Perturbed midpoints of all edges
    m01 = perturb(v0, v1)
    m02 = perturb(v0, v2)
    m03 = perturb(v0, v3)
    m12 = perturb(v1, v2)
    m13 = perturb(v1, v3)
    m23 = perturb(v2, v3)

    # Return 8 sub-tetrahedra created from the original geometry
    return [
        [v0, m01, m02, m03],
        [m01, v1, m12, m13],
        [m02, m12, v2, m23],
        [m03, m13, m23, v3],
        [m01, m02, m03, m13],
        [m01, m12, m13, m02],
        [m02, m12, m23, m13],
        [m02, m13, m03, m23],
    ]

def adaptive_formfactor(V1, V2, V3, qx, qy, qz, r0, DENSITY, det_T, depth=0, max_depth=2):
    """
    Recursively compute the form factor of a tetrahedron.
    If singularities are detected in the analytic expression, the tetrahedron is subdivided.

    Parameters:
    - V1, V2, V3 : np.ndarray (3,)
        Edge vectors from r0 defining the tetrahedron.
    - qx, qy, qz : np.ndarray
        Components of the q-vector grid.
    - r0 : np.ndarray (3,)
        Base vertex of the tetrahedron.
    - DENSITY : float
        Scattering length density contrast.
    - det_T : float
        Determinant of the edge matrix (volume factor).
    - depth : int
        Current recursion depth.
    - max_depth : int
        Maximum allowed subdivision depth to prevent infinite recursion.

    Returns:
    - Form factor (complex) over the provided q grid.
    """

    # Project q-vector onto tetrahedron edges
    Q1 = qx * V1[0] + qy * V1[1] + qz * V1[2]
    Q2 = qx * V2[0] + qy * V2[1] + qz * V2[2]
    Q3 = qx * V3[0] + qy * V3[1] + qz * V3[2]

    # Detect if denominator of analytic expression is ill-defined
    eps = 1e-8
    is_singular = np.any(np.abs(Q1) < eps) or np.any(np.abs(Q2) < eps) or np.any(np.abs(Q3) < eps) or \
                  np.any(np.abs(Q1 - Q2) < eps) or np.any(np.abs(Q1 - Q3) < eps) or np.any(np.abs(Q2 - Q3) < eps)

    if is_singular and depth < max_depth:
        # Reconstruct full vertices
        v0 = r0
        v1 = r0 + V1
        v2 = r0 + V2
        v3 = r0 + V3

        # Subdivide tetrahedron with jitter to avoid repeated singularities
        sub_tets = subdivide_tetrahedron(v0, v1, v2, v3)

        # Recursively evaluate all sub-tetrahedra
        F_sub = 0
        for tet in sub_tets:
            r0_sub = tet[0]
            V1_sub = tet[1] - r0_sub
            V2_sub = tet[2] - r0_sub
            V3_sub = tet[3] - r0_sub
            T_sub = np.array([V1_sub, V2_sub, V3_sub]).T
            det_T_sub = np.abs(np.linalg.det(T_sub))  # safe volume
            F_sub += adaptive_formfactor(V1_sub, V2_sub, V3_sub, qx, qy, qz, r0_sub, DENSITY, det_T_sub, depth + 1, max_depth)
        return F_sub

    else:
        # Safe evaluation of the analytic form factor expression
        phase = np.exp(1j * (qx * r0[0] + qy * r0[1] + qz * r0[2]))
        return DENSITY * det_T * (
            1j * np.exp(1j * Q3) / (Q3 * (Q3 - Q2) * (Q3 - Q1)) +
            1j * np.exp(1j * Q2) / (Q2 * (Q2 - Q1) * (Q2 - Q3)) +
            1j * np.exp(1j * Q1) / (Q1 * (Q1 - Q2) * (Q1 - Q3)) -
            1j / (Q1 * Q2 * Q3)
        ) * phase
        
def jittered_formfactor(V1, V2, V3, qx, qy, qz, r0, DENSITY, det_T, jitter=1e-3):
    """
    Evaluate the scattering form factor of a tetrahedron using analytic expression.
    If singularities are detected, jitter all four vertices once to regularize.

    Parameters:
    - V1, V2, V3: (3,) np.ndarray
        Edge vectors from base vertex r0 defining the tetrahedron
    - qx, qy, qz: (N,) flattened np.ndarrays
        Wave vector components
    - r0: (3,) np.ndarray
        Base vertex position
    - DENSITY: float
        Scattering length density contrast
    - det_T: float
        Volume factor (determinant of edge matrix)
    - jitter: float
        Standard deviation of the perturbation added to each vertex if needed

    Returns:
    - F_q: complex ndarray
        Scattering amplitude evaluated at all q points
    """
    # Project q onto edge vectors
    Q1 = qx * V1[0] + qy * V1[1] + qz * V1[2]
    Q2 = qx * V2[0] + qy * V2[1] + qz * V2[2]
    Q3 = qx * V3[0] + qy * V3[1] + qz * V3[2]

    eps = 1e-8
    is_singular = (
        np.any(np.abs(Q1) < eps) or
        np.any(np.abs(Q2) < eps) or
        np.any(np.abs(Q3) < eps) or
        np.any(np.abs(Q1 - Q2) < eps) or
        np.any(np.abs(Q1 - Q3) < eps) or
        np.any(np.abs(Q2 - Q3) < eps)
    )

    if is_singular:
        # Add random jitter to all vertices
        noise = lambda: np.random.normal(scale=jitter, size=3)
        r0_j = r0 + noise()
        r1_j = r0 + V1 + noise()
        r2_j = r0 + V2 + noise()
        r3_j = r0 + V3 + noise()

        V1_j = r1_j - r0_j
        V2_j = r2_j - r0_j
        V3_j = r3_j - r0_j
        T_j = np.array([V1_j, V2_j, V3_j]).T
        det_T_j = np.abs(np.linalg.det(T_j))

        Q1 = qx * V1_j[0] + qy * V1_j[1] + qz * V1_j[2]
        Q2 = qx * V2_j[0] + qy * V2_j[1] + qz * V2_j[2]
        Q3 = qx * V3_j[0] + qy * V3_j[1] + qz * V3_j[2]
        phase = np.exp(1j * (qx * r0_j[0] + qy * r0_j[1] + qz * r0_j[2]))

        return DENSITY * det_T_j * (
            1j * np.exp(1j * Q3) / (Q3 * (Q3 - Q2) * (Q3 - Q1)) +
            1j * np.exp(1j * Q2) / (Q2 * (Q2 - Q1) * (Q2 - Q3)) +
            1j * np.exp(1j * Q1) / (Q1 * (Q1 - Q2) * (Q1 - Q3)) -
            1j / (Q1 * Q2 * Q3)
        ) * phase

    else:
        phase = np.exp(1j * (qx * r0[0] + qy * r0[1] + qz * r0[2]))
        return DENSITY * det_T * (
            1j * np.exp(1j * Q3) / (Q3 * (Q3 - Q2) * (Q3 - Q1)) +
            1j * np.exp(1j * Q2) / (Q2 * (Q2 - Q1) * (Q2 - Q3)) +
            1j * np.exp(1j * Q1) / (Q1 * (Q1 - Q2) * (Q1 - Q3)) -
            1j / (Q1 * Q2 * Q3)
        ) * phase


def scattering_function_adaptive(verts, tetrahedra, q_grid, DENSITY=1.0):
    """
    Compute the scattering function over a q-vector grid using an adaptive tetrahedral mesh.
    Singularities are handled by local tetrahedron subdivision.

    Parameters:
    - verts : (N, 3) ndarray
        Vertex coordinates of the full 3D shape.
    - tetrahedra : (M, 4) ndarray
        Index list of tetrahedra.
    - q_grid : list of 3 (X, Y, Z) arrays
        Meshgrid of q-vector components.
    - DENSITY : float
        Scattering length density contrast.

    Returns:
    - I_q : ndarray
        Scattering intensity array of shape matching q_grid.
    """
    q_grid_flat = np.array([q_grid[0].flatten(), q_grid[1].flatten(), q_grid[2].flatten()])
    qx, qy, qz = q_grid_flat

    # Decompose tetrahedra
    r0 = verts[tetrahedra[:, 0]]
    r1 = verts[tetrahedra[:, 1]]
    r2 = verts[tetrahedra[:, 2]]
    r3 = verts[tetrahedra[:, 3]]
    V1 = r1 - r0
    V2 = r2 - r0
    V3 = r3 - r0

    # Accumulate form factor
    form_factors = np.zeros(qx.shape[0], dtype=complex)
    for i in range(tetrahedra.shape[0]):
        T = np.array([V1[i], V2[i], V3[i]]).T
        det_T = np.abs(np.linalg.det(T))
        form_factors += jittered_formfactor(V1[i], V2[i], V3[i], qx, qy, qz, r0[i], DENSITY, det_T)

    return np.abs(form_factors) ** 2