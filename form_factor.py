import numpy as np
import time

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