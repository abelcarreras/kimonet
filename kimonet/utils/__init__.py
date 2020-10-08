import numpy as np
from kimonet.utils.rotation import rotate_vector


def minimum_distance_vector(r_vector, supercell):
    # lattice periodicity
    r_vector = np.array(r_vector, dtype=float).copy()
    cell_vector = []

    for lattice in supercell:
        half_cell = np.array(lattice) / 2
        dot_ref = np.dot(half_cell, half_cell)
        dot = np.dot(half_cell, r_vector)
        n_n = (np.sqrt(np.abs(dot)) // np.sqrt(dot_ref)) * np.sign(dot)
        r_vector += np.array(lattice) * -n_n
        cell_vector.append(-n_n)

    return r_vector, np.array(cell_vector, dtype=int)


def distance_vector_periodic(r, supercell, cell_increment):
    return r + np.dot(cell_increment, supercell)


def distance_between_molecules(molecule1, molecule2, supercell, cell_state1, cell_state2):
    r_vector = molecule2.get_coordinates() - molecule1.get_coordinates()
    cell_incr = np.array(cell_state2 - cell_state1)
    return np.linalg.norm(distance_vector_periodic(r_vector, supercell, cell_incr))
