import numpy as np
import itertools
from kimonet.system import System


def regular_ordered_system(conditions,
                           molecule,
                           lattice=None,
                           orientation=(0, 0, 0)
                           ):

    if lattice is None:
        lattice = {'size': [1], 'parameters': [1.0]}  # default 1D 1 molecule system

    molecules = []                              # list of instances of class molecule
    for subset in itertools.product(*[list(range(n)) for n in lattice['size']]):
        coordinates = np.multiply(subset, lattice['parameters'])

        molecule = molecule.copy()  # copy of the generic instance
        molecule.set_coordinates(coordinates)
        molecule.set_orientation(orientation)
        molecules.append(molecule)

    supercell = np.diag(np.multiply(lattice['size'], lattice['parameters']))

    return System(molecules, conditions, supercell)


def regular_system(conditions,
                   molecule,
                   lattice=None,
                   orientation=None,
                   ):

    if lattice is None:
        lattice = {'size': [1], 'parameters': [1.0]}  # default 1D 1 molecule system

    molecules = []  # list of instances of class molecule
    for subset in itertools.product(*[list(range(n)) for n in lattice['size']]):
        coordinates = np.multiply(subset, lattice['parameters'])

        if orientation is None:
            orientation = np.random.random_sample(3) * 2*np.pi

        molecule = molecule.copy()  # copy of the generic instance
        molecule.set_coordinates(coordinates)
        molecule.set_orientation(orientation)
        molecules.append(molecule)

    supercell = np.diag(np.multiply(lattice['size'], lattice['parameters']))

    return System(molecules, conditions, supercell)


def crystal_system(conditions,
                   molecule,
                   scaled_coordinates,
                   dimensions=None,
                   unitcell=None,
                   orientations = None,
                   ):

    unitcell = np.array(unitcell)
    scaled_coordinates = np.array(scaled_coordinates)
    n_mol, n_dim = scaled_coordinates.shape

    if dimensions is None:
        dimensions = [1] * n_dim

    if orientations is None:
        orientations = [None for _ in range(n_mol)]

    molecules = []                              # list of instances of class molecule

    for coordinate, orientation in zip(scaled_coordinates, orientations):
        for subset in itertools.product(*[list(range(n)) for n in dimensions]):

            r_cell = np.sum([s * lattice_vector for s, lattice_vector in zip(subset, unitcell)], axis=0)
            coor = r_cell + np.dot(unitcell.T, coordinate)

            molecule = molecule.copy()  # copy of the generic instance
            molecule.set_coordinates(coor)

            if orientation is None:
                final_orientation = np.random.random_sample(3) * 2 * np.pi
            else:
                final_orientation = orientation

            molecule.set_orientation(final_orientation)
            molecules.append(molecule)

    supercell = np.dot(unitcell, np.diag(dimensions))

    return System(molecules, conditions, supercell)
