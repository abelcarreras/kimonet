import numpy as np
import itertools
from kimonet.system import System


def regular_ordered_system(molecule,
                           lattice=None,
                           orientation=None,
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

    return System(molecules, supercell)


def regular_system(molecule,
                   lattice=None,
                   orientation=None,
                   ):

    if lattice is None:
        lattice = {'size': [1], 'parameters': [1.0]}  # default 1D 1 molecule system

    molecules = []  # list of instances of class molecule
    for subset in itertools.product(*[list(range(n)) for n in lattice['size']]):
        coordinates = np.multiply(subset, lattice['parameters'])

        final_orientation = list(orientation)
        for j in range(3):
            if final_orientation[j] is None:
                final_orientation[j] = np.random.random_sample() * 2 * np.pi

        molecule = molecule.copy()  # copy of the generic instance
        molecule.set_coordinates(coordinates)
        molecule.set_orientation(final_orientation)
        molecules.append(molecule)

    supercell = np.diag(np.multiply(lattice['size'], lattice['parameters']))

    return System(molecules, supercell)


def crystal_system(molecules,
                   scaled_site_coordinates,
                   dimensions=None,
                   unitcell=None,
                   orientations=None,
                   ):

    unitcell = np.array(unitcell)
    scaled_site_coordinates = np.array(scaled_site_coordinates)
    n_mol, n_dim = scaled_site_coordinates.shape

    if dimensions is None:
        dimensions = [1] * n_dim

    if orientations is None:
        orientations = [None for _ in range(n_mol)]

    molecules_list = []                              # list of instances of class molecule

    for i, (coordinate, molecule_type) in enumerate(zip(scaled_site_coordinates, molecules)):
        for subset in itertools.product(*[list(range(n)) for n in dimensions]):

            r_cell = np.sum([s * lattice_vector for s, lattice_vector in zip(subset, unitcell)], axis=0)
            coor = r_cell + np.dot(unitcell.T, coordinate)

            molecule = molecule_type.copy()  # copy of the generic instance
            molecule.set_coordinates(coor)
            molecule.name = 'a{}'.format(i+1)

            final_orientation = list(orientations[i])
            for j in range(3):
                if final_orientation[j] is None:
                    final_orientation[j] = np.random.random_sample() * 2 * np.pi
                else:
                    final_orientation[j] = final_orientation[j]

            molecule.set_orientation(final_orientation)
            molecules_list.append(molecule)

    supercell = np.dot(unitcell.T, np.diag(dimensions)).T

    return System(molecules_list, supercell)
