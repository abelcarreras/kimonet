import numpy as np
import itertools
from kimonet.system import System


def ordered_system(conditions,
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


def disordered_system(conditions,
                      molecule,
                      lattice=None
                      ):

    if lattice is None:
        lattice = {'size': [1], 'parameters': [1.0]}  # default 1D 1 molecule system

    molecules = []  # list of instances of class molecule
    for subset in itertools.product(*[list(range(n)) for n in lattice['size']]):
        coordinates = np.multiply(subset, lattice['parameters'])
        orientation = np.random.random_sample(3) * 2*np.pi
        molecule = molecule.copy()  # copy of the generic instance
        molecule.set_coordinates(coordinates)
        molecule.set_orientation(orientation)
        molecules.append(molecule)

    supercell = np.diag(np.multiply(lattice['size'], lattice['parameters']))

    return System(molecules, conditions, supercell)
