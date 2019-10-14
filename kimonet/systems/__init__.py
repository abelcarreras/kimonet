import numpy as np
from numpy import pi
from scipy.spatial import distance
from kimonet.systems.excitation import excited_system, excite_system
import copy
import itertools
import warnings

#######################################################################################################################


def get_system(conditions,
               molecule,
               lattice={'dimensions': [1000],
                        'lattice_parameter': 1.0},
               amorphous=None,
               orientation='parallel',
               initial_excitation={'s1': ['centre']}):
    """
    This is an intermediate function that takes the flux of the program to other functions.
    The role of this function is to check whether the desired system has to be ordered or disodrdered
    """

    if amorphous is None:
        # when amorphous is None, lattice must be given, then an ordered system will be initialized. We use a different
        # function depending on the dimensionality. All 3 are collected in a dictionary: get_ordered_system. The keys
        # are the dimensionality.

        dimensionality = str(len(lattice['dimensions']))
        return get_ordered_system[dimensionality](conditions, molecule, lattice, orientation, initial_excitation)

    if lattice is None:
        # when lattice is None, amorphous must be given, then a disordered system will be initialized.

        return get_disordered_system(conditions, molecule, amorphous, orientation, initial_excitation)


#######################################################################################################################


def get_disordered_system(conditions,
                          generic_molecule,
                          amorphous={'dimensions': [25, 25],            # nm
                                  'num_molecules': 1000},
                          orientation='parallel',
                          initial_excitation={'s1': ['centre']}, ):

    """
    :param conditions: A dictionary with the physical conditions of the problem such as temperature.
    :param generic_molecule: generic instance of class Molecule with the internal information defined.
        In the function a position and orientation for the molecule are given
    :param amorphous: dictionary with the amorphous morphology information:
        dimensions: maximum physical dimensions of the system
        num_molecules: number of molecules
    :param orientation: String parameter. Indicates whether the orientation of the molecules is parallel,
    random or anti parallel (each molecule finds its 1sts neighbours anti parallely orientated)
    :param initial_excitation: Dictionary with the information of the excitons (type and a position for each).
    :return: A dictionary with a list of molecules and updated dictionary with the physical conditions.
    """

    physical_dimensions = amorphous['dimensions']
    number_molecules = amorphous['num_molecules']

    molecules_side = np.array(physical_dimensions) / generic_molecule.characteristic_length
    capacity = get_capacity(molecules_side)
#   computes the number of molecules per side from the physical dimensions of the system and the molecular
#   characteristic  length. With this information calculates the capacity (maximum number of molecules) of the system

    if orientation is 'antiparallel':           # anti parallelism is not defined for an amorphous system
        print('Not anti parallel orientation considered for an amorphous material')
        return None

    molecules = []                  # list of instances of class molecule
    molecule_count = 0
    while molecule_count <= number_molecules:

        too_close = True
        while too_close is True:                # ensures that two molecules do not overlap

            coordinates = []
            for physical_dimension in physical_dimensions:
                x = physical_dimensions*np.random.random() - physical_dimension/2
#               Picks a random coordinate value for each dimension. We want the distribution centered at (0,0,0)

                coordinates.append(x)

            too_close = distance_checking(coordinates, molecules)       # ensures that two molecules do not overlap

        orientation_vector = get_orientation(orientation, generic_molecule, len(physical_dimensions), pointing=1)
#       Defines the molecular orientation vector. generic_molecule has information about a reference orientation.

        molecule = copy.deepcopy(generic_molecule)              # copy of the generic instance
        molecule.set_coordinates(coordinates)            # initialization of the position
        molecule.set_orientation(orientation_vector)     # initialization of the orientation

        molecules.append(molecule)
        molecule_count += 1

        if molecule_count == capacity:                          # stops if the maximum number of molecules is reached.
            break

    centre_indexes = excited_system(molecules, initial_excitation, tolerance=0.05)
    # molecules list is modified with the excitation defined in initial_excitation.
    # returns a list with the indexes of the excited molecules (efficiency trick).

    conditions['amorphous'] = amorphous

    # definition of system as a dictionary with three main arguments:
    #       molecules: list with all class molecule instances
    #       condition: dictionary with the physical conditions
    #       amorphous: dictionary with the morphology information
    # and two arguments used as efficiency tricks:
    #       centres: list with the indexes of the excited molecules
    #       type type of the system
    system = {'molecules': molecules, 'conditions': conditions, 'amorphous': amorphous,
              'centres': centre_indexes, 'type': 'amorphous'}

    return system


#######################################################################################################################


def get_1d_ordered_system(conditions,
                          generic_molecule,
                          lattice={'dimensions': [1000],          # molecules / side
                                     'lattice_parameter': [1.0]},
                          orientation='parallel',
                          initial_excitation={'s1': ['centre']}, ):
    """
    :param conditions: A dictionary with the physical conditions of the problem such as temperature.
    :param generic_molecule: generic instance of class Molecule with the intern information defined.
        In the function a position and orientation for the molecule are given
    :param lattice: dictionary with the morphology parameter:
            dimensions: number of molecules per side
            lattice parameter: list with the lattice parameter for each dimension
    :param orientation: String parameter. Indicates whether the orientation of the molecules is parallel,
    random or anti parallel (each molecule finds its 1sts neighbours anti parallely orientated)
    :param initial_excitation: Dictionary with the information of the excitons (type and a position for each).
    :return: A dictionary with a list of molecules and updated dictionary with the physical conditions.
    """

    dimensions = lattice['dimensions']                      # super cell of the crystal (molecules / side)
    lattice_parameter = lattice['lattice_parameter']        # list with the lattice parameter for each direction

    physical_dimensions = np.array(dimensions) * np.array(lattice_parameter)        # physical dimensions of the finite crystal (nm)

    # ensures that none of the lattice parameters given is shorter than the molecular characteristic length
    if check_lattice(lattice_parameter, generic_molecule) is False:
        print('Lattice parameter is smaller than molecular characteristic length')
        return

    molecules = []                          # list of instances of class molecule

    pointing = 1                            # pointing and symmetry are assistant parameters used in the generation of
    symmetry = get_symmetry[orientation]    # the molecular orientation when anti parallel or parallel orientation is given

    x_max = physical_dimensions[0]/2
    for x in np.arange(-x_max, x_max, lattice_parameter[0]):
        coordinates = [x, 0, 0]
        # x values are taken between -x_max and x_max in order to have the distribution centered at 0.

        orientation_vector = get_orientation(orientation, generic_molecule,
                                             len(dimensions), pointing)
        # Defines the molecular orientation vector. generic_molecule has information about a reference orientation.

        molecule = copy.deepcopy(generic_molecule)                 # copy of the generic instance
        molecule.set_coordinates(coordinates)               # initialization of the position
        molecule.set_orientation(orientation_vector)        # initialization of the orientation

        molecules.append(molecule)

        pointing = pointing * symmetry[0]
        # see Anti parallelism in Docs to a further knowledge of how anti parallelism is defined.

    centre_indexes = excited_system(molecules, initial_excitation, tolerance=np.sum(np.array(lattice_parameter))/2)
    # molecules list is modified with the excitation defined in initial_excitation.
    # returns a list with the indexes of the excited molecules (efficiency trick).

    conditions['lattice'] = lattice

    # definition of system as a dictionary with three main arguments:
    #       molecules: list with all class molecule instances
    #       condition: dictionary with the physical conditions
    #       lattice: morphology information
    # and two arguments used as efficiency tricks:
    #       centres: list with the indexes of the excited molecules
    #       type type of the system
    system = {'molecules': molecules, 'conditions': conditions, 'lattice': lattice,
              'centres': centre_indexes, 'type': '1d_ordered'}

    return system


#######################################################################################################################


def get_2d_ordered_system(conditions,
                          generic_molecule,
                          lattice={'dimensions': [100, 100],               # molecules / side
                                     'lattice_parameter': [1.0, 1.0]},
                          orientation='parallel',
                          initial_excitation={'s1': ['centre']}):
    """
    :param conditions: A dictionary with the physical conditions of the problem such as temperature.
    :param generic_molecule: generic instance of class Molecule with the intern information defined.
        In the function a position and orientation for the molecule are given
    :param lattice: dictionary with the morphology parameters:
            dimensions: number of molecules per side
            lattice parameter: list with the lattice parameter for each dimension
    :param orientation: String parameter. Indicates whether the orientation of the molecules is parallel,
    random or anti parallel (each molecule finds its 1sts neighbours anti parallely orientated)
    :param initial_excitation: Dictionary with the information of the excitons (type and a position for each).
    :return: A dictionary with a list of molecules and updated dictionary with the physical conditions.
    """

    dimensions = lattice['dimensions']                              # super cell of the crystal (molecules / side)
    lattice_parameter = lattice['lattice_parameter']                # list with the lattice parameter for each direction

    physical_dimensions = np.array(dimensions) * np.array(lattice_parameter)    # physical dimensions of the finite crystal (nm)

    # ensures that none of the lattice parameters given is shorter than the molecular characteristic length
    if check_lattice(lattice_parameter, generic_molecule) is False:
        print('Lattice parameter is smaller than molecular characteristic length')
        return

    molecules = []                              # list of instances of class molecule

    symmetry = get_symmetry[orientation]
    x_count = 0
    # symmetry and x_count are assistant parameters used in the generation of  the molecular orientation when
    # anti parallel or parallel orientation is given

    x_max = physical_dimensions[0]/2
    y_max = physical_dimensions[1]/2
    # (x,y) values are taken between -x,y_max and x,y_max in order to have the distribution centered at (0,0).

    for x in np.arange(-x_max, x_max, lattice_parameter[0]):

        pointing = (-1)**x_count                # assistant parameter for anti parallelism

        for y in np.arange(-y_max, y_max, lattice_parameter[1]):
            coordinates = [x, y, 0]

            orientation_vector = get_orientation(orientation, generic_molecule, len(dimensions), pointing)
            # Defines the molecular orientation vector. generic_molecule has information about a reference orientation.

            molecule = copy.deepcopy(generic_molecule)              # copy of the generic instance
            molecule.set_coordinates(coordinates)            # initialization of the position
            molecule.set_orientation(orientation_vector)     # initialization of the orientation

            molecules.append(molecule)

            pointing = pointing * symmetry[0]
        x_count = x_count + symmetry[1]
        # see Anti parallelism in Docs to a further knowledge of how anti parallelism is defined.

    centre_indexes = excited_system(molecules, initial_excitation, tolerance=np.sum(np.array(lattice_parameter))/4)
    # molecules list is modified with the excitation defined in initial_excitation.
    # returns a list with the indexes of the excited molecules (efficiency trick).

    conditions['lattice'] = lattice

    # definition of system as a dictionary with three main arguments:
    #       molecules: list with all class molecule instances
    #       condition: dictionary with the physical conditions
    #       lattice: morphology information
    # and two arguments used as efficiency tricks:
    #       centres: list with the indexes of the excited molecules
    #       type type of the system
    system = {'molecules': molecules, 'conditions': conditions, 'lattice': lattice,
              'centres': centre_indexes, 'type': '2d_ordered'}

    return system


#######################################################################################################################


def get_3d_ordered_system(conditions,
                          generic_molecule,
                          lattice={'dimensions': [10, 10, 10],               # molecules / side
                                   'lattice_parameter': [1.0, 1.0, 1.0]},
                          orientation='parallel',
                          initial_excitation={'s1': ['centre']}):
    """
    :param conditions: A dictionary with the physical conditions of the problem such as temperature.
    :param generic_molecule: generic instance of class Molecule with the intern information defined.
        In the function a position and orientation for the molecule are given
    :param lattice: dictionary with the morphology parameters:
            dimensions: number of molecules per side
            lattice parameter: list with the lattice parameter for each dimension
    :param orientation: String parameter. Indicates whether the orientation of the molecules is parallel,
    random or anti parallel (each molecule finds its 1sts neighbours anti parallely orientated)
    :param initial_excitation: Dictionary with the information of the excitons (type and a position for each).
    :return: A dictionary with a list of molecules and updated dictionary with the physical conditions.
    """

    dimensions = lattice['dimensions']                              # super cell of the crystal (molecules / side)
    lattice_parameter = lattice['lattice_parameter']                # list with the lattice parameter for each direction

    physical_dimensions = np.array(dimensions) * np.array(lattice_parameter)    # physical dimensions of the finite crystal (nm)

    # ensures that none of the lattice parameters given is shorter than the molecular characteristic length
    if check_lattice(lattice_parameter, generic_molecule) is False:
        print('Lattice parameter is smaller than molecular characteristic length')
        return

    molecules = []                              # list of instances of class molecule

    x_count = 0
    symmetry = get_symmetry[orientation]
    # symmetry and x_count are assistant parameters used in the generation of  the molecular orientation when
    # anti parallel or parallel orientation is given

    max_coordinate = np.array(physical_dimensions)
    step = lattice_parameter
    # (x,y,z) values are taken between -x,y,z_max and x,y,z_max in order to have the distribution centered at (0,0,0).

    for x in np.arange(-max_coordinate[0], max_coordinate[0], step[0]):

        x_pointing = (-1)**x_count
        y_count = 0

        for y in np.arange(-max_coordinate[1], max_coordinate[1], step[1]):

            pointing = x_pointing * (-1)**y_count

            for z in np.arange(-max_coordinate[2], max_coordinate[2], step[2]):
                coordinates = [x, y, z]

                orientation_vector = get_orientation(orientation, generic_molecule,
                                                     len(dimensions), pointing)
                # Defines the molecular orientation. generic_molecule has information about a reference orientation.

                molecule = copy.deepcopy(generic_molecule)              # copy of the generic instance
                molecule.set_coordinates(coordinates)            # initialization of the position
                molecule.set_orientation(orientation_vector)     # initialization of the orientation

                molecules.append(molecule)

                pointing = pointing*symmetry[0]
            y_count = y_count + symmetry[1]
        x_count = x_count + symmetry[1]
        # see Anti parallelism in Docs to a further knowledge of how anti parallelism is defined.

    centre_indexes = excited_system(molecules, initial_excitation, tolerance=np.sum(np.array(lattice_parameter))/6)
    # molecules list is modified with the excitation defined in initial_excitation.
    # returns a list with the indexes of the excited molecules (efficiency trick).

    conditions['lattice'] = lattice_parameter

    # definition of system as a dictionary with three main arguments:
    #       molecules: list with all class molecule instances
    #       condition: dictionary with the physical conditions
    #       lattice: morphology information
    # and two arguments used as efficiency tricks:
    #       centres: list with the indexes of the excited molecules
    #       type type of the system
    system = {'molecules': molecules, 'conditions': conditions, 'lattice': lattice,
              'centres': centre_indexes, 'type': '3d_ordered'}

    return system


class System:
    def __init__(self,
                 molecules,
                 conditions,
                 supercell):

        self.molecules = molecules
        self.conditions = conditions
        self.supercell = supercell
        self.neighbors = {}

        # search centers
        self.centers = []
        for i, molecule in enumerate(self.molecules):
            if molecule.state != 'gs':
                self.centers.append(i)

    def get_neighbours(self, center):
        radius = self.conditions['cutoff_radius']
        center_position = self.molecules[center].get_coordinates()

        from kimonet.utils import minimum_distance_vector

        if not '{}_{}'.format(center, radius) in self.neighbors:
            neighbours = []
            jumps = []
            for i, molecule in enumerate(self.molecules):
                coordinates = molecule.get_coordinates()
                r_vec, cell_vec = minimum_distance_vector(coordinates - center_position, self.supercell)

                if 0 < np.linalg.norm(r_vec) < radius:
                    neighbours.append(i)
                    jumps.append(cell_vec)
                """
                
                for l_vector in np.array(self.supercell):
                    coord_plus = coordinates + np.array(l_vector)
                    coord_minus = coordinates - np.array(l_vector)

                    if 0 < distance.euclidean(center_position, coordinates) < radius:
                        neighbours.append(i)
                        jumps.append(0)

                    elif 0 < distance.euclidean(center_position, coord_plus) < radius:
                        neighbours.append(i)
                        jumps.append(1)

                    elif 0 < distance.euclidean(center_position, coord_minus) < radius:
                        neighbours.append(i)
                        jumps.append(-1)

                """

            indexes = np.unique(neighbours, return_index=True)[1]
            neighbours = np.array(neighbours)[indexes]
            jumps = np.array(jumps)[indexes]

            self.neighbors['{}_{}'.format(center, radius)] = [neighbours, jumps]

        return self.neighbors['{}_{}'.format(center, radius)]

    def reset(self):
        for molecule in self.molecules:
            molecule.state = 'gs'
        self.centers = []

    def get_num_molecules(self):
        return len(self.molecules)

    def get_data(self):
        return {'molecules': self.molecules,
                'conditions': self.conditions,
                'supercell': self.supercell,
                'centres': self.get_centers()}

    def get_number_of_excitations(self):
        return len(self.centers)

    def add_excitation_index(self, type, index):
        self.molecules[index].state = type
        if type == 'gs':
            try:
                self.centers.remove(index)
            except ValueError:
                pass
        else:
            if not index in self.centers:
                self.centers.append(index)

    def add_excitation_random(self, type, n):
        for i in range(n):
            while True:
                num = np.random.randint(0, self.get_num_molecules())
                if self.molecules[num].state == 'gs':
                    # self.molecules[num] = type
                    self.add_excitation_index(type, num)
                    break

    def add_excitation_center(self, type):
        center_coor = np.diag(self.supercell)/2
        min = np.linalg.norm(self.supercell[0])
        for i, molecule in enumerate(self.molecules):
            dist = distance.euclidean(center_coor, molecule.get_coordinates())
            if dist < min:
                min = dist
                index = i

        self.add_excitation_index(type, index)


def ordered_system(conditions,
                   molecule,
                   lattice={'size': (1),
                            'parameters': (1.0, 1.0, 1.0)},
                   orientation=(0, 0, 0)
                   ):

    # print('size:', lattice['size'])
    # print('parameters:', lattice['parameters'])

    molecules = []                              # list of instances of class molecule
    for subset in itertools.product(*[list(range(n)) for n in lattice['size']]):
        coordinates = np.multiply(subset, lattice['parameters'])

        molecule = molecule.copy()  # copy of the generic instance
        molecule.set_coordinates(coordinates)
        molecule.set_orientation(orientation)
        molecules.append(molecule)

    #centre_indexes = []
    #if excitations is not None:
    #    centre_indexes = excite_system(molecules, excitations)

    supercell = np.diag(np.multiply(lattice['size'], lattice['parameters']))

    return System(molecules, conditions, supercell)

    # return molecules, supercell


######################################################################################
#                           DICTIONARIES
######################################################################################

get_ordered_system = {'1': get_1d_ordered_system,
                      '2': get_2d_ordered_system,
                      '3': get_3d_ordered_system}


######################################################################################
#                              ASSISTANT FUNCTIONS
######################################################################################


def distance_checking(coordinates, molecules):
    """
    :param coordinates: coordinates of the studied molecule
    :param molecules: list of the molecules already defined
    :return: Boolean. Indicates if the new molecule is too close to some of the already defined.
    """
    coordinates = np.array(coordinates)

    for molecule in molecules:
        if distance.euclidean(coordinates, molecule.get_coordinates()) < molecule.characteristic_length:
            return True

    return False


def get_capacity(dimensions):
    """
    :param dimensions: Number of molecules per side
    :param number_molecules: total number of molecules given
    :return: Integer. Maximus number of molecules that could fit in the system.
    """
    capacity = 1
    for dimension in dimensions:
        capacity = capacity*int(dimension)
    return capacity


def check_lattice(lattice_parameter, generic_molecule):
    """
    :param lattice_parameter: Lattice parameter
    :param generic_molecule: instance of class molecule with all its natural parameters defined
    :return: Boolean. Checks if the lattice parameter chosen is smaller than the molecular characteristic length.
    """
    for lat_pam in lattice_parameter:
        if lat_pam < generic_molecule.characteristic_length:
            return False

    return True


######################################################################################

get_symmetry = {'parallel': [1, 2], 'antiparallel': [-1, 1], 'random': [0,0]}
# dictionary with the sets of values to build parallelism or antiparallelism


def get_orientation(orientation, generic_molecule, dimensionality, pointing):
    """
    :param orientation: clue parameter: parallel, antiparallel or random
    :param generic_molecule: generic instance of class molecule
    :param dimensionality: dimensionality of the system
    :param pointing: -1, 1 value used in the cases of antiparallelism or parallelism (respectively)
    :return: the orientation of the molecule in the global reference system
    if random is given computes a random orientation using 2 or 3 angles depending on the dimensionality
    else: gives a reference orientation given as an internal property of generic molecule (parallel) or the respective
    anti parallel vector. This reference orientation is the molecular orientation in which the transition dipole moment
    in the molecular reference system coincides with the itself in the global reference system
    """

    if orientation is 'random':
        if dimensionality == 3:
            # for a 3 dimensions disordered system 2 angles are taken for the generation of a random orientation
            phi = 2 * pi * np.random.rand()
            theta = pi * np.random.rand()

            return [np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)]

        else:
            # for a 1 or 2 dimensions disordered system 1 angles is taken for the generation of a random orientation
            phi = 2 * pi * np.random.rand()

            return [np.cos(phi), np.sin(phi), 0]

    else:
        # for a parallel or antiparallel orientation we take a reference orientation and built parallel or antiparallel
        # vectors using pointing and the dictionary get_symmetry
        reference_orientation = generic_molecule.dipole_moment_direction

        return reference_orientation * pointing
