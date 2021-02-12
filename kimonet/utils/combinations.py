import numpy as np
import itertools
from copy import deepcopy
from kimonet.utils import distance_vector_periodic
from kimonet.utils import get_supercell_increments
from kimonet.system.state import ground_state as _GS_

combinations_data = {}


def distance_matrix(molecules_list, supercell, max_distance=1):

    # Set maximum possible distance everywhere
    distances = np.ones((len(molecules_list), len(molecules_list))) * np.product(np.diag(supercell))

    for i, mol_i in enumerate(molecules_list):
        for j, mol_j in enumerate(molecules_list):

            coordinates = mol_i.get_coordinates()
            center_position = mol_j.get_coordinates()
            cell_increments = get_supercell_increments(supercell, max_distance)
            for cell_increment in cell_increments:
                r_vec = distance_vector_periodic(coordinates - center_position, supercell, cell_increment)
                d = np.linalg.norm(r_vec)
                if distances[i, j] > d:
                    distances[i, j] = d

    return distances


def is_connected(molecules_list, supercell, connected_distance=1):

    # check if connected
    for group in molecules_list:
        d = distance_matrix(group, supercell, max_distance=connected_distance)
        if len(d) > 1 and  not np.all(connected_distance >= np.min(np.ma.masked_where(d == 0, d), axis=1)):
            return False

    return True


def is_connected_states(molecules_list, supercell, final_states):

    # check if connected
    for i, group in enumerate(molecules_list):
        connected_distance = final_states[i].connected_distance
        d = distance_matrix(group, supercell, max_distance=connected_distance)
        if len(d) > 1 and not np.all(connected_distance >= np.min(np.ma.masked_where(d == 0, d), axis=1)):
            return False

    return True


def partition(elements_list, group_list):

    partition = []
    n = 0
    for i in group_list:
        partition.append(tuple(elements_list[n: n+i]))
        n += i

    return partition


def combinations_group(elements_list_o, process_final, supercell=None, include_self=True, ref_states=None):

    group_list = tuple([s.size for s in process_final])

    elements_list = tuple(range(len(elements_list_o)))
    combinations_list = []

    if (elements_list, group_list) not in combinations_data:

        def combination(data_list, i, cumulative_list):
            if i == 0:
                cumulative_list = []

            for data in itertools.combinations(data_list, group_list[i]):
                rest = tuple(set(data_list) - set(data))
                cumulative_list = cumulative_list[:i] + [data]
                if len(rest) > 0:
                    combination(rest, i+1, deepcopy(cumulative_list))
                else:
                    combinations_list.append(cumulative_list)

        combination(elements_list, 0, [])
        combinations_data[(elements_list, group_list)] = combinations_list
    else:
        combinations_list = combinations_data[(elements_list, group_list)]

    # filter by connectivity
    combinations_list = [[[elements_list_o[l] for l in state] for state in conf] for conf in combinations_list]
    if supercell is not None:
        for c in combinations_list:
            if not is_connected_states(c, supercell, process_final):
                combinations_list.remove(c)

    if not include_self:
        combinations_list = combinations_list[1:]

    return combinations_list


def get_molecules_centered_in_mol(central_mol, molecules_list, supercell, size=1, connected_distance=2):
    for c in itertools.combinations(molecules_list, size):
        if central_mol in c and is_connected([c], supercell, connected_distance=connected_distance):
            return list(c)

    return None


if __name__ == '__main__':

    from kimonet.system.state import State
    from kimonet.system.molecule import Molecule

    test_list = ['a', 'b', 'c']

    group_list = [1, 2]
    configuration = partition(test_list, group_list)
    print(configuration)

    combinations_list = combinations_group(test_list, group_list)

    for c in combinations_list:
        print(c, '---', configuration)
        print(c == configuration)
        print(c)

    s1 = State(label='s1', energy=1.0, multiplicity=1)
    molecule1 = Molecule(coordinates=[0])
    molecule2 = Molecule(coordinates=[1])
    molecule3 = Molecule(coordinates=[2])
    molecule4 = Molecule(coordinates=[4])

    supercell = [[6]]

    test_list = [molecule1, molecule2, molecule3, molecule4]

    group_list = [1, 3]

    configuration = partition(test_list, group_list)

    print('initial:', test_list)
    combinations_list = combinations_group(test_list, group_list, supercell)

    for c in combinations_list:
        print('total: ', c)

    a = distance_matrix(test_list, supercell, max_distance=1)
    print(a)

    print('centered:', molecule1)
    mols = get_molecules_centered_in_mol(molecule1, test_list, supercell, size=3)
    print('mols: ', mols)
    exit()
