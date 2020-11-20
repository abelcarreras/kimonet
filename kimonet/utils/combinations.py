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


def partition(elements_list, group_list):

    partition = []
    n=0
    for i in group_list:
        partition.append(tuple(elements_list[n: n+i]))
        n += i

    return partition


def combinations_group(elements_list_o, group_list, supercell=None, include_self=True):
    elements_list = tuple(range(len(elements_list_o)))
    group_list = tuple(group_list)

    combinations_list = []

    #print(elements_list, group_list)
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
            if not is_connected(c, supercell, connected_distance=1):
                combinations_list.remove(c)

    if not include_self:
        combinations_list = combinations_list[1:]

    return combinations_list

def distance_pairs(center_position, system, supercell, l=1):

    # Set maximum possible distance everywhere
    distances = np.ones_like(system) * np.product(np.diag(supercell))

    pairs = []

    for vector in itertools.product(*[range(len(system))]):
        print(vector)
        coordinates = np.array(vector)
        center_position = np.array(center_position)
        cell_increments = get_supercell_increments(supercell, l)
        for cell_increment in cell_increments:
            r_vec = distance_vector_periodic(coordinates - center_position, supercell, cell_increment)
            d = np.linalg.norm(r_vec)

            if distances[vector] > d and system[vector] < 1:
                pairs.append((vector, r_vec))

    return pairs


def get_exciton_space(position, system, supercell, size=1, l=1):

    # Check if site is occupied
    if system[position].label != _GS_.label:
        return False

    pairs = distance_pairs(position, system, supercell, l=1)
    indices = np.argsort([np.linalg.norm(item[1]) for item in pairs])

    # Check if enough neighbors
    if len(indices) < size:
        return False

    print(len(pairs))
    print(len(list(itertools.combinations(pairs, size))))

    path = []
    for i in range(size):
        print('pair', pairs[indices[i]], np.linalg.norm(pairs[indices[i]][1]))
        system[tuple(pairs[indices[i]][0])] = 2
        path.append(pairs[indices[i]])

    # check if connected
    d = distance_matrix(path)
    print('d')
    print(d)
    print(np.min(np.ma.masked_where(d == 0, d), axis=1))
    print(l >= np.min(np.ma.masked_where(d == 0, d), axis=1))
    if not np.all(l >= np.min(np.ma.masked_where(d == 0, d), axis=1)):
        return False

    return True


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

    exit()

    a = get_exciton_space(position=1, system=test_list, supercell=supercell, size=2)
    print('a: ', a)