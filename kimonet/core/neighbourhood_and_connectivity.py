import numpy as np
from scipy.spatial import distance

# neighbourhood is defined in this new file because it surely will be expanded when defining connectivities
# for some systems
data_center = {}

def neighbourhood2(centre, system, radius=1.05):
    """
    :param centre: Index of an excited Molecule object
    :param system: dictionary with all the information of the system
    :param radius: Effective distance where interaction may be considerable. Default 0.11
    :return: List of indexes of molecules in a neighbourhood of center
    If there is not any neighbours in the defined neighbourhood an alert is printed.
    """

    molecules = system['molecules']
    center_position = np.array(molecules[centre].get_coordinates())
    # position of the excited molecule (as a numpy array)

    # some connectivity tricks have been defined for specific cases: 1d, 2d ordered systems (squared systems).

    # Flux control. We ensure that variable type is defined (in general will be an argument of system).
    if 'type' in system:
        type = system['type']

    else:
        type = None

    # Connectivity 1d
    if type == '1d_ordered':
        neighbours = [centre-1, centre+1]

        if centre == 0:                      # removes the previous if the center is in the first position of the list
            neighbours.remove(centre-1)
        elif centre == len(molecules)-1:     # removes the next if the center is in the last position of the list
            neighbours.remove(centre+1)

    # Connectivity 2d
    if type == '2d_ordered':
        dimensions = system['lattice']['dimensions']
        lat_param = system['lattice']['lattice_parameter']

        neighbours = [centre-dimensions[1], centre-1, centre+1, centre+dimensions[1]]
        # the previous and the next molecule are taken. Besides, the molecules of the adjacent columns need to be taken
        # The number of molecules per column is used to take these last positions

        # if the excited molecule is on the border of the system the extra molecules are removed.
        if center_position[0] == -dimensions[0]*lat_param[0] / 2:                  # **
            neighbours.remove(centre-dimensions[1])
        elif center_position[1] == -dimensions[1]*lat_param[1] / 2:                # **
            neighbours.remove(centre-1)
        elif center_position[0] == dimensions[0] * lat_param[0] / 2 - lat_param[0]:   # **
            neighbours.remove(centre + dimensions[1])
        elif center_position[1] == dimensions[1] * lat_param[1] / 2 - lat_param[1]:   # **
            neighbours.remove(centre+1)
        # **: bordering coordinates for each side
        # the way they are chosen and removed has been tested.

    # for different type systems the neighbours are searched by hand. The program looks for molecules around the centre
    # closer than a radius.
    else:
        neighbours = []
        for i, molecule in enumerate(molecules):
            coordinates = np.array(molecule.coordinates)

            # checks if the distance is smaller than the given radius and bigger than 0 (avoid taking the center as
            # neighbour) If the molecule satisfies both its index is taken.
            if 0 < distance.euclidean(center_position, coordinates) < radius:
                neighbours.append(i)

    # Printed alert if no neighbours are found.
    if len(neighbours) == 0:
        print('No neighbours found. Check neighbourhood radius')

    return neighbours


def get_neighbours_naive(centre, system, radius=1.05):
    """
    :param centre: Index of an excited Molecule object
    :param system: dictionary with all the information of the system
    :param radius: Effective distance where interaction may be considerable. Default 0.11
    :return: List of indexes of molecules in a neighbourhood of center
    If there is not any neighbours in the defined neighbourhood an alert is printed.
    """

    if '{}'.format(centre) in data_center:
        return data_center['{}'.format(centre)]

    #print(system)
    molecules = system.molecules
    supercell = system.supercell

    # print([mol.get_coordinates() for mol in molecules])

    center_position = molecules[centre].get_coordinates()
    # position of the excited molecule (as a numpy array)

    # some connectivity tricks have been defined for specific cases: 1d, 2d ordered systems (squared systems).

    # Flux control. We ensure that variable type is defined (in general will be an argument of system).
    neighbours = []
    for i, molecule in enumerate(molecules):
        coordinates = molecule.get_coordinates()

        for l_vector in supercell:
            coord_plus = coordinates + np.array(l_vector)
            coord_minus = coordinates - np.array(l_vector)
            if 0 < distance.euclidean(center_position, coordinates) < radius:
                neighbours.append(i)
            elif 0 < distance.euclidean(center_position, coord_plus) < radius:
                neighbours.append(i)
            elif 0 < distance.euclidean(center_position, coord_minus) < radius:
                neighbours.append(i)

    neighbours = np.unique(neighbours)
    # data_center['{}'.format(centre)] = neighbours

    # Printed alert if no neighbours are found.
    if len(neighbours) == 0:
        print('No neighbours found. Check neighbourhood radius')

    return neighbours
