import numpy as np
from scipy.spatial import distance


def excited_system(molecules, initial_excitation, tolerance):
    """
    :param molecules: List of the defined molecules
    :param initial_excitation: Information about the desired excitation
    :param tolerance: (only used in pick_centre()).
    :return: list with the index of the excited molecules
    The function modifies the list of molecules with the excitons.
    """
    centre_list = []

    for state in initial_excitation:
        for position in initial_excitation[state]:
            index = get_excited_index(position, centre_list, molecules, tolerance)

            molecules[index].set_state(state)     # check the correct method
            centre_list.append(index)

    return centre_list


def excite_system(molecules, excitations):
    """
    :param molecules: List of the defined molecules
    :param excitations: Information about the desired excitation
    :param tolerance: (only used in pick_centre()).
    :return: list with the index of the excited molecules
    The function modifies the list of molecules with the excitons.
    """
    centre_list = []

    for state in excitations:
        print(state, excitations[state])

        index = get_excited_index(excitations[state], centre_list, molecules)

        print(centre_list)
        exit()

        molecules[index].set_state(state)     # check the correct method
        centre_list.append(index)

    return centre_list


########################################################################################


def get_excited_index(position, centre_list, molecules, tolerance=1e-6):
    """
    This function simply redirects the program to more specific functions
    """

    if type(position) == int:
        return pick_index(position, centre_list, molecules)

    if type(position) == list:
        return pick_with_coordinates(position, centre_list, molecules, tolerance)

    if position is 'random':
        return pick_random(centre_list, molecules)

    if position is 'first':
        return pick_first(centre_list, molecules)

    if position is 'last':
        return pick_last(centre_list, molecules)

    if position is 'centre':
        return pick_centre(centre_list, molecules, tolerance)

    if position is 'furthest':
        return pick_furthest(centre_list, molecules)


########################################################################################


def pick_random(centre_list, molecules):
    """
    :param centre_list: list with the indexes of the excited molecules
    :param molecules: list with all the instances of class Molecule
    :return: a random position of the list molecules
    """

    picked = True                   # the loop will work as long as a dexcited molecule is not chosen. If the chosen
    counter = 1                     # molecule is excited then picked = True and the loop will start again.
    while picked is True:

        index = np.random.randint(0, len(molecules))            # random position of the list molecules

        if index in centre_list:
            picked = True

            counter =+ 1
            if counter == len(molecules):                   # flux control. Stops the while loop if all molecules are
                print('All molecules excited')              # excited
                break

        else:
            picked = False
            break

    return index


def pick_first(centre_list, molecules):
    """
    :param centre_list: list with the indexes of the excited molecules
    :param molecules: list with all the instances of class Molecule
    :return: the first element of the list. If this element is already taken picks another randomly
    """
    index = 0

    if index in centre_list:
        index = pick_random(centre_list, molecules)

    return index


def pick_last(centre_list, molecules):
    """
    :param centre_list: list with the indexes of the excited molecules
    :param molecules: list with all the instances of class Molecule
    :return: the last element of the list. If this element is already taken picks another randomly
    """
    index = len(molecules)

    if index in centre_list:
        index = pick_random(centre_list, molecules)

    return index


def pick_centre(centre_list, molecules, tolerance):
    """
    :param centre_list: list with the indexes of the excited molecules
    :param molecules: list with all the instances of class Molecule
    :param tolerance: Since the centre may not coincide with the 0 point we give this extra parameter.
    The function looks for a molecule in a cercle of radius = tolerance and it is considered the centre.
    It is taken as the average over the lattice parameters in ordered systems and as 0.2 nm in disordered systems
    :return: the index of the excited molecule
    """

    index = None                                    # index is set to None until a number is chosen
    for i, molecule in enumerate(molecules):

        position = np.linalg.norm(molecule.get_coordinates())

        if position == 0:
            index = i
            break

        if position < tolerance:                  # checks if the chosen position if in a cercle of radius = tolerance
            index = i

    # if any other case occurs (the center is already excited or there is not a molecule in the nearby of (0,0,0)) a
    # random molecule from the list is chosen

    if index in centre_list:
        index = pick_random(centre_list, molecules)

    if index is None:
        index, closest = pick_closest(position=[0.0, 0.0, 0.0], molecules=molecules, centre_list=centre_list)

        if closest is True:
            print('No molecule found in the desired position. Picks the closest.')
            print('New position:', molecules[index].get_coordinates())

    return index


def pick_furthest(centre_list, molecules):
    """
    :param centre_list: list with the indexes of the excited molecules
    :param molecules: list with all the instances of class Molecule
    :return: the furthest molecule from the centre.
    """

    furthest_position = 0
    index = None                                    # index is set to None until a number is chosen
    for i, molecule in enumerate(molecules):

        position = np.linalg.norm(molecule.get_coordinates())
        if position > furthest_position:            # checks if the chosen position is further than the furthest at
            furthest_position = position            # the moment.
            index = i

    # if the furthest molecule from the center in the system is already excited, picks another randomly.

    if index in centre_list:
        index = pick_random(centre_list, molecules)

    return index


def pick_with_coordinates(position, centre_list, molecules, tolerance):
    """
    :param position: list with the coordinates of the excited molecule
    :param centre_list: list with the indexes of the excited molecules
    :param molecules: list of all instances of class molecule
    :param tolerance: Since the centre may not coincide with the exact point we give this extra parameter.
    The function looks for a molecule in a cercle of radius = tolerance centered in the desired point
    It is taken as the average over the lattice parameters in ordered systems and as 0.2 nm in disordered systems
    :return: the index of the excited molecule
    """

    desired_position = np.array(position)           # cartesian coordinates of the excited molecule

    index = None                                    # index will be considered None until a int is chosen

    for (i, molecule) in enumerate(molecules):
        molecular_position = molecule.get_coordinates()              # position of molecule i

        inter_distance = distance.euclidean(desired_position, molecular_position)
        # distance between molecule i and the desired position

        if inter_distance == 0:             # the loops stops if the exact position is found
            index = i
            break

        if inter_distance < tolerance:         # checks if the chosen position if in a cercle of radius = tolerance
            index = i

    if index in centre_list:                   # checks if the chosen molecule has been already excited
        index = pick_random(centre_list, molecules)

    if index is None:                           # if no molecule has been found in the position picks the closest.
        index, closest = pick_closest(position, molecules, centre_list)

        if closest is True:
            print('No molecule found in the desired position. Picks the closest.')
            print('New position:', molecules[index].get_coordinates())

    return index


def pick_closest(position, molecules, centre_list):
    """
    :param position: list with the coordinates of the excited molecule
    :param molecules: list of all instances of class molecule
    :param centre_list: list with the indexes of the excited molecules
    :return: the index of the excited molecule
    """
    index = None                        # index is set to None until a number is chosen
    closest = True                      # if true a closest molecule has been found

    desired_position = np.array(position)   # cartesian coordinates of the molecule

    closest_distance = 200.0                # nm
    # sets a maximum distance (it is supposed that a closest molecule will be found)

    for i, molecule in enumerate(molecules):
        molecular_position = molecule.get_coordinates()

        inter_distance = distance.euclidean(desired_position, molecular_position)
        # distance between the molecule and the desired point

        if inter_distance < closest_distance:         # saves only if the distance is shorter than the closest_distance
            closest_distance = inter_distance
            index = i

    # if the molecule is already excited or there is no close molecule picks one randomly

    if index in centre_list:
        index = pick_random(centre_list, molecules)

    if index is None:
        index = pick_random(centre_list, molecules)
        print('No molecule close to the desired position. Picks one randomly')
        print('New position:', molecules[index].get_coordinates())
        closest = False

    return index, closest


def pick_index(position, centre_list, molecules):
    """
    :param position: index of molecules list where the exciton will be setted
    :param centre_list: list with the excited molecules
    :param molecules: list of all instances of class molecule
    :return: the index of the excited molecule
    """

    if 0 <= position < len(molecules):
        index = position

        if index in centre_list:
            index = pick_random(centre_list, molecules)

    elif -len(molecules) <= position < 0:
        index = len(molecules) + position

        if index in centre_list:
            index = pick_random(centre_list, molecules)

    else:
        print('Index out of range. Index chosen randomly')
        index = pick_random(centre_list, molecules)

    return index
