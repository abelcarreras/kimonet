import numpy as np
from kimonet.utils import distance_vector_periodic
import inspect
from kimonet.utils.units import VAC_PERMITTIVITY
from kimonet.core.processes.transitions import Transition
from kimonet.utils.units import ATOMIC_TO_ANGS_EL
import kimonet.core.processes.forster as forster
from kimonet.utils import rotate_vector


coupling_data = {}


def generate_hash_2(function_name, d_transition, a_transition, d_orientation, a_orientation, r_vector, data_list):

    return hash((d_transition, a_transition,
                 d_orientation.tobytes(),
                 a_orientation.tobytes(),
                 function_name, tuple(data_list),
                 np.array(r_vector).tobytes()))


def generate_hash(function_name, initial, final, distance, data_list):

    return hash((initial, final, function_name,
                 function_name, tuple(data_list),
                 distance))


def forster_coupling(initial, final, ref_index=1, transitions=()):
    """
    Compute Forster coupling in eV

    :param initial: initial states
    :param final: final states
    :return: Forster coupling
    """

    d_transition = Transition(initial[0], final[1])
    a_transition = Transition(initial[1], final[0])

    d_orientation = initial[0].get_center().molecular_orientation()
    a_orientation = initial[1].get_center().molecular_orientation()

    #r_vector = initial[1].get_coordinates_absolute() - final[1].get_coordinates_absolute()
    r_vector = initial[0].get_coordinates_absolute() - final[0].get_coordinates_absolute()

    hash_string = generate_hash_2(inspect.currentframe().f_code.co_name,
                                  d_transition, a_transition,
                                  d_orientation, a_orientation,
                                  r_vector, [ref_index, tuple(transitions)])

    if hash_string in coupling_data:
        return coupling_data[hash_string]

    try:
        mu_a = transitions[transitions.index(a_transition)].tdm
        mu_d = transitions[transitions.index(d_transition)].tdm
    except ValueError:
        raise Exception('TDM for {} / {} not defined'.format(a_transition, d_transition))

    mu_d = rotate_vector(mu_d, d_orientation) * ATOMIC_TO_ANGS_EL
    mu_a = rotate_vector(mu_a, a_orientation) * ATOMIC_TO_ANGS_EL

    coupling_data[hash_string] = forster.dipole(mu_d, mu_a, r_vector, n=ref_index)

    return coupling_data[hash_string]


def forster_coupling_py(initial, final, ref_index=1, transitions=()):
    """
    Compute Forster coupling in eV

    :param initial: initial states
    :param final: final states
    :return: Forster coupling
    """

    d_transition = Transition(initial[0], final[1])
    a_transition = Transition(initial[1], final[0])

    d_orientation = initial[0].get_center().molecular_orientation()
    a_orientation = initial[1].get_center().molecular_orientation()

    # r_vector = initial[1].get_coordinates_absolute() - final[1].get_coordinates_absolute()
    r_vector = initial[0].get_coordinates_absolute() - final[0].get_coordinates_absolute()

    hash_string = generate_hash_2(inspect.currentframe().f_code.co_name,
                                  d_transition, a_transition,
                                  d_orientation, a_orientation,
                                  r_vector, [ref_index, tuple(transitions)])

    if hash_string in coupling_data:
        return coupling_data[hash_string]

    try:
        mu_a = transitions[transitions.index(a_transition)].tdm
        mu_d = transitions[transitions.index(d_transition)].tdm
    except ValueError:
        raise Exception('TDM for {} / {} not defined'.format(a_transition, d_transition))

    mu_d = rotate_vector(mu_d, d_orientation) * ATOMIC_TO_ANGS_EL
    mu_a = rotate_vector(mu_a, a_orientation) * ATOMIC_TO_ANGS_EL

    distance = np.linalg.norm(r_vector)

    k = orientation_factor(mu_d, mu_a, r_vector)              # orientation factor between molecules

    k_e = 1.0/(4.0*np.pi*VAC_PERMITTIVITY)

    coupling_data[hash_string] = k_e * k * np.dot(mu_d, mu_a) / (ref_index**2 * distance**3)

    return coupling_data[hash_string]


def forster_coupling_extended(initial, final, ref_index=1, transitions=(), longitude=3, n_divisions=300):
    """
    Compute Forster coupling in eV

    :param initial: initial states
    :param final: final states
    :param longitude: extension length of the dipole
    :param n_divisions: number of subdivisions. To use with longitude. Increase until convergence.
    :return: Forster coupling
    """

    d_transition = Transition(initial[0], final[1])
    a_transition = Transition(initial[1], final[0])

    d_orientation = initial[0].get_center().molecular_orientation()
    a_orientation = initial[1].get_center().molecular_orientation()

    r_vector = initial[0].get_coordinates_absolute() - final[0].get_coordinates_absolute()

    hash_string = generate_hash_2(inspect.currentframe().f_code.co_name,
                                  d_transition, a_transition,
                                  d_orientation, a_orientation,
                                  r_vector, [ref_index, tuple(transitions), longitude, n_divisions])

    if hash_string in coupling_data:
        return coupling_data[hash_string]

    mu_a = transitions[transitions.index(a_transition)].tdm
    mu_d = transitions[transitions.index(d_transition)].tdm

    mu_d = rotate_vector(mu_d, d_orientation) * ATOMIC_TO_ANGS_EL
    mu_a = rotate_vector(mu_a, a_orientation) * ATOMIC_TO_ANGS_EL

    # mu_d = donor.get_transition_moment(to_state=_GS_)            # transition dipole moment (donor) e*angs
    # mu_a = acceptor.get_transition_moment(to_state=donor.state)  # transition dipole moment (acceptor) e*angs

    #r_vector = intermolecular_vector(donor, acceptor, supercell, cell_increment)  # position vector between donor and acceptor

    coupling_data[hash_string] = forster.dipole_extended(r_vector, mu_a, mu_d,
                                                         n=ref_index,
                                                         longitude=longitude,
                                                         n_divisions=n_divisions)

    return coupling_data[hash_string]


def forster_coupling_extended_py(initial, final, ref_index=1, transitions=(), longitude=3, n_divisions=300):
    """
    Compute Forster coupling in eV (pure python version)

    :param initial: initial states
    :param final: final states
    :param longitude: extension length of the dipole
    :param n_divisions: number of subdivisions. To use with longitude. Increase until convergence.
    :return: Forster coupling
    """

    d_transition = Transition(initial[0], final[1])
    a_transition = Transition(initial[1], final[0])

    d_orientation = initial[0].get_center().molecular_orientation()
    a_orientation = initial[1].get_center().molecular_orientation()

    r_vector = initial[0].get_coordinates_absolute() - final[0].get_coordinates_absolute()

    hash_string = generate_hash_2(inspect.currentframe().f_code.co_name,
                                  d_transition, a_transition,
                                  d_orientation, a_orientation,
                                  r_vector, [ref_index, tuple(transitions), longitude, n_divisions])

    if hash_string in coupling_data:
        return coupling_data[hash_string]

    mu_a = transitions[transitions.index(a_transition)].tdm
    mu_d = transitions[transitions.index(d_transition)].tdm

    mu_d = rotate_vector(mu_d, d_orientation) * ATOMIC_TO_ANGS_EL
    mu_a = rotate_vector(mu_a, a_orientation) * ATOMIC_TO_ANGS_EL

    # mu_d = donor.get_transition_moment(to_state=_GS_)            # transition dipole moment (donor) e*angs
    # mu_a = acceptor.get_transition_moment(to_state=donor.state)  # transition dipole moment (acceptor) e*angs

    #r_vector = intermolecular_vector(donor, acceptor, supercell, cell_increment)  # position vector between donor and acceptor

    mu_ai = mu_a / n_divisions
    mu_di = mu_d / n_divisions

    k_e = 1.0 / (4.0 * np.pi * VAC_PERMITTIVITY)

    forster_coupling = 0
    for x in np.linspace(-0.5 + 0.5/n_divisions, 0.5 - 0.5/n_divisions, n_divisions):
        for y in np.linspace(-0.5 + 0.5/n_divisions, 0.5 - 0.5/ n_divisions, n_divisions):

            #print(x, y)
            dr_a = mu_a / np.linalg.norm(mu_a) * longitude * x
            dr_d = mu_d / np.linalg.norm(mu_d) * longitude * y
            r_vector_i = r_vector + dr_a + dr_d

            distance = np.linalg.norm(r_vector_i)

            k = orientation_factor(mu_ai, mu_di, r_vector_i)              # orientation factor between molecules

            forster_coupling += k_e * k * np.dot(mu_ai, mu_di) / (ref_index**2 * distance**3)

    coupling_data[hash_string] = forster_coupling                            # memory update for new couplings

    return forster_coupling


def forster_coupling_sine_py(initial, final, ref_index=1, transitions=(), longitude=0.0, n_divisions=1):
    """

    :param initial: initial states
    :param final: final states
    :param ref_index:
    :param transitions:
    :param longitude: extension length of the dipole
    :param n_divisions: number of subdivisions. To use with longitude. Increase until convergence
    :return:
    """

    n_divisions += 2

    d_transition = Transition(initial[0], final[1])
    a_transition = Transition(initial[1], final[0])

    d_orientation = initial[0].get_center().molecular_orientation()
    a_orientation = initial[1].get_center().molecular_orientation()

    r_vector = initial[0].get_coordinates_absolute() - final[0].get_coordinates_absolute()

    hash_string = generate_hash_2(inspect.currentframe().f_code.co_name,
                                  d_transition, a_transition,
                                  d_orientation, a_orientation,
                                  r_vector, [ref_index, tuple(transitions), longitude, n_divisions])

    if hash_string in coupling_data:
        return coupling_data[hash_string]

    mu_a = transitions[transitions.index(a_transition)].tdm
    mu_d = transitions[transitions.index(d_transition)].tdm

    mu_d = rotate_vector(mu_d, d_orientation) * ATOMIC_TO_ANGS_EL
    mu_a = rotate_vector(mu_a, a_orientation) * ATOMIC_TO_ANGS_EL

    k_e = 1.0 / (4.0 * np.pi * VAC_PERMITTIVITY)

    forster_coupling = 0
    for i, x in enumerate(np.linspace(-1 + 1/n_divisions, 1 - 1/n_divisions, n_divisions)):

        #test=[]
        for j, y in enumerate(np.linspace(-1 + 1 / n_divisions, 1 - 1 / n_divisions, n_divisions)):

            if n_divisions-1 > i > 0 and n_divisions-1 > j > 0:
            #if True:
                n_d = n_divisions - 1
                mu_ai = np.sin(i*np.pi/n_d)/np.sum([np.sin(ik*np.pi/n_d) for ik in range(n_d)])*mu_a
                mu_di = np.sin(j*np.pi/n_d)/np.sum([np.sin(jk*np.pi/n_d) for jk in range(n_d)])*mu_d
                #print(x, y, np.sin(j*np.pi/n_d)/np.sum([np.sin(jk*np.pi/n_d) for jk in range(n_d)]))

                dr_a = mu_a / np.linalg.norm(mu_a) * longitude * x
                dr_d = mu_d / np.linalg.norm(mu_d) * longitude * y

                r_vectori = r_vector + dr_a - dr_d

                distance = np.linalg.norm(r_vectori)

                #if np.linalg.norm(mu_ai) != 0 and np.linalg.norm(mu_di) != 0:
                    # print(i, j)
                k = orientation_factor(mu_ai, mu_di, r_vectori)              # orientation factor between molecules
                forster_coupling += k_e * k * np.dot(mu_ai, mu_di) / (distance**3)
                #test.append(np.linalg.norm(mu_di))

        #plt.plot(test, 'o-')
        #plt.show()
    coupling_data[hash_string] = forster_coupling                            # memory update for new couplings

    return coupling_data[hash_string]


def intermolecular_vector(molecule_1, molecule_2, supercell, cell_incr):
    """
    :param molecule_1: donor
    :param molecule_2: acceptor
    :return: the distance between the donor and the acceptor
    """

    position_d = molecule_1.get_coordinates()
    position_a = molecule_2.get_coordinates()
    r_vector = position_a - position_d
    r = distance_vector_periodic(r_vector, supercell, cell_incr)
    return r


def orientation_factor(u_d, u_a, r):
    """
    :param u_d: dipole transition moment of the donor
    :param u_a: dipole transition moment of the acceptor
    :param r:  intermolecular_distance
    :type u_d: np.ndarray
    :type u_a: np.ndarray
    :type r: float

    :return: the orientational factor between both molecules
    :rtype: float
    """
    nd = unit_vector(u_d)
    na = unit_vector(u_a)
    e = unit_vector(r)
    return np.dot(nd, na) - 3*np.dot(e, nd)*np.dot(e, na)


def unit_vector(vector):
    """
    :param vector:
    :return: computes a unity vector in the direction of vector
    """
    return vector / np.linalg.norm(vector)


def dexter_coupling(initial, final, k_factor=1):
    """
    Compute Dexter coupling in eV

    :param initial: initial states
    :param final: final states
    :param vdw_radius: dictionary with Van der Waals radius for each state
    :param k_factor: Dexter parameter K
    :return: Dexter coupling
    """

    function_name = inspect.currentframe().f_code.co_name

    r_vector = initial[0].get_coordinates_absolute() - final[0].get_coordinates_absolute()
    distance = np.linalg.norm(r_vector)

    # donor <-> acceptor interaction symmetry
    hash_string = generate_hash(function_name, initial, final, distance, [k_factor])

    if hash_string in coupling_data:
        return coupling_data[hash_string]

    vdw_radius_sum = initial[0].vdw_radius + final[0].vdw_radius

    # print('dexter: ', 2/vdw_radius_sum)

    dexter_coupling = k_factor * np.exp(-2 * distance / vdw_radius_sum)

    coupling_data[hash_string] = dexter_coupling                            # memory update for new couplings

    return dexter_coupling

