import numpy as np
from kimonet.utils import distance_vector_periodic
import inspect
from kimonet.utils.units import VAC_PERMITTIVITY
from kimonet.core.processes.transitions import Transition
from kimonet.utils.units import DEBYE_TO_ANGS_EL
import kimonet.core.processes.forster as forster
from kimonet.utils import rotate_vector


coupling_data = {}


def generate_hash(function_name, donor, acceptor, conditions, supercell, cell_incr):
    # return str(hash((donor, acceptor, function_name))) # No symmetry

    return str(hash((donor, acceptor, function_name)) +
               hash(frozenset(conditions.items()))
               ) + np.array2string(np.array(supercell), precision=12) + np.array2string(np.array(cell_incr, dtype=int))


def forster_coupling(initial, final, conditions, supercell, ref_index=1, transition_moment=None):
    """
    Compute Forster coupling in eV
    Only works for 1 molecule states

    :param donor: excited molecules. Donor
    :param acceptor: neighbouring molecule. Possible acceptor
    :param conditions: dictionary with physical conditions
    :param supercell: the supercell of the system
    :param cell_incr: integer vector indicating the difference between supercells of acceptor and donor
    :return: Forster coupling
    """

    donor = initial[0].get_center()
    acceptor = initial[1].get_center()

    function_name = inspect.currentframe().f_code.co_name

    cell_increment = np.array(final[0].get_center().cell_state) - np.array(initial[1].get_center().cell_state)

    # donor <-> acceptor interaction symmetry
    hash_string = generate_hash(function_name, donor, acceptor, conditions, supercell, cell_increment)

    if hash_string in coupling_data:
        return coupling_data[hash_string]

    mu_d = transition_moment[Transition(*initial)]
    mu_a = transition_moment[Transition(*final)]

    mu_d = rotate_vector(mu_d, donor.molecular_orientation()) * DEBYE_TO_ANGS_EL
    mu_a = rotate_vector(mu_a, acceptor.molecular_orientation()) * DEBYE_TO_ANGS_EL

    # mu_d = donor.get_transition_moment(to_state=_GS_)            # transition dipole moment (donor) e*angs
    # mu_a = acceptor.get_transition_moment(to_state=donor.state)  # transition dipole moment (acceptor) e*angs


    r_vector = intermolecular_vector(donor, acceptor, supercell, cell_increment) # position vector between donor and acceptor

    coupling_data[hash_string] = forster.dipole(mu_d, mu_a, r_vector, n=ref_index)
    distance = np.linalg.norm(r_vector)

    k = orientation_factor(mu_d, mu_a, r_vector)              # orientation factor between molecules

    k_e = 1.0/(4.0*np.pi*VAC_PERMITTIVITY)

    forster_coupling = k_e * k**2 * np.dot(mu_d, mu_a) / (ref_index**2 * distance**3)

    coupling_data[hash_string] = forster_coupling                            # memory update for new couplings

    # print('f:', forster_coupling, distance, cell_incr)
    return forster_coupling


def forster_coupling_py(initial, final, conditions, supercell, ref_index=1, transition_moment=None):
    """
    Compute Forster coupling in eV

    :param donor: excited molecules. Donor
    :param acceptor: neighbouring molecule. Possible acceptor
    :param conditions: dictionary with physical conditions
    :param supercell: the supercell of the system
    :param cell_incr: integer vector indicating the difference between supercells of acceptor and donor
    :return: Forster coupling
    """

    donor = initial[0].get_center()
    acceptor = initial[1].get_center()

    function_name = inspect.currentframe().f_code.co_name

    cell_increment = np.array(final[0].get_center().cell_state) - np.array(initial[1].get_center().cell_state)

    # donor <-> acceptor interaction symmetry
    hash_string = generate_hash(function_name, donor, acceptor, conditions, supercell, cell_increment)

    if hash_string in coupling_data:
        return coupling_data[hash_string]


    mu_d = transition_moment[Transition(*initial)]
    mu_a = transition_moment[Transition(*final)]

    mu_d = rotate_vector(mu_d, donor.molecular_orientation()) * DEBYE_TO_ANGS_EL
    mu_a = rotate_vector(mu_a, acceptor.molecular_orientation()) * DEBYE_TO_ANGS_EL

    # mu_d = donor.get_transition_moment(to_state=_GS_)            # transition dipole moment (donor) e*angs
    # mu_a = acceptor.get_transition_moment(to_state=donor.state)  # transition dipole moment (acceptor) e*angs

    r_vector = intermolecular_vector(donor, acceptor, supercell, cell_increment) # position vector between donor and acceptor

    distance = np.linalg.norm(r_vector)

    k = orientation_factor(mu_d, mu_a, r_vector)              # orientation factor between molecules

    k_e = 1.0/(4.0*np.pi*VAC_PERMITTIVITY)

    coupling_data[hash_string] = k_e * k**2 * np.dot(mu_d, mu_a) / (ref_index**2 * distance**3)

    return coupling_data[hash_string]


def forster_coupling_extended(initial, final, conditions, supercell, ref_index=1, transition_moment=None,
                              longitude=3, n_divisions=300):
    """
    Compute Forster coupling in eV

    :param donor: excited molecules. Donor
    :param acceptor: neighbouring molecule. Possible acceptor
    :param conditions: dictionary with physical conditions
    :param supercell: the supercell of the system
    :param cell_incr: integer vector indicating the difference between supercells of acceptor and donor
    :param longitude: extension length of the dipole
    :param n_divisions: number of subdivisions. To use with longitude. Increase until convergence.
    :return: Forster coupling
    """

    donor = initial[0].get_center()
    acceptor = initial[1].get_center()

    function_name = inspect.currentframe().f_code.co_name

    cell_increment = np.array(final[0].get_center().cell_state) - np.array(initial[1].get_center().cell_state)

    # donor <-> acceptor interaction symmetry
    hash_string = generate_hash(function_name, donor, acceptor, conditions, supercell, cell_increment)
    # hash_string = str(hash((donor, acceptor, function_name))) # No symmetry

    if hash_string in coupling_data:
        return coupling_data[hash_string]

    mu_d = transition_moment[Transition(*initial)]
    mu_a = transition_moment[Transition(*final)]

    mu_d = rotate_vector(mu_d, donor.molecular_orientation()) * DEBYE_TO_ANGS_EL
    mu_a = rotate_vector(mu_a, acceptor.molecular_orientation()) * DEBYE_TO_ANGS_EL

    # mu_d = donor.get_transition_moment(to_state=_GS_)            # transition dipole moment (donor) e*angs
    # mu_a = acceptor.get_transition_moment(to_state=donor.state)  # transition dipole moment (acceptor) e*angs

    r_vector = intermolecular_vector(donor, acceptor, supercell, cell_increment)  # position vector between donor and acceptor

    coupling_data[hash_string] = forster.dipole_extended(r_vector, mu_a, mu_d,
                                                         n=ref_index,
                                                         longitude=longitude,
                                                         n_divisions=n_divisions)

    return coupling_data[hash_string]


def forster_coupling_extended_py(initial, final, conditions, supercell, ref_index=1, transition_moment=None,
                                 longitude=3, n_divisions=300):
    """
    Compute Forster coupling in eV (pure python version)

    :param donor: excited molecules. Donor
    :param acceptor: neighbouring molecule. Possible acceptor
    :param conditions: dictionary with physical conditions
    :param supercell: the supercell of the system
    :param cell_incr: integer vector indicating the difference between supercells of acceptor and donor
    :param longitude: extension length of the dipole
    :param n_divisions: number of subdivisions. To use with longitude. Increase until convergence.
    :return: Forster coupling
    """

    donor = initial[0].get_center()
    acceptor = initial[1].get_center()

    function_name = inspect.currentframe().f_code.co_name

    cell_increment = np.array(final[0].get_center().cell_state) - np.array(initial[1].get_center().cell_state)

    # donor <-> acceptor interaction symmetry
    hash_string = generate_hash(function_name, donor, acceptor, conditions, supercell, cell_increment)
    # hash_string = str(hash((donor, acceptor, function_name))) # No symmetry

    if hash_string in coupling_data:
        return coupling_data[hash_string]


    mu_d = transition_moment[Transition(*initial)]
    mu_a = transition_moment[Transition(*final)]

    mu_d = rotate_vector(mu_d, donor.molecular_orientation()) * DEBYE_TO_ANGS_EL
    mu_a = rotate_vector(mu_a, acceptor.molecular_orientation()) * DEBYE_TO_ANGS_EL

    # mu_d = donor.get_transition_moment(to_state=_GS_)            # transition dipole moment (donor) e*angs
    # mu_a = acceptor.get_transition_moment(to_state=donor.state)  # transition dipole moment (acceptor) e*angs

    # ref_index = conditions['refractive_index']                      # refractive index of the material

    r_vector = intermolecular_vector(donor, acceptor, supercell, cell_increment)  # position vector between donor and acceptor

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

            forster_coupling += k_e * k**2 * np.dot(mu_ai, mu_di) / (ref_index**2 * distance**3)

    coupling_data[hash_string] = forster_coupling                            # memory update for new couplings

    return forster_coupling


def intermolecular_vector(donor, acceptor, supercell, cell_incr):
    """
    :param donor: donor
    :param acceptor: acceptor
    :return: the distance between the donor and the acceptor
    """
    position_d = donor.get_coordinates()
    position_a = acceptor.get_coordinates()
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


def dexter_coupling(initial, final, conditions, supercell):
    """
    Compute Dexter coupling in eV

    :param donor: excited molecules. Donor
    :param acceptor: neighbouring molecule. Possible acceptor
    :param conditions: dictionary with physical conditions
    :param supercell: the supercell of the system
    :return: Dexter coupling
    """

    function_name = inspect.currentframe().f_code.co_name

    cell_increment = np.array(final[0].get_center().cell_state) - np.array(initial[1].get_center().cell_state)
    donor = initial[0].get_center()
    acceptor = initial[1].get_center()

    # donor <-> acceptor interaction symmetry
    hash_string = generate_hash(function_name, donor, acceptor, conditions, supercell, cell_increment)

    if hash_string in coupling_data:
        return coupling_data[hash_string]

    r_vector = intermolecular_vector(donor, acceptor, supercell, cell_increment)       # position vector between donor and acceptor

    distance = np.linalg.norm(r_vector)

    k_factor = conditions['dexter_k']
    vdw_radius_sum = donor.get_vdw_radius() + acceptor.get_vdw_radius()
    dexter_coupling = k_factor * np.exp(-2 * distance / vdw_radius_sum)

    coupling_data[hash_string] = dexter_coupling                            # memory update for new couplings

    return dexter_coupling

