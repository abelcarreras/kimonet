import numpy as np
from kimonet.utils import minimum_distance_vector
import inspect
from kimonet.utils.units import VAC_PERMITTIVITY


foster_data = {}
dexter_data = {}


def forster_coupling(donor, acceptor, conditions, supercell):
    """
    Compute Forster coupling in eV

    :param donor: excited molecules. Donor
    :param acceptor: neighbouring molecule. Possible acceptor
    :param conditions: dictionary with physical conditions
    :param supercell: the supercell of the system
    :return: Forster coupling
    """

    function_name = inspect.currentframe().f_code.co_name

    # donor <-> acceptor interaction symmetry
    hash_string = str(hash((donor, function_name)) + hash((acceptor, function_name)))
    # hash_string = str(hash((donor, acceptor, function_name))) # No symmetry

    if hash_string in foster_data:
        return foster_data[hash_string]

    mu_d = donor.get_transition_moment(to_state='gs')            # transition dipole moment (donor) e*angs
    mu_a = acceptor.get_transition_moment(to_state=donor.state)  # transition dipole moment (acceptor) e*angs

    r_vector = intermolecular_vector(donor, acceptor)       # position vector between donor and acceptor
    r_vector, _ = minimum_distance_vector(r_vector, supercell)

    distance = np.linalg.norm(r_vector)
    # print('donor', donor.get_coordinates())

    n = conditions['refractive_index']                      # refractive index of the material

    k = orientation_factor(mu_d, mu_a, r_vector)              # orientation factor between molecules

    k_e = 1.0/(4.0*np.pi*VAC_PERMITTIVITY)
    forster_coupling = k_e * k**2 * np.dot(mu_d, mu_a) / (n**2 * distance**3)

    foster_data[hash_string] = forster_coupling                            # memory update for new couplings

    # print('f:', forster_coupling, distance)
    return forster_coupling


def forster_coupling_extended(donor, acceptor, conditions, supercell, longitude=1, n_divisions=10):
    """
    Compute Forster coupling in eV

    :param donor: excited molecules. Donor
    :param acceptor: neighbouring molecule. Possible acceptor
    :param conditions: dictionary with physical conditions
    :param supercell: the supercell of the system
    :param longitude: extension length of the dipole
    :param n_divisions: number of subdivisions. To use with longitude. Increase until convergence.
    :return: Forster coupling
    """

    function_name = inspect.currentframe().f_code.co_name

    # donor <-> acceptor interaction symmetry
    hash_string = str(hash((donor, function_name)) + hash((acceptor, function_name)))
    # hash_string = str(hash((donor, acceptor, function_name))) # No symmetry

    if hash_string in foster_data:
        return foster_data[hash_string]

    mu_d = donor.get_transition_moment(to_state='gs')              # transition dipole moment (donor) e*angs
    mu_a = acceptor.get_transition_moment(to_state=donor.state)    # transition dipole moment (acceptor) e*angs

    mu_ai = mu_a / n_divisions
    mu_di = mu_d / n_divisions

    n = conditions['refractive_index']                      # refractive index of the material

    r_vector = intermolecular_vector(donor, acceptor)  # position vector between donor and acceptor
    r_vector, _ = minimum_distance_vector(r_vector, supercell)

    k_e = 1.0 / (4.0 * np.pi * VAC_PERMITTIVITY)

    forster_coupling = 0
    for x in np.linspace(-1 + 1/n_divisions, 1 - 1/n_divisions, n_divisions):
        for y in np.linspace(-1 + 1 / n_divisions, 1 - 1 / n_divisions, n_divisions):

            dr_a = mu_a / np.linalg.norm(mu_a) * longitude * x
            dr_d = mu_d / np.linalg.norm(mu_d) * longitude * y
            r_vector_i = r_vector + dr_a + dr_d

            distance = np.linalg.norm(r_vector_i)

            k = orientation_factor(mu_ai, mu_di, r_vector_i)              # orientation factor between molecules

            forster_coupling += k_e * k**2 * np.dot(mu_ai, mu_di) / (n**2 * distance**3)

    foster_data[hash_string] = forster_coupling                            # memory update for new couplings

    return forster_coupling


def intermolecular_vector(donor, acceptor):
    """
    :param donor: donor
    :param acceptor: acceptor
    :return: the distance between the donor and the acceptor
    """
    position_d = donor.get_coordinates()
    position_a = acceptor.get_coordinates()
    r = position_a - position_d

    return r


def orientation_factor(u_d, u_a, r):
    """
    :param u_d: dipole transition moment of the donor
    :param u_a: dipole transition moment of the acceptor
    :param r:  intermolecular_distance
    :return: the orientational factor between both molecules
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


def dexter_coupling(donor, acceptor, conditions, supercell):
    """
    Compute Dexter coupling in eV

    :param donor: excited molecules. Donor
    :param acceptor: neighbouring molecule. Possible acceptor
    :param conditions: dictionary with physical conditions
    :param supercell: the supercell of the system
    :return: Dexter coupling
    """

    function_name = inspect.currentframe().f_code.co_name

    # donor <-> acceptor interaction symmetry
    hash_string = str(hash((donor, function_name)) + hash((acceptor, function_name)))
    # hash_string = str(hash((donor, acceptor, function_name))) # No symmetry

    if hash_string in dexter_data:
        return dexter_data[hash_string]

    r_vector = intermolecular_vector(donor, acceptor)       # position vector between donor and acceptor
    r_vector, _ = minimum_distance_vector(r_vector, supercell)

    distance = np.linalg.norm(r_vector)

    k_factor = conditions['dexter_k']
    vdw_radius_sum = donor.get_vdw_radius() + acceptor.get_vdw_radius()
    dexter_coupling = k_factor * np.exp(-2 * distance / vdw_radius_sum)

    dexter_data[hash_string] = dexter_coupling                            # memory update for new couplings

    return dexter_coupling
