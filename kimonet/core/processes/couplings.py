import numpy as np
from kimonet.utils import distance_vector_periodic
import inspect
from kimonet.utils.units import VAC_PERMITTIVITY
from kimonet import _ground_state_
import kimonet.core.processes.forster as forster

coupling_data = {}


def generate_hash(function_name, donor, acceptor, conditions, supercell, cell_incr):
    # return str(hash((donor, acceptor, function_name))) # No symmetry

    return str(hash((donor, acceptor, function_name)) +
               hash(frozenset(conditions.items()))
               ) + np.array2string(np.array(supercell), precision=12) + np.array2string(np.array(cell_incr, dtype=int))


def forster_coupling(donor, acceptor, conditions, supercell, cell_incr, ref_index=1):
    """
    Compute Forster coupling in eV

    :param donor: excited molecules. Donor
    :param acceptor: neighbouring molecule. Possible acceptor
    :param conditions: dictionary with physical conditions
    :param supercell: the supercell of the system
    :param cell_incr: integer vector indicating the difference between supercells of acceptor and donor
    :return: Forster coupling
    """

    function_name = inspect.currentframe().f_code.co_name

    # donor <-> acceptor interaction symmetry
    hash_string = generate_hash(function_name, donor, acceptor, conditions, supercell, cell_incr)

    if hash_string in coupling_data:
        return coupling_data[hash_string]

    mu_d = donor.get_transition_moment(to_state=_ground_state_)            # transition dipole moment (donor) e*angs
    mu_a = acceptor.get_transition_moment(to_state=donor.state.label)  # transition dipole moment (acceptor) e*angs

    r_vector = intermolecular_vector(donor, acceptor, supercell, cell_incr) # position vector between donor and acceptor

    coupling_data[hash_string] = forster.dipole(mu_d, mu_a, r_vector, n=ref_index)
    distance = np.linalg.norm(r_vector)

    k = orientation_factor(mu_d, mu_a, r_vector)              # orientation factor between molecules

    k_e = 1.0/(4.0*np.pi*VAC_PERMITTIVITY)

    forster_coupling = k_e * k**2 * np.dot(mu_d, mu_a) / (ref_index**2 * distance**3)

    coupling_data[hash_string] = forster_coupling                            # memory update for new couplings

    # print('f:', forster_coupling, distance, cell_incr)
    return forster_coupling


def forster_coupling_py(donor, acceptor, conditions, supercell, cell_incr, ref_index=1):
    """
    Compute Forster coupling in eV

    :param donor: excited molecules. Donor
    :param acceptor: neighbouring molecule. Possible acceptor
    :param conditions: dictionary with physical conditions
    :param supercell: the supercell of the system
    :param cell_incr: integer vector indicating the difference between supercells of acceptor and donor
    :return: Forster coupling
    """

    function_name = inspect.currentframe().f_code.co_name

    # donor <-> acceptor interaction symmetry
    hash_string = generate_hash(function_name, donor, acceptor, conditions, supercell, cell_incr)

    if hash_string in coupling_data:
        return coupling_data[hash_string]

    mu_d = donor.get_transition_moment(to_state=_ground_state_)            # transition dipole moment (donor) e*angs
    mu_a = acceptor.get_transition_moment(to_state=donor.state.label)  # transition dipole moment (acceptor) e*angs

    r_vector = intermolecular_vector(donor, acceptor, supercell, cell_incr) # position vector between donor and acceptor

    distance = np.linalg.norm(r_vector)

    k = orientation_factor(mu_d, mu_a, r_vector)              # orientation factor between molecules

    k_e = 1.0/(4.0*np.pi*VAC_PERMITTIVITY)

    coupling_data[hash_string] = k_e * k**2 * np.dot(mu_d, mu_a) / (ref_index**2 * distance**3)

    return coupling_data[hash_string]


def forster_coupling_extended(donor, acceptor, conditions, supercell, cell_incr,
                              ref_index=1, longitude=3, n_divisions=300):
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

    function_name = inspect.currentframe().f_code.co_name

    # donor <-> acceptor interaction symmetry
    hash_string = generate_hash(function_name, donor, acceptor, conditions, supercell, cell_incr)
    # hash_string = str(hash((donor, acceptor, function_name))) # No symmetry

    if hash_string in coupling_data:
        return coupling_data[hash_string]

    mu_d = donor.get_transition_moment(to_state=_ground_state_)              # transition dipole moment (donor) e*angs
    mu_a = acceptor.get_transition_moment(to_state=donor.state.label)    # transition dipole moment (acceptor) e*angs

    r_vector = intermolecular_vector(donor, acceptor, supercell, cell_incr)  # position vector between donor and acceptor

    coupling_data[hash_string] = forster.dipole_extended(r_vector, mu_a, mu_d,
                                                         n=ref_index,
                                                         longitude=longitude,
                                                         n_divisions=n_divisions)

    return coupling_data[hash_string]


def forster_coupling_extended_py(donor, acceptor, conditions, supercell, cell_incr,
                                 ref_index=1,longitude=3, n_divisions=300):
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
    function_name = inspect.currentframe().f_code.co_name

    # donor <-> acceptor interaction symmetry
    hash_string = generate_hash(function_name, donor, acceptor, conditions, supercell, cell_incr)
    # hash_string = str(hash((donor, acceptor, function_name))) # No symmetry

    if hash_string in coupling_data:
        return coupling_data[hash_string]

    mu_d = donor.get_transition_moment(to_state=_ground_state_)              # transition dipole moment (donor) e*angs
    mu_a = acceptor.get_transition_moment(to_state=donor.state.label)    # transition dipole moment (acceptor) e*angs

    # ref_index = conditions['refractive_index']                      # refractive index of the material

    r_vector = intermolecular_vector(donor, acceptor, supercell, cell_incr)  # position vector between donor and acceptor

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


def dexter_coupling(donor, acceptor, conditions, supercell, cell_incr):
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
    hash_string = generate_hash(function_name, donor, acceptor, conditions, supercell, cell_incr)

    # hash_string = str(hash((donor, acceptor, function_name))) # No symmetry

    if hash_string in coupling_data:
        return coupling_data[hash_string]

    r_vector = intermolecular_vector(donor, acceptor, supercell, cell_incr)       # position vector between donor and acceptor

    distance = np.linalg.norm(r_vector)

    k_factor = conditions['dexter_k']
    vdw_radius_sum = donor.get_vdw_radius() + acceptor.get_vdw_radius()
    dexter_coupling = k_factor * np.exp(-2 * distance / vdw_radius_sum)

    coupling_data[hash_string] = dexter_coupling                            # memory update for new couplings

    return dexter_coupling


if __name__ == '__main__':
    import kimonet.core.processes.forster as forster
    # help(forster.dipole)

    #a = forster.dipole([1, 2, 3],[1., 2., 3.], 4)
    #print(a)
    #print('\n')
    #print('Original:', orientation_factor([1., 2., 3],[4., 2., 3.], [1., 7., 3.]))
    #print('C function: ', forster.dipole_extended([1., 2., 3.],[4., 2., 3.], [1., 7., 3.]))

