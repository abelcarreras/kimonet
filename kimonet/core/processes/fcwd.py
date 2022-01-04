import numpy as np
from kimonet.utils.units import BOLTZMANN_CONSTANT
from scipy.integrate import quad
import math

###########################################################################################################
#                 Frank-Condon weighted density (OLD and deprecated)
###########################################################################################################
overlap_data = {}


# deprecated
def general_fcwd(donor, acceptor, process, conditions):
    """
    :param donor:
    :param acceptor:
    :param process:
    :param conditions:
    :return: The spectral overlap between the donor and the acceptor
    """

    # testing normalization

    # info = str(hash((donor, acceptor, process, str(conditions), 'general_fcwd')))
    # info = str(hash(donor) + hash(acceptor) + hash(str(conditions) + 'general_fcwd'))

    transition_donor = (process.initial[0], process.final_safe[0])
    transition_acceptor = (process.initial[1], process.final_safe[1])

    donor_vib_dos = donor.get_vib_dos(transition_donor)
    acceptor_vib_dos = acceptor.get_vib_dos(transition_acceptor)

    # print(donor_vib_dos)
    info = str(hash(donor_vib_dos) + hash(acceptor_vib_dos))

    if info in overlap_data:
        # the memory is used if the overlap has been already computed
        return overlap_data[info]

    # test_donor = quad(donor_vib_dos, 0, np.inf,  epsabs=1e-20)[0]
    # test_acceptor = quad(acceptor_vib_dos, 0, np.inf,  epsabs=1e-20)[0]

    # print('test_donor', test_donor)
    # print('test_acceptor', test_acceptor)

    # assert math.isclose(test_donor, 1.0, abs_tol=0.01)
    # assert math.isclose(test_acceptor, 1.0, abs_tol=0.01)

    def overlap(x):
        return donor_vib_dos(x) * acceptor_vib_dos(x)

    overlap_data[info] = quad(overlap, 0, np.inf,  epsabs=1e-5, limit=1000)[0]

    return overlap_data[info]

    # return quad(integrand, 0, np.inf, args=(donor, acceptor))[0]


# deprecated (only used in test)
def marcus_fcwd_old(donor, acceptor, conditions):
    """
    :param donor:
    :param acceptor:
    :param conditions:
    :return: The spectral overlap between the donor and the acceptor according to Marcus formula.
    """
    import warnings

    warnings.warn("This method will be deprecated, use vibrations classes instead",
                  DeprecationWarning)

    T = conditions['temperature']       # temperature (K)

    excited_state = donor.state.label
    gibbs_energy = donor.state_energies[excited_state] - acceptor.state_energies[excited_state]
    # Gibbs energy: energy difference between the equilibrium points of the excited states

    reorganization = acceptor.reorganization_energies[('gs', 's1')] + acceptor.reorganization_energies[('s1', 'gs')]
    # acceptor reorganization energy of the excited state

    info = str(hash((T, gibbs_energy, reorganization, 'marcus')))
    # we define a compact string with the characteristic information of the spectral overlap

    if info in overlap_data:
        # the memory is used if the overlap has been already computed
        overlap = overlap_data[info]

    else:
        overlap = 1.0 / (2 * np.sqrt(np.pi * BOLTZMANN_CONSTANT * T * reorganization)) * \
                  np.exp(-(gibbs_energy+reorganization)**2 / (4 * BOLTZMANN_CONSTANT * T * reorganization))

        overlap_data[info] = overlap
        # new values are added to the memory

    return overlap
    # Since we have a quantity in 1/eV, we use the converse function from_ev_to_au in inverse mode
    # to have a 1/au quantity.
