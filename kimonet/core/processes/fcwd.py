import numpy as np
from kimonet.utils.units import BOLTZMANN_CONSTANT

###########################################################################################################
#                                 Frank-Condon weighted density
###########################################################################################################
overlap_data = {}


def marcus_fcwd(donor, acceptor, conditions):
    """
    :param donor:
    :param acceptor:
    :param conditions:
    :return: The spectral overlap between the donor and the acceptor according to Marcus formula.
    """
    T = conditions['temperature']       # temperature (K)

    excited_state = donor.electronic_state()
    gibbs_energy = donor.state_energies[excited_state] - acceptor.state_energies[excited_state]
    # Gibbs energy: energy difference between the equilibrium points of the excited states

    reorganization = acceptor.reorganization_energies[excited_state]
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


def gaussian_fcwd(donor, acceptor, conditions):

    """
    :param donor: energy diference between states
    :param acceptor: deviation in energy units
    :return: Franck-Condon-weighted density of states in gaussian aproximation
    """

    excited_state = donor.electronic_state()
    delta = donor.state_energies[excited_state] - acceptor.state_energies[excited_state]
    deviation = conditions['a_e_spectra_deviation'] / 27.211     # atomic units

    info = str(hash((delta, deviation)))

    if info in overlap_data:
        fcwd = overlap_data[info]

    else:
        fcwd = np.exp(- delta**2 / (2 * deviation) ** 2) / (2 * np.sqrt(np.pi) * deviation)
        overlap_data[info] = fcwd

    return fcwd

