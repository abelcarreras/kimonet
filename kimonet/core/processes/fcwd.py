import numpy as np
from kimonet.utils.units import BOLTZMANN_CONSTANT
from scipy.integrate import quad
import math

###########################################################################################################
#                                 Frank-Condon weighted density
###########################################################################################################
overlap_data = {}


def general_fcwd(donor, acceptor, process, conditions=None):
    """
    :param donor:
    :param acceptor:
    :param process:
    :param conditions:
    :return: The spectral overlap between the donor and the acceptor
    """

    # testing normalization

    info = str(hash((donor, acceptor, process, str(conditions), 'general_fcwd')))

    if info in overlap_data:
        # the memory is used if the overlap has been already computed
        return overlap_data[info]

    transition_donor = (process.initial[0], process.final[0])
    transition_acceptor = (process.initial[1], process.final[1])

    donor_vib_dos = donor.get_vib_dos(transition_donor)
    if donor_vib_dos is None:
        donor_vib_dos = marcus_vib_spectrum(donor, transition_donor, conditions)

    acceptor_vib_dos = acceptor.get_vib_dos(transition_acceptor)
    if acceptor_vib_dos is None:
        acceptor_vib_dos = marcus_vib_spectrum(acceptor, transition_acceptor, conditions)

    test_donor = quad(donor_vib_dos, 0, np.inf,  epsabs=1e-20)[0]
    test_acceptor = quad(acceptor_vib_dos, 0, np.inf,  epsabs=1e-20)[0]

    # print('test_donor', test_donor)
    # print('test_acceptor', test_acceptor)

    assert math.isclose(test_donor, 1.0, abs_tol=0.01)
    assert math.isclose(test_acceptor, 1.0, abs_tol=0.01)

    def integrand(x):
        return donor_vib_dos(x) * acceptor_vib_dos(x)

    overlap_data[info] = quad(integrand, 0, np.inf,  epsabs=1e-20)[0]

    return overlap_data[info]

    # return quad(integrand, 0, np.inf, args=(donor, acceptor))[0]


def marcus_vib_spectrum(molecule, transition, conditions):

    import warnings
    warnings.warn('Using Marcus method')

    temp = conditions['temperature']  # temperature (K)
    reorg_ene = np.sum(molecule.reorganization_energies[transition])

    elec_trans_ene = molecule.state_energies[transition[1]] - molecule.state_energies[transition[0]]
    sign = np.sign(elec_trans_ene)

    # print('T', temp, molecule.reorganization_energies[transition], elec_trans_ene, -sign)

    def vib_spectrum(e):
        return 1.0 / (np.sqrt(4.0 * np.pi * BOLTZMANN_CONSTANT * temp * reorg_ene)) * \
               np.exp(-(elec_trans_ene - e * sign + reorg_ene) ** 2 / (4 * BOLTZMANN_CONSTANT * temp * reorg_ene))

    return vib_spectrum


def levich_jortner_vib_spectrum(molecule, transition, conditions):

    temp = conditions['temperature']       # temperature (K)
    reorg_ene = molecule.reorganization_energies[transition]
    reorg_ene = molecule.frequencies[transition]


    elec_trans_ene = molecule.state_energies[transition[1]] - molecule.state_energies[transition[0]]
    sign = np.sign(elec_trans_ene)

    # print('T', temp, molecule.reorganization_energies[transition], elec_trans_ene, -sign)

    def vib_spectrum(e):
        return 1.0 / (np.sqrt(4.0 * np.pi * BOLTZMANN_CONSTANT * temp * reorg_ene)) * \
               np.exp(-(elec_trans_ene - e * sign + reorg_ene) ** 2 / (4 * BOLTZMANN_CONSTANT * temp * reorg_ene))

    return vib_spectrum


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
