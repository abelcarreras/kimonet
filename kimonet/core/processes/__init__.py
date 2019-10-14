import numpy as np
from kimonet.core.processes.coupling_functions import couplings
from kimonet.conversion_functions import from_ns_to_au, from_ev_to_au


# Memory for the calculated decay rates and spectral overlaps is introduced.
decay_memory = {}
overlap_memory = {}


###########################################################################################################
#                                   FUNCTION 1: TRANSFER RATES
###########################################################################################################


def get_transfer_rates(centre, system, exciton_index):
    """
    :param centre: Index of the studies excited molecule
    :param system: Dictionary with the list of molecules and additional physical information
    :param exciton_index
    :return: Two lists, one with the transfer rates and the other with the transfer processes.
    For each possible acceptor in neighbour_indexes computes the transfer rate using the Fermi's Golden Rule:
        - For the spectral overlap the Marcus Formula is used for all cases.
        - For the electronic coupling an external dictionary is defined. It contains the possible couplings between
            two states (more than one allowed). The keys of this dictionary are like:
                'state1_state2' + str(additional information)
            If the key 'state1_state2' is not in the dictionary the electronic coupling shall be taken as 0.
    """

    neighbour_indexes, cell_increment = system.get_neighbours(centre)

    conditions = system.conditions           # physical conditions of the system

    donor = system.molecules[centre]         # excited molecule

    transfer_rates = []                         # list that collects the transfer rates (only the numerical values)
    transfer_processes = []                     # list that collects the transfer processes dict(donor,process,acceptor)

    for neighbour, cell_incr in zip(neighbour_indexes, cell_increment):
        acceptor = system.molecules[neighbour]
        # acceptor = molecule instance for each neighbour index.

        spectral_overlap = marcus_fcwd(donor, acceptor, conditions)
        # compute the spectral overlap with the Marcus Formula (for all possible couplings)

        prefix_key = '{}_{}'.format(donor.electronic_state(), acceptor.electronic_state())
        # first key. Only with the information of both implicated states: 'state1_state2'

        possible_couplings = slice_dict(couplings, prefix_key)
        # new dictionary with all the keys of couplings that start with prefix_key

        if len(possible_couplings) == 0:
            # when no couplings are found, the rate is taken as 0
            rate = 0
            transfer_rates.append(rate)

            process = {'donor': centre, 'process': prefix_key,
                       'acceptor': neighbour, 'index': exciton_index, 'cell_increment': cell_incr}

            transfer_processes.append(process)

        else:
            # for the possible couplings computes the transfer rate by the Fermi's Golden Rule.
            for key in possible_couplings:
                e_coupling = possible_couplings[key](donor, acceptor, conditions, system.supercell)

                rate = 2*np.pi * e_coupling**2 * spectral_overlap          # rate in a.u -- Fermi's Golden Rule
                transfer_rates.append(from_ns_to_au(rate, 'direct'))    # rate in ns⁻¹

                transfer_processes.append({'donor': int(centre), 'process': key, 'acceptor': int(neighbour),
                                           'index': exciton_index, 'cell_increment': cell_incr})

    # the process include: the index of the donor (in molecules), the key of the process,
    # the index of the acceptor (in molecules) and the index of the exciton (in centres).
    # This last parameter acts as the name of the exciton
    return transfer_processes, transfer_rates


###########################################################################################################
#                               FUNCTION 2: DECAY RATES
###########################################################################################################


def get_decay_rates(centre, system, exciton_index):
    """
    :param centre: index of the excited molecule
    :param system: Dictionary with all the information of the system
    :param exciton_index
    :return: A dictionary with the possible decay rates
    For computing them the method get_decay_rates of class molecule is call.
    """
    donor = system.molecules[centre]

    info = str(hash(donor.state))
    # we define a compact string with the characteristic information of the decays: electronic state

    decay_processes = []            # list of decays processes: dicts(donor, process, acceptor)
    decay_rates = []                # list of the decay rates (numerical values)

    if info in decay_memory:
        decay_complete = decay_memory[info]
        # the decay memory defined is used if the decay have been already computed

    else:
        decay_complete = donor.decay_rates()        # returns a dict {decay_process, decay_rate}
        decay_memory[info] = decay_complete         # saves it if not in memory

    # splits the dictionary in two lists
    for key in decay_complete:
        decay_processes.append({'donor': centre, 'process': key, 'acceptor': centre, 'index': exciton_index})
        decay_rates.append(decay_complete[key])

    # the process include: the index of the donor (in molecules), the key of the process,
    # the index of the acceptor (in molecules) and the index of the exciton (in centres).
    # This last parameter acts as the name of the exciton

    return decay_processes, decay_rates


###########################################################################################################
#                           FUNCTION 3: UPDATE OF THE SYSTEM AND CENTRE INDEXES
###########################################################################################################
def update_step(chosen_process, system):
    """
    :param chosen_process: dictionary like dict(center, process, neighbour)
    Modifies the state of the donor and the acceptor. Removes the donor from the centre_indexes list
    and includes the acceptor. If its a decay only changes and removes the donor

    New if(s) entrances shall be defined for more processes.
    """

    if chosen_process['process'] is 's1_gs':
        system.add_excitation_index('gs', chosen_process['donor'])     # des excitation of the donor
        system.add_excitation_index('s1', chosen_process['acceptor'])  # excitation of the acceptor
        system.molecules[chosen_process['acceptor']].cell_state =- chosen_process['cell_increment']
        # print('here', chosen_process['donor'], chosen_process['acceptor'], chosen_process['cell_increment'])

        system.molecules[chosen_process['donor']].cell_state *= 0

    if chosen_process['process'] == 'Singlet_radiative_decay':
        system.add_excitation_index('gs', chosen_process['donor'])
        system.molecules[chosen_process['donor']].cell_state *= 0


def update_step_old(chosen_process, molecules, centre_indexes):
    """
    :param chosen_process: dictionary like dict(center, process, neighbour)
    :param molecules: list of instances of molecule
    :param centre_indexes: list of the indexes of the excited molecules
    Modifies the state of the donor and the acceptor. Removes the donor from the centre_indexes list
    and includes the acceptor. If its a decay only changes and removes the donor

    New if(s) entrances shall be defined for more processes.
    """
    if chosen_process['process'] is 's1_gs':
        molecules[chosen_process['donor']].set_state('gs')          # des excitation of the donor
        molecules[chosen_process['acceptor']].set_state('s1')       # excitation of the acceptor

        centre_indexes[chosen_process['index']] = chosen_process['acceptor']
        # modification of the excited molecules indexes list.

    if chosen_process['process'] == 'Singlet_radiative_decay':
        molecules[chosen_process['donor']].set_state('gs')          # des excitation of the donor

        centre_indexes[chosen_process['index']] = None
        # modification of the excited molecules indexes list

    # When updating the centre_indexes list, the changed index is the index given by process. So the position with
    # the donor (given by index) is overwritten with the acceptor.
    # If it is a decay, this position, from now on, will be considered as None.

    # No return function. Updates molecules and centre_indexes.


###########################################################################################################
#                           ASSISTANT FUNCTIONS: spectral overlaps
###########################################################################################################


def marcus_fcwd(donor, acceptor, conditions):
    """
    :param donor:
    :param acceptor:
    :param conditions:
    :return: The spectral overlap between the donor and the acceptor according to Marcus formula.
    """
    kb = 8.617333e-5                    # Boltzmann constant in eV * K^(-1)
    T = conditions['temperature']       # temperature (K)

    excited_state = donor.electronic_state()
    gibbs_energy = donor.state_energies[excited_state] - acceptor.state_energies[excited_state]
    # Gibbs energy: energy difference between the equilibrium points of the excited states

    reorganization = acceptor.reorganization_energies[excited_state]
    # acceptor reorganization energy of the excited state

    info = str(hash((T, gibbs_energy, reorganization, 'marcus')))
    # we define a compact string with the characteristic information of the spectral overlap

    if info in overlap_memory:
        # the memory is used if the overlap has been already computed
        overlap = overlap_memory[info]

    else:
        overlap = 1. / (2 * np.sqrt(np.pi*kb*T*reorganization)) * \
                  np.exp(-(gibbs_energy+reorganization)**2 / (4*kb*T*reorganization))

        overlap_memory[info] = overlap
        # new values are added to the memory

    return from_ev_to_au(overlap, 'inverse')
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

    if info in overlap_memory:
        fcwd = overlap_memory[info]

    else:
        fcwd = np.exp(- delta**2 / (2 * deviation) ** 2) / (2 * np.sqrt(np.pi) * deviation)
        overlap_memory[info] = fcwd

    return fcwd


###########################################################################################################
#          Take all the keys of a dictionary that start with the same string
###########################################################################################################


def slice_dict(original_dict, key_string):
    """
    :param original_dict: Original (and larger) dictionary with all entrances
    :param key_string: key string (desired starting string)
    :return: A new dictionary with all the entrances of the original that start with key_string
    """
    newdict = {}

    for key in original_dict:
        if key.startswith(key_string):
            newdict[key] = original_dict[key]

    return newdict


