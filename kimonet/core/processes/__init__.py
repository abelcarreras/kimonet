import numpy as np
from kimonet.core.processes.coupling_functions import functions_dict
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

        coupling_functions = allowed_processes(donor, acceptor)

        # for the possible couplings computes the transfer rate by the Fermi's Golden Rule.
        for process, coupling_function in coupling_functions.items():

            e_coupling = coupling_function(donor, acceptor, conditions, system.supercell)

            rate = 2*np.pi * e_coupling**2 * spectral_overlap          # rate in a.u -- Fermi's Golden Rule
            transfer_rates.append(from_ns_to_au(rate, 'direct'))       # rate in ns-1

            transfer_processes.append({'donor': int(centre), 'process': process, 'acceptor': int(neighbour),
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

    if type(chosen_process['process']).__name__ == 'Transfer':

        donor_state = chosen_process['process'].final[0]
        acceptor_state = chosen_process['process'].final[1]

        system.add_excitation_index(donor_state, chosen_process['donor'])  # des excitation of the donor
        system.add_excitation_index(acceptor_state, chosen_process['acceptor'])  # excitation of the acceptor
        system.molecules[chosen_process['acceptor']].cell_state = - chosen_process['cell_increment']

        if chosen_process['process'].final[0] == 'gs':
            system.molecules[chosen_process['donor']].cell_state *= 0

    if type(chosen_process['process']).__name__ == 'Decay':
        final_state = chosen_process['process'].final
        # print('final_state', final_state)
        system.add_excitation_index(final_state, chosen_process['donor'])

        if final_state == 'gs':
            system.molecules[chosen_process['donor']].cell_state *= 0


###########################################################################################################
#                                 Frank-Condon weighted density
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


def allowed_processes(donor, acceptor):
    """
    Get the allowed processes given donor and acceptor

    :param donor: Molecule class instance
    :param acceptor: Molecule class instance
    :return: Dictionary with the allowed coupling functions
    """

    allowed_couplings = {}
    for coupling in functions_dict:
        if coupling.initial == (donor.electronic_state(), acceptor.electronic_state()):
            allowed_couplings[coupling] = functions_dict[coupling]

    return allowed_couplings

