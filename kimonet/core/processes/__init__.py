import numpy as np
from kimonet.core.processes.coupling import functions_dict
from kimonet.core.processes.fcwd import marcus_fcwd

decay_data = {}


def get_processes_and_rates(centre, system, i):
    """
    :param i: index of the exciton
    :param centre: Index of the studied excited molecule (Donor)
    :param system: Instance of System class
    Computes the transfer and decay rates and builds two dictionaries:
            One with the decay process as key and its rate as argument
            One with the transferred molecule index as key and {'process': rate} as argument

    :return:    process_list: List of elements like dict(center, process, new molecule)
                rate_list: List with the respective rates
                The list indexes coincide.
    """

    transfer_processes, transfer_rates = get_transfer_rates(centre, system, i)
    # calls an external function that computes the transfer rates for all possible transfer processes between
    # the centre and all its neighbours

    decay_processes, decay_rates = get_decay_rates(centre, system, i)
    # calls an external function that computes the decay rates for all possible decay processes of the centre.
    # Uses a method of the class Molecule

    # merges all processes in a list and the same for the rates
    # the indexes of the list must coincide (same length. rate 'i' is related to process 'i')
    process_list = decay_processes + transfer_processes
    rate_list = decay_rates + transfer_rates

    return process_list, rate_list


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

        # compute the spectral overlap using Marcus formula
        spectral_overlap = marcus_fcwd(donor, acceptor, conditions)

        allowed_processes = get_allowed_processes(donor, acceptor)

        for process, coupling_function in allowed_processes.items():

            e_coupling = coupling_function(donor, acceptor, conditions, system.supercell)
            transfer_rates.append(2*np.pi * e_coupling**2 * spectral_overlap)  # Fermi's Golden Rule

            transfer_processes.append({'donor': int(centre), 'process': process, 'acceptor': int(neighbour),
                                       'index': exciton_index, 'cell_increment': cell_incr})

    return transfer_processes, transfer_rates


def get_decay_rates(centre, system, exciton_index):
    """
    :param centre: index of the excited molecule
    :param system: Dictionary with all the information of the system
    :param exciton_index
    :return: A dictionary with the possible decay rates
    For computing them the method get_decay_rates of class molecule is call.
    """
    donor = system.molecules[centre]

    decay_processes = []            # list of decays processes: dicts(donor, process, acceptor)
    decay_rates = []                # list of the decay rates (numerical values)

    decay_complete = donor.decay_rates()        # returns a dict {decay_process, decay_rate}

    # splits the dictionary in two lists
    for key in decay_complete:
        decay_processes.append({'donor': centre, 'process': key, 'acceptor': centre, 'index': exciton_index})
        decay_rates.append(decay_complete[key])

    # the process include: the index of the donor (in molecules), the key of the process,
    # the index of the acceptor (in molecules) and the index of the exciton (in centres).
    # This last parameter acts as the name of the exciton

    return decay_processes, decay_rates


def get_allowed_processes(donor, acceptor):
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

