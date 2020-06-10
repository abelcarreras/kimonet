import numpy as np
from kimonet.core.processes.fcwd import general_fcwd
from kimonet.utils.units import HBAR_PLANCK
from kimonet.core.processes.types import GoldenRule, DecayRate, DirectRate


def get_processes_and_rates(centre, system):
    """
    :param centre: Index of the studied excited molecule (Donor)
    :param system: Instance of System class
    Computes the transfer and decay rates and builds two dictionaries:
            One with the decay process as key and its rate as argument
            One with the transferred molecule index as key and {'process': rate} as argument

    :return:    process_list: List of processes in named tuple format
                rate_list: List with the respective rates
    """

    transfer_processes, transfer_rates = get_transfer_rates(centre, system)

    decay_processes, decay_rates = get_decay_rates(centre, system)

    # merges all processes & rates
    process_list = decay_processes + transfer_processes
    rate_list = decay_rates + transfer_rates

    return process_list, rate_list


def get_transfer_rates(center, system):
    """
    :param center: Index of the studies excited molecule
    :param system: Dictionary with the list of molecules and additional physical information
    :return: Two lists, one with the transfer rates and the other with the transfer processes.
    """

    neighbour_indexes, cell_increment = system.get_neighbours(center)

    conditions = system.conditions           # physical conditions of the system

    donor = system.molecules[center]         # excited molecule

    transfer_rates = []                         # list that collects the transfer rates (only the numerical values)
    transfer_processes = []                     # list that collects the transfer processes dict(donor,process,acceptor)

    for neighbour, cell_incr in zip(neighbour_indexes, cell_increment):
        acceptor = system.molecules[neighbour]

        allowed_processes = get_allowed_processes(donor, acceptor, system.transfer_scheme)

        for process in allowed_processes:

            transfer_rates.append(process.get_rate_constant(donor, acceptor, conditions, system.supercell, cell_incr))
            transfer_processes.append({'donor': int(center), 'process': process, 'acceptor': int(neighbour),
                                       'cell_increment': cell_incr})

    return transfer_processes, transfer_rates


def get_decay_rates(center, system):
    """
    :param center: index of the excited molecule
    :param system: Dictionary with all the information of the system
    :return: A dictionary with the possible decay rates
    For computing them the method get_decay_rates of class molecule is call.
    """
    donor = system.molecules[center]

    decay_processes = []            # list of decays processes: dicts(donor, process, acceptor)
    decay_rates = []                # list of the decay rates (numerical values)

    decay_complete = donor.decay_rates()        # returns a dict {decay_process, decay_rate}

    # splits the dictionary in two lists
    for key in decay_complete:
        decay_processes.append({'donor': center, 'process': key, 'acceptor': center})
        decay_rates.append(decay_complete[key])

    return decay_processes, decay_rates


def get_allowed_processes(donor, acceptor, transfer_scheme):
    """
    Get the allowed processes for a given donor and acceptor

    :param donor: Molecule class instance
    :param acceptor: Molecule class instance
    :return: Dictionary with the allowed coupling functions
    """

    allowed_couplings = []
    for coupling in transfer_scheme:
        if coupling.initial == (donor.electronic_state(), acceptor.electronic_state()):
            allowed_couplings.append(coupling)

    return allowed_couplings
