import numpy as np
from kimonet.core.processes.fcwd import marcus_fcwd
from collections import namedtuple


Transfer = namedtuple("Transfer", ["initial", "final", "description"])
Decay = namedtuple("Decay", ["initial", "final", "description"])


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

        # compute the spectral overlap using Marcus formula
        spectral_overlap = marcus_fcwd(donor, acceptor, conditions)

        allowed_processes = get_allowed_processes(donor, acceptor)

        for process, coupling_function in allowed_processes.items():

            e_coupling = coupling_function(donor, acceptor, conditions, system.supercell)
            transfer_rates.append(2*np.pi * e_coupling**2 * spectral_overlap)  # Fermi's Golden Rule

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


def get_allowed_processes(donor, acceptor):
    """
    Get the allowed processes for a given donor and acceptor

    :param donor: Molecule class instance
    :param acceptor: Molecule class instance
    :return: Dictionary with the allowed coupling functions
    """

    allowed_couplings = {}
    for coupling in transfer_scheme:
        if coupling.initial == (donor.electronic_state(), acceptor.electronic_state()):
            allowed_couplings[coupling] = transfer_scheme[coupling]

    return allowed_couplings


# Transfer tuple format:

transfer_scheme = {# Transfer(initial=('s1', 'gs'), final=('gs', 's1'), description='forster'): forster_coupling,
          # Transfer(initial=('s1', 'gs'), final=('gs', 's2'), description='test'): compute_forster_coupling,
          # Transfer(initial=('s2', 'gs'), final=('gs', 's1'), description='test2'): compute_forster_coupling,
          # Transfer(initial=('s2', 'gs'), final=('gs', 's2'), description='test3'): compute_forster_coupling
          }
