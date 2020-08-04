from kimonet.core.processes.fcwd import general_fcwd
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

    transfer_processes = get_transfer_rates(centre, system)
    decay_processes = get_decay_rates(centre, system)

    return decay_processes + transfer_processes


def get_transfer_rates(center, system):
    """
    :param center: Index of the studies excited molecule
    :param system: Dictionary with the list of molecules and additional physical information
    :return: Two lists, one with the transfer rates and the other with the transfer processes.
    """

    neighbour_indexes, cell_increment = system.get_neighbours(center)

    donor = system.molecules[center]

    transfer_steps = []
    for neighbour, cell_incr in zip(neighbour_indexes, cell_increment):
        acceptor = system.molecules[neighbour]
        allowed_processes = get_allowed_processes(donor, acceptor, system.transfer_scheme)

        for process in allowed_processes:
            # I don't like this very much
            process.cell_increment = cell_incr
            process.supercell = system.supercell

            transfer_steps.append({'donor': int(center), 'process': process, 'acceptor': int(neighbour),
                                   'cell_increment': cell_incr})

    return transfer_steps


def get_decay_rates(center, system):
    """
    :param center: index of the excited molecule
    :param system: Dictionary with all the information of the system
    :return: A dictionary with the possible decay rates
    For computing them the method get_decay_rates of class molecule is call.
    """
    from copy import deepcopy

    donor = system.molecules[center]

    decay_complete = donor.decay_rates()        # returns a dict {decay_process, decay_rate}

    decay_steps = []
    for process in decay_complete:
        new_process = deepcopy(process)
        new_process.donor = donor
        #new_process.acceptor = donor

        decay_steps.append({'donor': center, 'process': new_process, 'acceptor': center})

    return decay_steps


def get_allowed_processes(donor, acceptor, transfer_scheme):
    """
    Get the allowed processes for a given donor and acceptor

    :param donor: Molecule class instance
    :param acceptor: Molecule class instance
    :return: Dictionary with the allowed coupling functions
    """
    from copy import deepcopy

    allowed_couplings = []
    for coupling in transfer_scheme:
        if (coupling.initial[0].label, coupling.initial[1].label) == (donor.state.label, acceptor.state.label):
            new_coupling = deepcopy(coupling)
            new_coupling.donor = donor
            new_coupling.acceptor = acceptor
            allowed_couplings.append(new_coupling)

    return allowed_couplings
