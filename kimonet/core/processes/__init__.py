from kimonet.core.processes.fcwd import general_fcwd
from kimonet.core.processes.types import GoldenRule, DecayRate, DirectRate
from copy import deepcopy


def get_processes(state, system):
    """
    :param state: excited state for which find possible processes
    :param system: Instance of System class
    Computes the transfer and decay rates and builds two dictionaries:
            One with the decay process as key and its rate as argument
            One with the transferred molecule index as key and {'process': rate} as argument

    :return:    process_list: List of processes in named tuple format
                rate_list: List with the respective rates
    """

    transfer_processes = get_transfer_rates(state, system)
    decay_processes = get_decay_rates(state, system)

    return decay_processes + transfer_processes


def get_transfer_rates(state, system):
    """
    :param center: Index of the studies excited molecule
    :param system: Dictionary with the list of molecules and additional physical information
    :return: Two lists, one with the transfer rates and the other with the transfer processes.
    """

    donor = state.get_center()
    neighbour_indexes, cell_increment = system.get_neighbours(donor)
    # origin = system.get_molecule_index(donor)

    transfer_steps = []
    for acceptor, cell_incr in zip(neighbour_indexes, cell_increment):
        allowed_processes = get_allowed_processes(donor, acceptor, system.transfer_scheme)

        for process in allowed_processes:
            # I don't like this very much
            process.cell_increment = cell_incr  # TODO: cell_incr should be independent by molecule (dict?)
            process.supercell = system.supercell
            # target = system.get_molecule_index(acceptor)

            #transfer_steps.append({'donor': int(origin), 'process': process, 'acceptor': int(target),
            #                       'cell_increment': cell_incr})
            transfer_steps.append(process)

    #for tr in transfer_steps:
    #    print(system.get_molecule_index(tr.donor), system.get_molecule_index(tr.acceptor), tr.cell_increment, tr.supercell)

    return transfer_steps


def get_decay_rates(state, system):
    """
    :param center: index of the excited molecule
    :param system: Dictionary with all the information of the system
    :return: A dictionary with the possible decay rates
    For computing them the method get_decay_rates of class molecule is call.
    """

    donor = state.get_center()
    # origin = system.get_molecule_index(donor)

    decay_complete = donor.decay_rates()        # returns a dict {decay_process, decay_rate}

    decay_steps = []
    for process in decay_complete:
        new_process = deepcopy(process)
        new_process.donor = donor
        #new_process.acceptor = donor

        #decay_steps.append({'donor': origin, 'process': new_process, 'acceptor': origin})
        decay_steps.append(new_process)

    return decay_steps


def get_allowed_processes(donor, acceptor, transfer_scheme):
    """
    Get the allowed processes for a given donor and acceptor

    :param donor: Molecule class instance
    :param acceptor: Molecule class instance
    :return: Dictionary with the allowed coupling functions
    """
    # print('allowedTrans', donor.state, acceptor.state)

    allowed_couplings = []
    for coupling in transfer_scheme:
        if (coupling.initial[0].label, coupling.initial[1].label) == (donor.state.label, acceptor.state.label):
            new_coupling = deepcopy(coupling)
            new_coupling.donor = donor
            new_coupling.acceptor = acceptor

            #print('initial: ', new_coupling.initial[0].get_molecules(), new_coupling.initial[1].get_molecules(),
            #      new_coupling.final[0].get_molecules(), new_coupling.final[1].get_molecules())

            if True:
                new_coupling.initial = (donor.state, acceptor.state)

                new_coupling.initial[0].remove_molecules()
                new_coupling.initial[1].remove_molecules()

                new_coupling.final[0].remove_molecules()
                new_coupling.final[1].remove_molecules()

                new_coupling.initial[0].add_molecule(donor)
                new_coupling.initial[1].add_molecule(acceptor)

                new_coupling.final[0].add_molecule(donor)
                new_coupling.final[1].add_molecule(acceptor)

            allowed_couplings.append(new_coupling)

    return allowed_couplings
