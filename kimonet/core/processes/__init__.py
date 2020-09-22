from kimonet.core.processes.fcwd import general_fcwd
from kimonet.core.processes.types import GoldenRule, DecayRate, DirectRate
from copy import deepcopy
from kimonet.system.state import ground_state as _GS_


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

    transfer_steps = []
    for acceptor, cell_incr in zip(neighbour_indexes, cell_increment):
        allowed_processes = get_allowed_processes(donor, acceptor, system.transfer_scheme, cell_incr)

        for process in allowed_processes:
            process.supercell = system.supercell
            transfer_steps.append(process)

    return transfer_steps


def get_decay_rates(state, system):
    """
    :param center: index of the excited molecule
    :param system: Dictionary with all the information of the system
    :return: A dictionary with the possible decay rates
    For computing them the method get_decay_rates of class molecule is call.
    """

    decay_complete = system.decay_scheme

    decay_steps = []
    for process in decay_complete:
        new_process = deepcopy(process)
        new_process.initial = (state,)
        decay_steps.append(new_process)

    return decay_steps


def get_allowed_processes(donor, acceptor, transfer_scheme, cell_incr):
    """
    Get the allowed processes for a given donor and acceptor

    :param donor: Molecule class instance
    :param acceptor: Molecule class instance
    :return: Dictionary with the allowed coupling functions
    """

    allowed_couplings = []
    for coupling in transfer_scheme:
        if (coupling.initial[0].label, coupling.initial[1].label) == (donor.state.label, acceptor.state.label):
            new_coupling = deepcopy(coupling)

            new_coupling.initial = (donor.state, acceptor.state)

            new_coupling.initial[0].remove_molecules()
            new_coupling.initial[1].remove_molecules()
            new_coupling.initial[0].add_molecule(donor)
            new_coupling.initial[1].add_molecule(acceptor)

            for i in range(2):
                new_coupling.final[i].remove_molecules()
                for mol in new_coupling.initial[i].get_molecules():
                    mol2 = deepcopy(mol)
                    mol2.set_state(new_coupling.final[i])
                    new_coupling.final[i].add_molecule(mol2)

            acceptor_cell_state = new_coupling.final[1].get_center().cell_state
            donor_cell_state = new_coupling.final[0].get_center().cell_state

            new_coupling.final[1].get_center().cell_state = donor_cell_state - cell_incr
            new_coupling.final[0].get_center().cell_state = acceptor_cell_state + cell_incr

            if new_coupling.final[0] == _GS_.label:
                new_coupling.final[0].get_center().cell_state *= 0

            if new_coupling.final[1] == _GS_.label:
                new_coupling.final[1].get_center().cell_state *= 0

            allowed_couplings.append(new_coupling)

    return allowed_couplings
