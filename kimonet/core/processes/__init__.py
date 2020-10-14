from kimonet.core.processes.fcwd import general_fcwd
from kimonet.core.processes.types import GoldenRule, DecayRate, DirectRate
from copy import deepcopy
from kimonet.system.state import ground_state as _GS_
import numpy as np


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

    neighbors, cell_increments = system.get_state_neighbors(state)

    if False:
        print('**** SYSTEM STATE ****')
        print('donor: ', donor.state.cell_state)
        for i, state in enumerate(system.get_states()):
            print(i, state.label, state, state.cell_state)
        print('****************')

    transfer_steps = []
    for acceptor_state, cell_incr in zip(neighbors, cell_increments):
        allowed_processes = get_allowed_processes(state, acceptor_state, system.transfer_scheme, cell_incr)
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
        new_process.initial = [state]

        for mol in new_process.initial[0].get_molecules():
            new_process.final_test[0].remove_molecules()
            new_process.final_test[0].add_molecule(mol)

        for mol in new_process.initial[0].get_molecules():
            new_process.cell_states[mol] = np.zeros(mol.get_dim())

        new_process.reset_cell_states()

        decay_steps.append(new_process)

    return decay_steps


def get_allowed_processes(donor_state, acceptor_state, transfer_scheme, cell_incr):
    """
    Get the allowed processes for a given donor and acceptor

    :param donor_state: State instance containing the donor state
    :param acceptor_state: State instance containing the acceptor state
    :param cell_incr: List with the cell_state increments for the given acceptor (diff between acceptor and donor cell states)
    :return: Dictionary with the allowed coupling functions
    """

    allowed_couplings = []
    for coupling in transfer_scheme:
        if (coupling.initial[0].label, coupling.initial[1].label) == (donor_state.label, acceptor_state.label):

            new_coupling = deepcopy(coupling)
            new_coupling.initial = (donor_state, acceptor_state)

            # Binding states
            for final in new_coupling.final_test:
                final.cell_state = donor_state.cell_state * 0  # TODO: add better looking dimension independent zero set
                for initial in new_coupling.initial:
                    if initial.label == final.label and initial.label != _GS_.label:
                        final.cell_state = initial.cell_state

            # Binding new molecules
            for initial, final in zip(new_coupling.initial, new_coupling.final_test):
                final.remove_molecules()
                for mol in initial.get_molecules():
                    final.add_molecule(mol)

            for mol in new_coupling.final_test[0].get_molecules():
                new_coupling.cell_states[mol] = np.array(cell_incr)

            for mol in new_coupling.final_test[1].get_molecules():
                new_coupling.cell_states[mol] = -np.array(cell_incr)

            #print('CS0b', new_coupling.cell_states[new_coupling.final_test[0].get_center()])
            #print('CS1b', new_coupling.cell_states[new_coupling.final_test[1].get_center()])

            for initial, final in zip(new_coupling.initial, new_coupling.final_test):
                if final.label != _GS_.label:
                    cell_diff = np.array(np.average([new_coupling.cell_states[mol] for mol in final.get_molecules()], axis=0),
                                         dtype=int)

                    #print('cd', cell_diff, [new_coupling.cell_states[mol] for mol in final.get_molecules()])
                    for mol in final.get_molecules():
                        #print('cell', np.array(new_coupling.cell_states[mol]))
                        new_coupling.cell_states[mol] -= cell_diff

                    final.cell_state += cell_diff

            allowed_couplings.append(new_coupling)

    return allowed_couplings