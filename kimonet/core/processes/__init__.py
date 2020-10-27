from kimonet.core.processes.fcwd import general_fcwd
from kimonet.core.processes.types import GoldenRule, DecayRate, DirectRate
from copy import deepcopy, copy
from kimonet.system.state import ground_state as _GS_
from kimonet.utils.combinations import combinations_group
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


        elements_list = state.get_molecules()
        group_list = [state.size for state in process.final_test]

        configurations = combinations_group(elements_list, group_list, supercell=process.supercell)

        for configuration in configurations:

            new_process = deepcopy(process)
            new_process.initial = [state]

            for molecules, final in zip(configuration, new_process.final_test):
                final.remove_molecules()
                final.cell_state = np.zeros_like(molecules[0].cell_state) # set zero to final state cell_states
                for mol in molecules:
                    new_process.cell_states[mol] = mol.cell_state  # keep molecules with same cell_states
                    final.add_molecule(mol)

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

            # Getting all possible configurations for the final states
            elements_list = [state.get_molecules() for state in (donor_state, acceptor_state)]
            elements_list = [item for sublist in elements_list for item in sublist]
            group_list = [state.size for state in coupling.final_test]

            configurations = combinations_group(elements_list, group_list, supercell=coupling.supercell)

            for configuration in configurations:

                new_coupling = deepcopy(coupling)
                new_coupling.initial = (donor_state, acceptor_state)

                # Binding states
                for final in new_coupling.final_test:
                    final.cell_state = donor_state.cell_state * 0  # TODO: add better looking dimension independent zero set
                    for initial in new_coupling.initial:
                        if initial.label == final.label and initial.label != _GS_.label:
                            final.cell_state = initial.cell_state

                # Binding new molecules
                for molecules, final in zip(configuration, new_coupling.final_test):
                    final.remove_molecules()
                    for mol in molecules:
                        final.add_molecule(mol)


                def is_same_type(state_list1, state_list2):
                    """
                    :param state_list1:
                    :param state_list2:
                    :return:
                    """
                    #for state1, state2 in zip(state_list1, state_list2):
                    #    print(hash(state1), hash(state2))

                    sum1 = np.multiply(*[hash(state) for state in state_list1])
                    sum2 = np.multiply(*[hash(state) for state in state_list2])

                    return sum1 == sum2

                if is_same_type(new_coupling.initial, new_coupling.final_test):
                    continue

                # Define cell positions of molecules in final states
                for mol in new_coupling.final_test[0].get_molecules():
                    new_coupling.cell_states[mol] = np.array(cell_incr)

                for mol in new_coupling.final_test[1].get_molecules():
                    new_coupling.cell_states[mol] = -np.array(cell_incr)

                for initial, final in zip(new_coupling.initial, new_coupling.final_test):
                    if final.label != _GS_.label:
                        cell_diff = np.array(np.average([new_coupling.cell_states[mol] for mol in final.get_molecules()], axis=0),
                                             dtype=int)

                        for mol in final.get_molecules():
                            new_coupling.cell_states[mol] -= cell_diff

                        final.cell_state += cell_diff

                allowed_couplings.append(new_coupling)

    return allowed_couplings