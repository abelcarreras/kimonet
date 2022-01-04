from copy import deepcopy
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
        print('**** SYSTEM STATE INTER ****')
        for i, mol in enumerate(system.molecules):
            print(i, mol.state.label, mol.state, mol.state.cell_state, mol.name, mol)
        print('----')
        for state in system.get_states():
            print(state.label, state, state.supercell)
        print('****************')

    transfer_steps = []
    for acceptor_state, cell_incr in zip(neighbors, cell_increments):
        allowed_processes = get_allowed_processes(state, acceptor_state, system.transfer_scheme, cell_incr)
        for process in allowed_processes:
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

        if process.initial[0].label == state.label:

            elements_list = state.get_molecules()
            # group_list = [s.size for s in process.final_test]

            configurations = combinations_group(elements_list, process.final, supercell=process.supercell)

            for configuration in configurations:

                new_process = deepcopy(process)
                new_process.initial = (state,)

                for molecules, final in zip(configuration, new_process.final):
                    final.remove_molecules()
                    final.cell_state = np.zeros_like(molecules[0].cell_state) # set zero to final state cell_states
                    for mol in molecules:
                        new_process.cell_states[mol] = mol.cell_state  # keep molecules with same cell_states
                        final.add_molecule(mol)

                new_process.reset_cell_states()

                # print(new_process.initial[0].label, '--', new_process.final[0].label)
                decay_steps.append(new_process)

    return decay_steps


def get_allowed_processes(donor_state, acceptor_state, transfer_scheme, cell_incr):
    """
    Get the allowed processes for a given donor and acceptor

    :param donor_state: State instance containing the donor state
    :param acceptor_state: State instance containing the acceptor state
    :param cell_incr: List with the cell_state increments for the given acceptor (diff between acceptor and donor cell states)
    :return: Dictionary with the allowed process functions
    """

    allowed_processes = []
    for process in transfer_scheme:

        # handle self interaction
        if donor_state == acceptor_state:
            process = process.get_self_interaction_process()
            if process is None:
                continue

        if (process.initial[0].label, process.initial[1].label) == (donor_state.label, acceptor_state.label):

            def is_same_p_configuration(p_configuration_1, p_configuration_2):
                if len(p_configuration_1) != len(p_configuration_2):
                    return False

                sum1 = np.multiply(*[hash(state) for state in p_configuration_1])
                sum2 = np.multiply(*[hash(state) for state in p_configuration_2])

                return sum1 == sum2

            include_self = not is_same_p_configuration(process.initial, process.final)

            # Getting all possible configurations for the final states
            elements_list = [state.get_molecules() for state in (donor_state, acceptor_state)]
            elements_list = [item for sublist in elements_list for item in sublist]
            # group_list = [state.size for state in process.final_test]

            configurations = combinations_group(elements_list, process.final, supercell=process.supercell, include_self=include_self)

            for configuration in configurations:

                new_process = deepcopy(process)
                new_process.initial = (donor_state, acceptor_state)
                new_process.set_cell_increment(cell_incr)

                # Binding final states to initial states if equal
                #for final in new_process.final_test:
                #    final.cell_state = np.zeros_like(donor_state.cell_state)
                #    for initial in new_process.initial:
                #        if initial.label == final.label and initial.label != _GS_.label:
                #            print(final, initial)
                #            final.cell_state = initial.cell_state

                for final in new_process.final:
                    final.cell_state = np.zeros_like(donor_state.cell_state)
                for initial, final_list in new_process.get_transport_connections().items():
                    for final in final_list:
                        final.cell_state = initial.cell_state

                # Binding molecules to states
                for molecules, final in zip(configuration, new_process.final):
                    final.remove_molecules()
                    for mol in molecules:
                        final.add_molecule(mol)

                # redefine cell states for molecules in the process
                for mol in new_process.get_molecules():
                    new_process.cell_states[mol] = np.array(mol.cell_state)
                    if mol in new_process.initial[0].get_molecules():
                        new_process.cell_states[mol] += np.array(cell_incr)
                    elif mol in new_process.initial[1].get_molecules():
                        new_process.cell_states[mol] -= np.array(cell_incr)
                    else:
                        raise Exception('molecules from initial and final do not match')

                # redefine state cell_state from molecules cell_state
                for initial, final in zip(new_process.initial, new_process.final):
                    if final.label != _GS_.label:
                        cell_diff = np.array(np.average([new_process.cell_states[mol] for mol in final.get_molecules()], axis=0),
                                             dtype=int)

                        for mol in final.get_molecules():
                            new_process.cell_states[mol] -= cell_diff

                        final.cell_state += cell_diff

                allowed_processes.append(new_process)

    return allowed_processes
