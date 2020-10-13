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

    # get_all_rates(state, system)
    transfer_processes = get_transfer_rates(state, system)
    decay_processes = get_decay_rates(state, system)

    return decay_processes + transfer_processes


def get_all_rates(state, system):

    total_processes = system.transfer_scheme + system.decay_scheme

    for c, neigh in system.get_state_neighbours(state):
        print(c, neigh)

    #print('neight: ', neigh)
    exit()
    for proc in total_processes:
        initial = proc.initial

        for s in initial:
            unique_mol = set()
            for mol in s.get_molecules():
                for neight in mol:
                    unique_mol.add(neight)

            for mol in s.get_molecules():
                unique_mol.remove(mol)

            print(list(unique_mol))
            # list_mol = [system.get_neighbours(mol) for mol in ]


    exit()
    return None


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
        allowed_processes = get_allowed_processes_new(state, acceptor_state, system.transfer_scheme, cell_incr)
        for process in allowed_processes:
            process.supercell = system.supercell
            transfer_steps.append(process)

    return transfer_steps

def get_transfer_rates_old(state, system):
    """
    :param center: Index of the studies excited molecule
    :param system: Dictionary with the list of molecules and additional physical information
    :return: Two lists, one with the transfer rates and the other with the transfer processes.
    """

    donor = state.get_center()
    neighbour_indexes, cell_increment = system.get_neighbours(donor)
    # states = system.get_state_neighbours(donor.state)

    if False:
        print('**** SYSTEM STATE ****')
        print('donor: ', donor.state.cell_state)
        for i, state in enumerate(system.get_states()):
            print(i, state.label, state, state.cell_state)
        print('****************')

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
        new_process.initial = [state]

        for mol in new_process.initial[0].get_molecules():
            #mol2 = deepcopy(mol)
            #mol2.set_state(new_process.final[0])
            #new_process.final[0].add_molecule(mol2)

            new_process.final_test[0].remove_molecules()
            new_process.final_test[0].add_molecule(mol)

        for mol in new_process.initial[0].get_molecules():
            new_process.cell_states[mol] = mol.cell_state * 0
        #for mol in new_process.final[0].get_molecules():
        #    new_process.cell_states[mol] = mol.cell_state * 0


        print('----------------')
        #print(new_process.initial[0].get_molecules(), new_process.final[0].get_molecules(), new_process.final_test[0].get_molecules())
        new_process.cell_states[new_process.initial[0].get_molecules()[0]] = [0]
        print(new_process.cell_states)
        print('++++++++++++++++')
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

            for final in new_coupling.final:
                final.cell_state = donor.state.cell_state * 0  # TODO: add better looking dimension independent zero set
                for initial in new_coupling.initial:
                    if initial.label == final.label and initial.label != _GS_.label:
                        final.cell_state = initial.cell_state

            # TODO: expand this to a arbitrary number of states
            #print(acceptor, new_coupling.initial[1].get_center())
            #print('------------')
            #new_coupling.initial[0].remove_molecules()
            #new_coupling.initial[0].add_molecule(donor)

            #new_coupling.initial[1].remove_molecules()
            #new_coupling.initial[1].add_molecule(acceptor)

            for initial, final in zip(new_coupling.initial, new_coupling.final):

                #final.remove_molecules()
                for mol in initial.get_molecules():
                    mol2 = deepcopy(mol)
                    mol2.set_state(final)
                    final.add_molecule(mol2)


            # print('initial', new_coupling.final[0].cell_state, new_coupling.final[1].cell_state)

            # acceptor_cell_state = new_coupling.final[1].get_center().cell_state
            # donor_cell_state = new_coupling.final[0].get_center().cell_state

            new_coupling.final[1].get_center().cell_state -= cell_incr
            new_coupling.final[0].get_center().cell_state += cell_incr

            # TODO: To be removed in the future if multiexction systems work
            #if new_coupling.final[0] == _GS_.label:
            #    new_coupling.final[0].get_center().cell_state *= 0

            #if new_coupling.final[1] == _GS_.label:
            #    new_coupling.final[1].get_center().cell_state *= 0

            for state in new_coupling.final:
                if state.label != _GS_.label:
                    state.reorganize_cell()

            # print('final1', new_coupling.final[0].cell_state, new_coupling.final[1].cell_state)
            # print('final2', new_coupling.final[0].get_center().cell_state, new_coupling.final[1].get_center().cell_state)
            # print('final3', new_coupling.final[0].get_center().cell_state_2, new_coupling.final[1].get_center().cell_state_2)
            # print('------------------------------')

            allowed_couplings.append(new_coupling)
    return allowed_couplings


def get_allowed_processes_new(donor_state, acceptor_state, transfer_scheme, cell_incr):
    """
    Get the allowed processes for a given donor and acceptor

    :param donor: Molecule class instance
    :param acceptor: Molecule class instance
    :return: Dictionary with the allowed coupling functions
    """

    allowed_couplings = []
    for coupling in transfer_scheme:
        if (coupling.initial[0].label, coupling.initial[1].label) == (donor_state.label, acceptor_state.label):

            new_coupling = deepcopy(coupling)
            new_coupling.initial = (donor_state, acceptor_state)

            all_logic = False

            if all_logic:
                for final in new_coupling.final:
                    final.cell_state = donor_state.cell_state * 0  # TODO: add better looking dimension independent zero set
                    for initial in new_coupling.initial:
                        if initial.label == final.label and initial.label != _GS_.label:
                            final.cell_state = initial.cell_state

                for initial, final in zip(new_coupling.initial, new_coupling.final):
                    final.remove_molecules()
                    for mol in initial.get_molecules():
                        mol2 = deepcopy(mol)
                        mol2.set_state(final)
                        final.add_molecule(mol2)

                for mol in new_coupling.final[0].get_molecules():
                    mol.cell_state += cell_incr
                for mol in new_coupling.final[1].get_molecules():
                    mol.cell_state -= cell_incr

                print('CS0', new_coupling.final[0].get_center().cell_state)
                print('CS1', new_coupling.final[1].get_center().cell_state)

                for final in new_coupling.final:
                    if final.label != _GS_.label:
                        cell_diff = np.array(np.average([np.array(mol.cell_state) for mol in final.get_molecules()], axis=0),
                                             dtype=int)
                        print('cd2', cell_diff, [np.array(mol.cell_state) for mol in final.get_molecules()])

                        for mol in final.get_molecules():
                            print('cell', np.array(mol.cell_state))
                            mol.cell_state = np.array(mol.cell_state) - cell_diff

                        final.cell_state += cell_diff

                print('CS0f', new_coupling.final[0].get_center().cell_state)
                print('CS1f', new_coupling.final[1].get_center().cell_state)
                print('CS0f_', new_coupling.final[1].cell_state)
                print('CS1f_', new_coupling.final[1].cell_state)
                print('------------')

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

            print('CS0b', new_coupling.cell_states[new_coupling.final_test[0].get_center()])
            print('CS1b', new_coupling.cell_states[new_coupling.final_test[1].get_center()])

            for initial, final in zip(new_coupling.initial, new_coupling.final_test):
                if final.label != _GS_.label:
                    cell_diff = np.array(np.average([new_coupling.cell_states[mol] for mol in final.get_molecules()], axis=0),
                                         dtype=int)

                    print('cd', cell_diff, [new_coupling.cell_states[mol] for mol in final.get_molecules()])
                    for mol in final.get_molecules():
                        print('cell', np.array(new_coupling.cell_states[mol]))
                        new_coupling.cell_states[mol] -= cell_diff

                    final.cell_state += cell_diff

            print('CS0b*', new_coupling.cell_states[new_coupling.final_test[0].get_center()])
            print('CS1b*', new_coupling.cell_states[new_coupling.final_test[1].get_center()])

            if False:
                new_coupling.cell_states = {}
                for state in new_coupling.initial:
                    if state.label != _GS_.label:
                        for mol in state.get_molecules():
                            new_coupling.cell_states[mol] = np.array(mol.cell_state)
                            pass
                    else:
                        for mol in state.get_molecules():
                            new_coupling.cell_states[mol] = np.array(mol.cell_state) * 0
                            pass

            print('CS0b**', new_coupling.cell_states[new_coupling.final_test[0].get_center()])
            print('CS1b**', new_coupling.cell_states[new_coupling.final_test[1].get_center()])

            print('CS0f_', new_coupling.final_test[1].cell_state)
            print('CS1f_', new_coupling.final_test[1].cell_state)
            print('===================')

            if all_logic:
                assert (new_coupling.final[0].cell_state == new_coupling.final_test[0].cell_state).all()
                assert (new_coupling.final[1].cell_state == new_coupling.final_test[1].cell_state).all()
            else:
                print('->', new_coupling.final_test[0].cell_state)
                print('->', new_coupling.final_test[1].cell_state)
                print('->>', new_coupling.cell_states[new_coupling.final_test[0].get_center()])
                print('->>', new_coupling.cell_states[new_coupling.final_test[1].get_center()])

            new_coupling_final = deepcopy(new_coupling.final_test)

            for state in new_coupling_final:
                for mol in state.get_molecules():
                    mol.cell_state = new_coupling.cell_states[mol]
                    mol.set_state(state)

            # recover final from final_test
            if not all_logic:
                new_coupling.final = new_coupling_final

            for s1, s2 in zip(new_coupling.final, new_coupling_final):
                print('s1/s2', s1.cell_state, s2.cell_state)
                for mol1, mol2 in zip(s1.get_molecules(), s2.get_molecules()):
                    print('mol1/mol2', mol1.cell_state, mol2.cell_state)
                assert s1 == s2
            assert new_coupling.final == new_coupling_final

            allowed_couplings.append(new_coupling)
    return allowed_couplings