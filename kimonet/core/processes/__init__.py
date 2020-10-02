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

    donor = state.get_center()
    neighbour_indexes, cell_increment = system.get_neighbours(donor)

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

            for final in new_coupling.final:
                final.cell_state = donor.state.cell_state * 0  # TODO: add better looking dimension independent zero set
                for initial in new_coupling.initial:
                    if initial.label == final.label and initial.label != _GS_.label:
                        final.cell_state = initial.cell_state

            # TODO: expand this to a arbitrary number of states
            new_coupling.initial[0].remove_molecules()
            new_coupling.initial[0].add_molecule(donor)

            new_coupling.initial[1].remove_molecules()
            new_coupling.initial[1].add_molecule(acceptor)

            for initial, final in zip(new_coupling.initial, new_coupling.final):
                final.remove_molecules()
                for mol in initial.get_molecules():
                    mol2 = deepcopy(mol)
                    mol2.set_state(final)
                    final.add_molecule(mol2)


            # print('initial', new_coupling.final[0].cell_state, new_coupling.final[1].cell_state)

            acceptor_cell_state = new_coupling.final[1].get_center().cell_state
            donor_cell_state = new_coupling.final[0].get_center().cell_state

            new_coupling.final[1].get_center().cell_state = donor_cell_state - cell_incr
            new_coupling.final[0].get_center().cell_state = acceptor_cell_state + cell_incr

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
