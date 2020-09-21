from kimonet.core.kmc import kmc_algorithm
from kimonet.core.processes import get_processes
from kimonet.core.processes import GoldenRule, DirectRate, DecayRate
from kimonet.system.state import ground_state as _GS_

import warnings


def do_simulation_step(system):
    """
    :param system: system object
    :return: the chosen process
    """

    process_collector = []                          # list with the respective processes (for all centers)
    for state in system.get_states():
        process_collector += get_processes(state, system)

    # If no process available system cannot evolve and simulation is finished
    if len(process_collector) == 0:
        system.is_finished = True
        return None, 0
    chosen_process, time = kmc_algorithm(process_collector, system)

    # chooses one of the processes and gives it a duration using the Kinetic Monte-Carlo algorithm
    update_step(chosen_process, system)        # updates both lists according to the chosen process

    # finally the chosen process and the advanced time are returned
    return chosen_process, time


def update_step(process, system):
    """
    Gets the data in process object and updates the system accordingly

    :param process: instance of Process object containing all the information of the chosen process
    """

    if isinstance(process, (GoldenRule, DirectRate, DecayRate)):

        n_dim = len(process.initial)
        for i in range(n_dim):
            system.remove_exciton(process.initial[i])

            for molecule_t, molecule_o in zip(process.initial[i].get_molecules(), process.final[i].get_molecules()):
                molecule_t.set_state(molecule_o.state)
                molecule_t.cell_state = molecule_o.cell_state

            process.final[i]._molecules_set = process.initial[i].get_molecules()
            system.add_exciton(process.final[i])

        if False:
            print('**** SYSTEM ****')
            for i, mol in enumerate(system.molecules):
                print(i, mol.state.label, mol.state, mol.cell_state)
            print('****************')

    else:
        raise Exception('Process type not recognized')


def system_test_info(system):
    from kimonet.core.processes.fcwd import general_fcwd
    from kimonet.utils.units import HBAR_PLANCK
    from kimonet.utils import distance_vector_periodic
    import numpy as np

    # molecules = system.molecules                # list of all instances of Molecule

    for state in system.get_states():
        center = system.get_molecule_index(state.get_center())
        print('*' * 80 + '\n CENTER {}\n'.format(center) + '*' * 80)

        process_list = get_processes(state, system)
        total_r = 0
        for p in process_list:
            proc = p#['process']
            print('proc: ', proc)
            #exit()
            # print('{}'.format(p))
            #print('Process: {}'.format(proc))
            #print('donor: ', proc.acceptor)
            print()

            i_donor = system.get_molecule_index(proc.initial[0].get_center())
            try:
                i_acceptor = system.get_molecule_index(proc.acceptor)
            except Exception:
                i_acceptor = i_donor

            print('Donor: {} / Acceptor: {}'.format(i_donor, i_acceptor))

            position_d = proc.initial[0].get_center().get_coordinates()
            r = proc.get_rate_constant(system.conditions, system.supercell)

            if isinstance(proc, (GoldenRule, DirectRate)):
                position_a = proc.initial[1].get_center().get_coordinates()

                distance = np.linalg.norm(distance_vector_periodic(position_a - position_d,
                                                                   system.supercell,
                                                                   proc.cell_increment[0]))
                print('Distance: ', distance, 'angs')

            if isinstance(proc, GoldenRule):
                print('Cell_increment: {} '.format(proc.cell_increment))

                spectral_overlap = proc.get_fcwd()

                e_coupling = proc.get_electronic_coupling(system.conditions)

                print('Electronic coupling: ', e_coupling, 'eV')
                print('Spectral overlap:    ', spectral_overlap, 'eV-1')
                # anal_data.append([distance, r])

            print('Rate constant :      ', r, 'ns-1')

            print('-' * 80)
            total_r += r

        print('Total rate sum: {}'.format(total_r))

        # import matplotlib.pyplot as plt
        # plt.scatter(np.array(anal_data).T[0], np.array(anal_data).T[1])
        # plt.show()

