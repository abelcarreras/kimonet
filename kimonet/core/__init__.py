from kimonet.core.kmc import kmc_algorithm
from kimonet.core.processes import get_processes
from kimonet.core.processes import GoldenRule, DirectRate, DecayRate
from kimonet.utils import distance_vector_periodic
import numpy as np

import warnings


def do_simulation_step(system):
    """
    :param system: system object
    :return: the chosen process and time
    """

    if False:
        print('**** SYSTEM STATE INITIAL ****')
        for i, mol in enumerate(system.molecules):
            print(i, mol.state.label, mol.state, mol.state.cell_state, mol.name, mol)
        print('****************')

    process_collector = []                          # list with the respective processes (for all centers)
    for state in system.get_states():
        process_collector += get_processes(state, system)

    # If no process available system cannot evolve and simulation is finished
    if len(process_collector) == 0:
        system.is_finished = True
        return None, 0
    chosen_process, time = kmc_algorithm(process_collector, system)

    system.update(chosen_process)

    if False:
        print('**** SYSTEM STATE FINAL ****')
        for i, mol in enumerate(system.molecules):
            print(i, mol.state.label, mol.state, mol.state.cell_state, mol.name, mol)
        print('****************')

    return chosen_process, time


def system_test_info(system):
    for state in system.get_states():
        center = system.get_molecule_index(state.get_center())
        print('*' * 80 + '\n CENTER {}\n'.format(center) + '*' * 80)

        process_list = get_processes(state, system)
        total_r = 0
        for proc in process_list:

            i_donor = system.get_molecule_index(proc.initial[0].get_center())
            try:
                i_acceptor = system.get_molecule_index(proc.acceptor)
            except Exception:
                i_acceptor = i_donor

            print('Description:', proc.description)
            print('Donor: {} / Acceptor: {}'.format(i_donor, i_acceptor))

            position_d = proc.initial[0].get_center().get_coordinates()
            r = proc.get_rate_constant(system.conditions, system.supercell)

            # if isinstance(proc, (GoldenRule, DirectRate)):
            if len(proc.initial) == 2:
                position_a = proc.initial[1].get_center().get_coordinates()

                cell_increment = np.array(proc.final[0].get_center().cell_state) - np.array(proc.initial[1].get_center().cell_state)

                distance = np.linalg.norm(distance_vector_periodic(position_a - position_d,
                                                                   system.supercell,
                                                                   cell_increment))
                print('Distance: ', distance, 'angs')
                print('Cell_increment: {} '.format(cell_increment))

            if isinstance(proc, GoldenRule):

                spectral_overlap = proc.get_fcwd()

                e_coupling = proc.get_electronic_coupling()

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

