from kimonet.core.kmc import kmc_algorithm
from kimonet.core.processes import get_processes
from kimonet.core.processes.types import GoldenRule, DirectRate, DecayRate
import numpy as np


def do_simulation_step(system):
    """
    :param system: system object
    :return: the chosen process and time
    """

    if False:
        print('**** SYSTEM STATE INITIAL ****')
        system.print_status()
        print('****************')

    process_collector = []                          # list with the respective processes (for all centers)
    for state in system.get_states():
        process_collector += get_processes(state, system)

    # If no process available system cannot evolve and simulation is finished
    if len(process_collector) == 0:
        system.is_finished = True
        return None, 0
    chosen_process, time = kmc_algorithm(process_collector)

    system.update(chosen_process)

    if False:
        print('**** SYSTEM STATE FINAL ****')
        system.print_status()
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
                i_acceptor = system.get_molecule_index(proc.initial[1].get_center())
            except Exception:
                i_acceptor = i_donor

            print('Description: {}'.format(proc.description))
            print('Donor: {} / Acceptor: {}'.format(i_donor, i_acceptor))

#            position_d = proc.initial[0].get_center().get_coordinates()
            r = proc.get_rate_constant()

            # if isinstance(proc, (GoldenRule, DirectRate)):
            if len(proc.initial) == 2:
                cell_increment = proc.initial_safe[1].cell_state - proc.initial_safe[0].cell_state
                distance = np.linalg.norm([proc.initial_safe[1].get_coordinates_absolute() - proc.initial_safe[0].get_coordinates_absolute()])
                print('Distance: {:.4} angs'.format(distance))
                print('Cell_increment: {} '.format(cell_increment))

            if isinstance(proc, GoldenRule):

                spectral_overlap = proc.get_fcwd()

                e_coupling = proc.get_electronic_coupling()

                print('Electronic coupling: {:.4} eV'.format(e_coupling))
                print('Spectral overlap:    {:.4} eV-1'.format(spectral_overlap))
                # anal_data.append([distance, r])

            print('Rate constant: {:.4} ns-1'.format(r))

            print('-' * 80)
            total_r += r

        print('Total rate sum: {:.4}'.format(float(total_r)))

        # import matplotlib.pyplot as plt
        # plt.scatter(np.array(anal_data).T[0], np.array(anal_data).T[1])
        # plt.show()


def local_diffusion_model(system):

    total_centers = []
    for state in system.get_states():
        center = system.get_molecule_index(state.get_center())
        #print('*' * 80 + '\n CENTER {}\n'.format(center) + '*' * 80)

        process_list = get_processes(state, system)

        diffusion_matrix_list = []

        for proc in process_list:

            k = proc.get_rate_constant()

            # if isinstance(proc, (GoldenRule, DirectRate)):
            if len(proc.initial) == 2:
                #cell_increment = proc.initial_absolute[1].cell_state - proc.initial_absolute[0].cell_state
                #distance = np.linalg.norm([proc.initial_absolute[1].get_coordinates_absolute() - proc.initial_absolute[0].get_coordinates_absolute()])
                r_vec = proc.initial_safe[1].get_coordinates_absolute() - proc.initial_safe[0].get_coordinates_absolute()

                #print(r_vec)
                #print('r_vector: {}'.format(r_vec))
                #print('Distance: {:.4} angs'.format(distance))
                #print('Cell_increment: {} '.format(cell_increment))
                #print('Rate constant: {:.4} ns-1'.format(k))

                diffusion_matrix_list.append(np.outer(r_vec, r_vec) * k)
                #continue

                #diffusion_matrix.append(np.outer(r_vec, r_vec) * k)

            #print('-' * 80)

        diffusion_matrix = np.sum(diffusion_matrix_list, axis=0)/2

        coeff = np.mean(np.diag(diffusion_matrix))
        # print('matrix\n', diffusion_matrix)
        # print('coef: ', np.mean(np.diag(diffusion_matrix)))

        total_centers.append({'diff_coef': coeff,
                              'diff_mat': diffusion_matrix})

    return total_centers
