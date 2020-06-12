from kimonet.core.kmc import kmc_algorithm
from kimonet.core.processes import get_processes_and_rates
from kimonet.core.processes import GoldenRule, DirectRate, DecayRate
from kimonet import _ground_state_

import warnings


def do_simulation_step(system):
    """
    :param system: Dictionary with all the information of the system
    Dictionary system already has the indexes of the excited molecules
    1. Looks for the neighbourhood of every center.
    2. Chooses a process for every exciton (KMC). This path variables include: dict(center, process, new molecule)
    3. Considering all the calculated rates computes the time interval for each process.
    4. Updates the system according to the chosen path and the time passed.
    :return: the chosen process and the advanced time
    """

    molecules = system.molecules                # list of all instances of Molecule
    centre_indexes = system.centers              # tricky list with the indexes of all excited molecules

    rate_collector = []                             # list with all the rates (for all centers)
    process_collector = []                          # list with the respective processes (for all centers)
    # the indexes of both lists coincide.

    for center in system.centers:
        if type(center) == int:
            # looks for the all molecules in a circle of radius centered at the position of the excited molecule

            process_list, rate_list = get_processes_and_rates(center, system)
            # for each center computes all the decay rates and all the transfer rates for all neighbours
            # return them as a list

            rate_collector += rate_list
            process_collector += process_list
            # merging of the new list with the rates and processes previously computed

    # If no process available system cannot evolve and simulation is finished
    if len(process_collector) == 0:
        system.is_finished = True
        return None, 0
    chosen_process, time = kmc_algorithm(rate_collector, process_collector)
    # chooses one of the processes and gives it a duration using the Kinetic Monte-Carlo algorithm
    update_step(chosen_process, system)        # updates both lists according to the chosen process

    # finally the chosen process and the advanced time are returned
    return chosen_process, time


def update_step(chosen_process, system):
    """
    :param chosen_process: dictionary like dict(center, process, neighbour)
    Modifies the state of the donor and the acceptor. Removes the donor from the centre_indexes list
    and includes the acceptor. If its a decay only changes and removes the donor

    New if(s) entrances shall be defined for more processes.
    """

    if type(chosen_process['process']) in (GoldenRule, DirectRate):

        donor_state = chosen_process['process'].final[0]
        acceptor_state = chosen_process['process'].final[1]

        system.add_excitation_index(donor_state, chosen_process['donor'])  # des excitation of the donor
        system.add_excitation_index(acceptor_state, chosen_process['acceptor'])  # excitation of the acceptor

        # cell state assumes symmetric states cross: acceptor -> donor & donor -> acceptor
        acceptor_cell_state = system.molecules[chosen_process['acceptor']].cell_state
        donor_cell_state = system.molecules[chosen_process['donor']].cell_state
        system.molecules[chosen_process['acceptor']].cell_state = donor_cell_state - chosen_process['cell_increment']
        system.molecules[chosen_process['donor']].cell_state = acceptor_cell_state + chosen_process['cell_increment']

        # system.molecules[chosen_process['donor']].cell_state *= 0

        if chosen_process['process'].final[0] == _ground_state_:
            system.molecules[chosen_process['donor']].cell_state *= 0

        if chosen_process['process'].final[1] == _ground_state_:
            system.molecules[chosen_process['acceptor']].cell_state *= 0

    if type(chosen_process['process']) == DecayRate:
        final_state = chosen_process['process'].final
        # print('final_state', final_state)
        system.add_excitation_index(final_state, chosen_process['donor'])

        if final_state == _ground_state_:
            system.molecules[chosen_process['donor']].cell_state *= 0


def system_test_info(system):
    from kimonet.core.processes.fcwd import general_fcwd
    from kimonet.utils.units import HBAR_PLANCK
    from kimonet.utils import distance_vector_periodic
    import numpy as np

    molecules = system.molecules                # list of all instances of Molecule

    for center in system.centers:
        total_r = 0
        # anal_data = []
        if type(center) == int:
            # looks for the all molecules in a circle of radius centered at the position of the excited molecule

            print('*'*80 + '\n CENTER {}\n'.format(center) + '*'*80)
            process_list, rate_list = get_processes_and_rates(center, system)
            for p, r in zip(process_list, rate_list):
                # print('{}'.format(p))
                print('Process: {}'.format(p['process']))
                print('Donor: {} / Acceptor: {}'.format(p['donor'], p['acceptor']))

                position_d = molecules[p['donor']].get_coordinates()
                position_a = molecules[p['acceptor']].get_coordinates()

                if type(p['process']) == (GoldenRule or DirectRate):
                    distance = np.linalg.norm(distance_vector_periodic(position_a - position_d,
                                                                       system.supercell,
                                                                       p['cell_increment']))
                    print('Distance: ', distance, 'angs')

                if type(p['process']) == GoldenRule:
                    print('Cell_increment: {} '.format(p['cell_increment']))

                    spectral_overlap = general_fcwd(molecules[p['donor']],
                                                    molecules[p['acceptor']],
                                                    p['process'],
                                                    system.conditions)

                    e_coupling = p['process'].get_electronic_coupling(molecules[p['donor']],
                                                                      molecules[p['acceptor']],
                                                                      system.conditions,
                                                                      system.supercell,
                                                                      p['cell_increment'])

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
