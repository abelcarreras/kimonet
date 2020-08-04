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

    # molecules = system.molecules                # list of all instances of Molecule
    # centre_indexes = system.centers              # tricky list with the indexes of all excited molecules

    process_collector = []                          # list with the respective processes (for all centers)
    for center in system.centers:
        if isinstance(center, int):
            process_list = get_processes_and_rates(center, system)
            process_collector += process_list

    # If no process available system cannot evolve and simulation is finished
    if len(process_collector) == 0:
        system.is_finished = True
        return None, 0
    chosen_process, time = kmc_algorithm(process_collector, system)
    # chooses one of the processes and gives it a duration using the Kinetic Monte-Carlo algorithm
    update_step(chosen_process, system)        # updates both lists according to the chosen process

    # finally the chosen process and the advanced time are returned
    return chosen_process, time


def update_step(change_step, system):
    """
    :param change_step: dictionary like dict(center, process, neighbour)
    Modifies the state of the donor and the acceptor. Removes the donor from the centre_indexes list
    and includes the acceptor. If its a decay only changes and removes the donor

    New if(s) entrances shall be defined for more processes.
    """

    process = change_step['process']
    if isinstance(process, (GoldenRule, DirectRate)):

        donor_state = process.final[0]
        acceptor_state = process.final[1]

        system.add_excitation_index(donor_state, change_step['donor'])  # des excitation of the donor
        system.add_excitation_index(acceptor_state, change_step['acceptor'])  # excitation of the acceptor

        # cell state assumes symmetric states cross: acceptor -> donor & donor -> acceptor
        acceptor_cell_state = process.acceptor.cell_state
        donor_cell_state = process.donor.cell_state
        process.acceptor.cell_state = donor_cell_state - process.cell_increment
        process.donor.cell_state = acceptor_cell_state + process.cell_increment

        # system.molecules[chosen_process['donor']].cell_state *= 0

        if process.final[0] == _ground_state_:
            process.donor.cell_state *= 0

        if process.final[1] == _ground_state_:
            process.acceptor.cell_state *= 0

    elif isinstance(process, DecayRate):
        final_state = process.final[0]
        # print('final_state', final_state)
        system.add_excitation_index(final_state, change_step['donor'])

        if final_state == _ground_state_:
            process.donor.cell_state *= 0
    else:
        raise Exception('Process type not recognized')


def system_test_info(system):
    from kimonet.core.processes.fcwd import general_fcwd
    from kimonet.utils.units import HBAR_PLANCK
    from kimonet.utils import distance_vector_periodic
    import numpy as np

    # molecules = system.molecules                # list of all instances of Molecule

    for center in system.centers:
        total_r = 0
        # anal_data = []
        if isinstance(center, int):
            # looks for the all molecules in a circle of radius centered at the position of the excited molecule

            print('*'*80 + '\n CENTER {}\n'.format(center) + '*'*80)
            process_list = get_processes_and_rates(center, system)
            print('plist', process_list)
            for p in process_list:
                proc = p['process']
                # print('{}'.format(p))
                print('Process: {}'.format(proc))
                print('Donor: {} / Acceptor: {}'.format(p['donor'], p['acceptor']))

                position_d = proc.donor.get_coordinates()
                r = proc.get_rate_constant(system.conditions, system.supercell)

                if isinstance(proc, (GoldenRule, DirectRate)):
                    position_a = proc.acceptor.get_coordinates()

                    distance = np.linalg.norm(distance_vector_periodic(position_a - position_d,
                                                                       system.supercell,
                                                                       proc.cell_increment))
                    print('Distance: ', distance, 'angs')

                if isinstance(proc, GoldenRule):
                    print('Cell_increment: {} '.format(proc.cell_increment))

                    spectral_overlap = proc.get_fcwd()
                    #spectral_overlap = general_fcwd(proc.donor,
                    #                                proc.acceptor,
                    #                                proc,
                    #                                system.conditions)

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

