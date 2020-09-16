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


def update_step(change_step, system):
    """
    :param change_step: dictionary like dict(center, process, neighbour)
    Modifies the state of the donor and the acceptor. Removes the donor from the centre_indexes list
    and includes the acceptor. If its a decay only changes and removes the donor

    New if(s) entrances shall be defined for more processes.
    """

    process = change_step #['process']
    if isinstance(process, (GoldenRule, DirectRate)):

        #donor_state = process.final[0]
        #acceptor_state = process.final[1]

        #donor_index = system.get_molecule_index(process.donor)
        #acceptor_index = system.get_molecule_index(process.acceptor)

        #print('initial: ', process.initial)
        #print('initial 0', process.initial[0], system.molecules[donor_index].state)
        #print('initial 1', process.initial[1], system.molecules[acceptor_index].state)

        # print('states', donor_state, acceptor_state)
        #print('states: ', system.get_states())
        #print('->', process.initial[0], process.initial[1])

        # print(donor_state.label, donor_index, acceptor_state.label, acceptor_index, process.get_rate_constant({'custom_constant': 1}, [1]))

        system.remove_exciton(process.initial[0])
        system.remove_exciton(process.initial[1])

        system.add_exciton(process.final[0])
        system.add_exciton(process.final[1])

        # system.add_excitation_index(donor_state, donor_index)  # de-excitation of the donor
        # system.add_excitation_index(acceptor_state, acceptor_index)  # excitation of the acceptor


        # print(process.donor.get_center())

        print('**** SYSTEM ****')
        for i, mol in enumerate(system.molecules):
            print(i, mol.state.label, mol.state)
        print('****************')

        # cell state assumes symmetric states cross: acceptor -> donor & donor -> acceptor
        acceptor_cell_state = process.acceptor.cell_state
        donor_cell_state = process.donor.cell_state
        process.acceptor.cell_state = donor_cell_state - process.cell_increment
        process.donor.cell_state = acceptor_cell_state + process.cell_increment

        # system.molecules[chosen_process['donor']].cell_state *= 0

        if process.final[0] == _GS_.label:
            process.donor.cell_state *= 0

        if process.final[1] == _GS_.label:
            process.acceptor.cell_state *= 0

        print('cell:', process.acceptor.cell_state, process.acceptor.cell_state)


    elif isinstance(process, DecayRate):
        final_state = process.final[0]
        # print('final_state', final_state)
        donor_index = system.get_molecule_index(process.donor)
        system.add_excitation_index(final_state, donor_index)

        if final_state == _GS_.label:
            process.donor.cell_state *= 0
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

            i_donor = system.get_molecule_index(proc.donor)
            try:
                i_acceptor = system.get_molecule_index(proc.acceptor)
            except Exception:
                i_acceptor = i_donor

            print('Donor: {} / Acceptor: {}'.format(i_donor, i_acceptor))

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

