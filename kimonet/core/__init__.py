from kimonet.core.kmc import kmc_algorithm
from kimonet.core.processes import get_processes_and_rates
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

    if type(chosen_process['process']).__name__ == 'Transfer':

        donor_state = chosen_process['process'].final[0]
        acceptor_state = chosen_process['process'].final[1]

        system.add_excitation_index(donor_state, chosen_process['donor'])  # des excitation of the donor
        system.add_excitation_index(acceptor_state, chosen_process['acceptor'])  # excitation of the acceptor
        system.molecules[chosen_process['acceptor']].cell_state = system.molecules[chosen_process['donor']].cell_state - chosen_process['cell_increment']

        # system.molecules[chosen_process['donor']].cell_state *= 0

        if chosen_process['process'].final[0] == 'gs':
            system.molecules[chosen_process['donor']].cell_state *= 0

    if type(chosen_process['process']).__name__ == 'Decay':
        final_state = chosen_process['process'].final
        # print('final_state', final_state)
        system.add_excitation_index(final_state, chosen_process['donor'])

        if final_state == 'gs':
            system.molecules[chosen_process['donor']].cell_state *= 0

