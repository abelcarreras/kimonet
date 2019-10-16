from kimonet.core.kmc import kmc_algorithm
from kimonet.core.processes import get_transfer_rates, update_step, get_decay_rates
import warnings


def update_system(system):
    """
    :param system: Dictionary with all the information of the system
    Dictionary system already has the indexes of the excited molecules
    1. Looks for the neighbourhood of every centre.
    2. Chooses a process for every exciton (KMC). This path variables include: dict(centre, process, new molecule)
    3. Considering all the calculated rates computes the time interval for each process.
    4. Updates the system according to the chosen path and the time passed.
    :return: the chosen process and the advanced time
    """

    molecules = system.molecules                # list of all instances of Molecule
    centre_indexes = system.centers              # tricky list with the indexes of all excited molecules

    rate_collector = []                             # list with all the rates (for all centers)
    process_collector = []                          # list with the respective processes (for all centers)
    # the indexes of both lists coincide.

    for i, centre in enumerate(system.centers):
        if type(centre) == int:
            # looks for the all molecules in a circle of radius centered at the position of the excited molecule

            process_list, rate_list = get_processes_and_rates(centre, system, i)
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


def get_processes_and_rates(centre, system, i):
    """
    :param i: index of the exciton
    :param centre: Index of the studied excited molecule (Donor)
    :param system: Instance of System class
    Computes the transfer and decay rates and builds two dictionaries:
            One with the decay process as key and its rate as argument
            One with the transferred molecule index as key and {'process': rate} as argument

    :return:    process_list: List of elements like dict(center, process, new molecule)
                rate_list: List with the respective rates
                The list indexes coincide.
    """

    transfer_processes, transfer_rates = get_transfer_rates(centre, system, i)
    # calls an external function that computes the transfer rates for all possible transfer processes between
    # the centre and all its neighbours

    decay_processes, decay_rates = get_decay_rates(centre, system, i)
    # calls an external function that computes the decay rates for all possible decay processes of the centre.
    # Uses a method of the class Molecule

    # merges all processes in a list and the same for the rates
    # the indexes of the list must coincide (same length. rate 'i' is related to process 'i')
    process_list = decay_processes + transfer_processes
    rate_list = decay_rates + transfer_rates

    return process_list, rate_list
