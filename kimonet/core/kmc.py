import numpy as np


def kmc_algorithm(process_list, system):
    """
    :param rate_list: List with all the computed rates for all the neighbours for all the centers
    :param process_list: List of elements dict(center, process, new molecule).
    The indexes of each rate in rate_list have the same index that the associated process in
    process_list.
    Chooses a process using the list of rates and associates a time with this process using
    the BKL Kinetic Monte-Carlo algorithm. The algorithm uses 2 random number, one to choose the process and the other
    for the time. The usage of each random number is in an independent function
    :return:    plan: The chosen proces and the new molecule affected
                time: the duration of the process
    """

    rate_list = [proc.get_rate_constant(system.conditions, system.supercell) for proc in process_list]

    process_index = select_process(rate_list)
    chosen_process = process_list[process_index]

    time = time_advance(rate_list)

    return chosen_process, time


def select_process(constant_list):
    """
    :param constant_list: List with the constant rates
    :return: Chooses a position of the list chosen proportionally to its value.
    """
    r = np.sum(constant_list) * np.random.rand()
    # random number picked from the uniform distribution U(0, rates sum)

    sub_list = np.where(r > np.cumsum(constant_list))

    return len(sub_list[0])


def time_advance(rate_list):
    """
    :param rate_list: List with all the rates. Considering all the processes for all exciton
    :return: Process duration. Picks a random time from an exponential distribution
    """
    r = 1 - np.random.rand()  # interval [0, 1) -> (0, 1]
    return (-np.log(r)) / (np.sum(rate_list))
