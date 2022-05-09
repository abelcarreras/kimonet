__version__ = '0.1.4'
from kimonet.core import do_simulation_step, system_test_info
from kimonet.analysis import Trajectory, visualize_system
from warnings import warn
import numpy as np
import time
import sys


def calculate_kmc(system, num_trajectories=100, max_steps=10000, silent=False):

    trajectories = []
    for j in range(num_trajectories):
        system_copy = system.copy()

        if not silent:
            print('Trajectory: {}'.format(j))

        trajectory = Trajectory(system_copy)

        for i in range(max_steps):

            change_step, step_time = do_simulation_step(system_copy)
            if system_copy.is_finished:
                break

            trajectory.add_step(change_step, step_time)

            #print('===============================')
            #system_test_info(system_copy)
            #visualize_system(system_copy)

            if i == max_steps-1:
                warn('Maximum number of steps reached!!')

        # trajectory.plot_2d('s1').show()
        # trajectory.plot_2d('t1').show()
        # trajectory.plot_distances('s1').show()
        # trajectory.plot_distances('s2').show()
        # trajectory.plot_graph().show()
        # for node in trajectory.graph.nodes:
        #     print(trajectory.graph.nodes[node]['state'])
        trajectories.append(trajectory)

    return trajectories


def _run_trajectory(index, system, max_steps, silent):
    np.random.seed(int(index * time.time() % 1 * 1e8))

    system_copy = system.copy()
    trajectory = Trajectory(system_copy)
    for i in range(max_steps):

        chosen_process, step_time = do_simulation_step(system_copy)
        if system_copy.is_finished:
            break

        trajectory.add_step(chosen_process, step_time)

        if i == max_steps-1:
            warn('Maximum number of steps reached!!')

    if not silent:
        print('Trajectory {} done!'.format(index))
    return trajectory



def _run_trajectory_intermediate(args):
    system, max_steps, silent = args[1]
    index = args[0]
    return _run_trajectory(index, system, max_steps, silent)


def calculate_kmc_parallel_py2(system, num_trajectories=100, max_steps=10000, silent=False, processors=2):
    import itertools
    from multiprocessing import Pool #, freeze_support

    pool = Pool(processes=processors)
    a_args = list(range(num_trajectories))
    second_arg = [system, max_steps, silent]
    trajectories = pool.map(_run_trajectory_intermediate, itertools.izip(a_args, itertools.repeat(second_arg)))
    return trajectories


def calculate_kmc_parallel_py3(system, num_trajectories=100, max_steps=10000, silent=False, processors=2, use_fork=True):
    # This function only works in Python3
    import concurrent.futures as futures
    from multiprocessing import set_start_method

    if use_fork:
        set_start_method("fork")

    # executor = futures.ThreadPoolExecutor(max_workers=processors)
    executor = futures.ProcessPoolExecutor(max_workers=processors)

    futures_list = []
    for i in range(num_trajectories):
        futures_list.append(executor.submit(_run_trajectory, i, system, max_steps, silent))

    trajectories = []
    for f in futures.as_completed(futures_list):
        trajectories.append(f.result())

    return trajectories


def calculate_kmc_parallel(system, num_trajectories=100, max_steps=10000, silent=False, processors=2):

    if sys.version_info[0] < 3:
        # Run python 2
        return calculate_kmc_parallel_py2(system, num_trajectories, max_steps, silent, processors)
    else:
        # Run python 3
        return calculate_kmc_parallel_py3(system, num_trajectories, max_steps, silent, processors)


def calculate_kmc_parallel_alternative(system, num_trajectories=100, max_steps=10000, silent=False, processors=2):

    from multiprocessing import cpu_count, Pool
    from functools import partial
    pool = Pool(processes=processors)
    trajectories = pool.map(partial(_run_trajectory, system=system, max_steps=max_steps, silent=silent), range(num_trajectories))
    return trajectories
