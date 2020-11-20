__version__ = '0.1'
from kimonet.core import do_simulation_step, system_test_info
from kimonet.analysis import Trajectory
from warnings import warn
import numpy as np
import time


def calculate_kmc(system, num_trajectories=100, max_steps=10000, silent=False):

    trajectories = []
    for j in range(num_trajectories):
        system_copy = system.copy()

        if not silent:
            print('Trajectory: ', j)

        trajectory = Trajectory(system_copy)

        for i in range(max_steps):

            change_step, step_time = do_simulation_step(system_copy)
            if system_copy.is_finished:
                break

            trajectory.add_step(change_step, step_time)

            if i == max_steps-1:
                warn('Maximum number of steps reached!!')

        # trajectory.plot_2d('s1').show()
        # trajectory.plot_distances('s1').show()
        # trajectory.plot_distances('s2').show()
        # trajectory.plot_graph().show()
        #for node in trajectory.graph.nodes:
        #    print(trajectory.graph.nodes[node]['state'])
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

        print('wer')
        trajectory.add_step(chosen_process, step_time)

        if i == max_steps-1:
            warn('Maximum number of steps reached!!')

    if not silent:
        print('Trajectory {} done!'.format(index))
    return trajectory


def calculate_kmc_parallel(system, num_trajectories=100, max_steps=10000, silent=False, processors=2):
    # This function only works in Python3
    import concurrent.futures as futures

    # executor = futures.ThreadPoolExecutor(max_workers=processors)
    executor = futures.ProcessPoolExecutor(max_workers=processors)

    futures_list = []
    for i in range(num_trajectories):
        futures_list.append(executor.submit(_run_trajectory, i, system, max_steps, silent))

    trajectories = []
    for f in futures.as_completed(futures_list):
        trajectories.append(f.result())

    return trajectories


def calculate_kmc_parallel_alternative(system, num_trajectories=100, max_steps=10000, silent=False, processors=2):

    from multiprocessing import cpu_count, Pool
    from functools import partial
    pool = Pool(processes=processors)
    trajectories = pool.map(partial(_run_trajectory, system=system, max_steps=max_steps, silent=silent), range(num_trajectories))
    return trajectories
