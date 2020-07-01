__version__ = '0.1'
_ground_state_ = 'gs'
from kimonet.core import do_simulation_step, system_test_info
from kimonet.analysis import Trajectory
from warnings import warn
import numpy as np


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

        trajectories.append(trajectory)

    return trajectories


def _run_trajectory(system, index, max_steps, silent):
    np.random.seed(index)

    system = system.copy()
    trajectory = Trajectory(system)
    for i in range(max_steps):

        change_step, step_time = do_simulation_step(system)

        if system.is_finished:
            break

        trajectory.add_step(change_step, step_time)

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
        futures_list.append(executor.submit(_run_trajectory, system, i, max_steps, silent))

    trajectories = []
    for f in futures.as_completed(futures_list):
        trajectories.append(f.result())

    return trajectories
