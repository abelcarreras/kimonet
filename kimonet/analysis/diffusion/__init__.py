import numpy as np
from scipy.stats import linregress
from kimonet.analysis.diffusion.diffusion_plots import final_distances_histogram, diffusion_length_and_lifetime_convergence,\
    diffusion_line


def statistical_diffusion_study(trajectories, theoretical_values, system_information):
    """
    :param trajectories: list with all the trajectories, each is a dictionary with a list with the positions,
    time to reach each position and the exciton transfer process.
    :param theoretical_values: theoretical values computed
    :param system_information:
    :return: the diffusivity constant computed using all the points of all the trajectories.
        For each step we computed the mean square of the distance and the mean time.
        Doing the lineal regression of all these couple of values, the diffusivity constant is defined as the
        slope of the line (except for a dimensional parameter depending of the dimensionality.
    """

    # STATISTICAL CALCULATION OF D
    mean_square_distances, mean_lifetimes = get_r2_t_averaged_lists(trajectories)

    linear_regression = linregress(mean_lifetimes, mean_square_distances)

    # PLOT OF <l²> vs <t> at each step
    diffusion_line(mean_square_distances, mean_lifetimes, linear_regression)

    # Diffusion length value (using theoretical lifetime):
    dimensionality = len(system_information['lattice']['dimensions'])
    theo_lifetime = theoretical_values['lifetime']

    # the diffusion constant is the slope of the line gained by the linear regression of the data
    diffusion_constant = linear_regression[0] / 2 * dimensionality

    return {'diffusion_constant': diffusion_constant,
            'diffusion_length': np.sqrt(2*dimensionality*diffusion_constant*theo_lifetime)}


###############################################################################################################


def get_r2_t_averaged_lists(trajectories):
    """
    :param trajectories: list with all the trajectories, each is a dictionary with a list with the positions,
    time to reach each position and the exciton transfer process.
    :return: list with the mean square distances of the trajectories at every step
             list with the average time over all trajectories at every step
    """

    mean_square_distance_list = []                  # mean square distances at each step
    mean_lifetime_list = []                         # averaged time over the trajectories at each step

    # a fixed number of steps is defined. Taken as the averaged length of trajectories:
    lengths = []
    for trajectory in trajectories:
        lengths.append(len(trajectory['positions']))
    steps = int(np.average(np.array(lengths)))

    for i in range(0, steps):
        squared_distances = []              # list of the squared distances of all trajectories at a fixed step
        time = []                           # analogous for the time

        for trajectory in trajectories:

            if i >= len(trajectory['positions'])-1:
                count = 0           # no values added if the trajectory is shorter than the fixed step (length)

            else:
                distance_i_squared = np.linalg.norm(np.array(trajectory['positions'][i][0][0]))**2
                # we take the exciton position at step i.
                # first 0 index --> position of the firts exciton (in case there were more excitons transferring)
                # second 0 index --> takes the coordinates of the excited molecule (ignores the electronic state)

                time_i = trajectory['time'][i]

                squared_distances.append(distance_i_squared)
                time.append(time_i)

        mean_square_distance_list.append(np.average(np.array(squared_distances)))
        mean_lifetime_list.append(np.average(np.array(time)))

    return mean_square_distance_list, mean_lifetime_list


###############################################################################################################


def diffusion_parameters(trajectories, theoretical_values, system_information):
    """
    :param trajectories: list with all the trajectories, each is a dictionary with a list with the positions,
    time to reach each position and the exciton transfer process.
    :param system_information
    :param theoretical_values
    :return: the diffusion length computed as the root mean square of the final positions of the trajectory
             the exciton life time computed as the average of the total time advance
             the diffusion constant computed with the two parameters above
    """

    # dimensionality. Given by the length of the list dimensions
    dimensionality = len(system_information['lattice']['dimensions'])

    ############################################################################################################
    # the experimental diffusion length is computed as the root mean square of
    # the final position distances of the exciton
    squared_distances = []

    # the life time of the exciton is taken as the average of the total
    # time duration of the exciton over all trajectories
    durations = []
    ############################################################################################################

    # the convergence of the root mean square of the final distances and total duration to the theoretical
    # values when adding trajectories will also be studied.
    root_mean_square = []
    lifetimes = []

    for trajectory in trajectories:
        coordinates = trajectory['positions'][-1][0][0]
        # we take the position of the last excited molecule.
        # first 0 index --> position of the first exciton (in case there were more excitons transferring)
        # second 0 index --> takes the coordinates of the excited molecule (ignores the electronic state)

        final_distance_squared = np.linalg.norm(np.array(coordinates)) ** 2
        final_time = trajectory['time'][-1]
        # time -1 takes into account the decay time too

        # squared final distances and total times for each trajectory are collected
        squared_distances.append(final_distance_squared)
        durations.append(final_time)

        # we average these list for each new value added
        root_mean_square.append(np.sqrt(np.average(squared_distances)))
        lifetimes.append(np.average(np.average(durations)))

    ############################################################################################################

    #   CONVERGENCE OF THE EXPERIMENTAL VALUES PLOT
    diffusion_length_and_lifetime_convergence(root_mean_square, lifetimes, theoretical_values)

    #   FINAL DISTANCES HISTOGRAM
    final_distances_histogram(squared_distances)

    #############################################################################################

    # returns the diffusion parameters in a dictionary
    return {'diffusion_length': root_mean_square[-1],
            'lifetime': lifetimes[-1],
            'diffusion_constant': np.average(squared_distances) / (2 * dimensionality * lifetimes[-1])}
