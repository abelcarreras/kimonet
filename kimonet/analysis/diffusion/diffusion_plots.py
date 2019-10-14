import matplotlib.pyplot as plt
import numpy as np
import json


def final_distances_histogram(squared_distances, bins=10):
    """
    :param squared_distances:
    :param bins:
    :return: Plots an histogram with the final positions of the excitons
    """
    plt.hist(np.sqrt(squared_distances), bins=bins)
    plt.title('Final distances histogram')
    plt.xlabel('Final position (nm)')
    plt.ylabel('# of occurrences')
    plt.show()


def diffusion_length_and_lifetime_convergence(root_mean_square, lifetimes, theoretical_values):
    """
    :param root_mean_square:
    :param lifetimes:
    :param theoretical_values:
    :return: Plots the convergence of the simulation values to the theoretical values with the number of
    trajectories averaged.
    """

    iterations = np.arange(len(root_mean_square))

    if theoretical_values['diffusion_length'] is None:              # case where there isn't a theoretical model

        plt.plot(iterations, root_mean_square, 'ro', label='simulation values')
        plt.xlabel('# of trajectories averaged')
        plt.ylabel('Diffusion length (nm)')
        plt.title('Convergence of diffusion length')
        plt.legend()
        plt.show()

    else:
        # diffusion length convergence:
        theo_diff_length = np.ones((len(iterations), 1)) * theoretical_values['diffusion_length']

        plt.plot(iterations, theo_diff_length, label='theoretical value')
        plt.plot(iterations, root_mean_square, 'ro', label='simulation values')
        plt.xlabel('# of trajectories averaged')
        plt.ylabel('$L_D$ (nm)')
        plt.title('Convergence of diffusion length')
        plt.legend()
        plt.show()

    # exciton lifetime convergence:
    # there is always a theoretical value since the decay rate is finite
    theo_lifetime = np.ones((len(iterations), 1)) * theoretical_values['lifetime']

    plt.plot(iterations, theo_lifetime, label='theoretical value')
    plt.plot(iterations, lifetimes, 'ro', label='simulation values')
    plt.xlabel('# of trajectories averaged')
    plt.ylabel('Lifetime (ns)')
    plt.title('Convergence of exciton lifetime')
    plt.legend()
    plt.show()


def diffusion_line(mean_squared_distances, mean_lifetimes, linear_regression):
    """
    :param mean_squared_distances:
    :param mean_lifetimes:
    :param linear_regression:
    :return: Plots the points <l²> vs <t> at every step and also the linear regression.
    """

    regression_l_values = linear_regression[0] * np.array(mean_lifetimes) + linear_regression[1]

    plt.plot(mean_lifetimes, regression_l_values, label='Regression. $R^{2} = %.3f$' % linear_regression[2])
    plt.plot(mean_lifetimes, mean_squared_distances, 'ro', label='Simulation values')
    plt.xlabel('$<t>, ns$')
    plt.ylabel('$<l^{2}>, nm$')
    plt.title('Statistical study of $D$')
    plt.legend()
    plt.show()

