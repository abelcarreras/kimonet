import matplotlib.pyplot as plt
import numpy as np


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
    :return: Plots the points <l^2> vs <t> at every step and also the linear regression.
    """

    regression_l_values = linear_regression[0] * np.array(mean_lifetimes) + linear_regression[1]

    plt.plot(mean_lifetimes, regression_l_values, label='Regression. $R^{2} = %.3f$' % linear_regression[2])
    plt.plot(mean_lifetimes, mean_squared_distances, 'ro', label='Simulation values')
    plt.xlabel('$<t>, ns$')
    plt.ylabel('$<l^{2}>, nm$')
    plt.title('Statistical study of $D$')
    plt.legend()
    plt.show()


def plot_polar_plot(tensor_full, plane=(0, 1), title='', max=None, crystal_labels=False):

    tensor = np.array(tensor_full)[np.array(plane)].T[np.array(plane)].T

    if max is None:
        max = np.max(tensor) * 1.2

    r = []
    theta = []
    for i in np.arange(0, np.pi*2, 0.01):
        x = np.cos(i)
        y = np.sin(i)

        rmat = np.array([np.cos(i), np.sin(i)])

        np.linalg.norm(np.dot(rmat, tensor))

        # print(np.linalg.norm([a,b]), np.linalg.norm(tensor[0]*x + tensor[1]*y))
        r.append(np.linalg.norm(np.dot(tensor, rmat)))

        # r.append(np.linalg.norm(tensor[0]*x + tensor[1]*y))
        theta.append(i)

    labels = {'cartesian': ['x', 'y', 'z'],
              'crystal': ['a', 'b', 'c']}

    if crystal_labels:
        labels_plot = [labels['crystal'][i] for i in plane]
    else:
        labels_plot = [labels['cartesian'][i] for i in plane]

    ax = plt.subplot(111, projection='polar')
    ax.arrow(0., 0., np.pi, max,  edgecolor='black', lw=1, zorder=5)
    ax.arrow(0., 0., 3./2*np.pi, max,  edgecolor='black', lw=1, zorder=5)
    ax.annotate("", xy=(0, max), xytext=(0, 0), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(np.pi/2, max), xytext=(0, 0), arrowprops=dict(arrowstyle="->"))
    ax.plot(theta, r)
    ax.set_rmax(max)
    ax.set_rticks(list(np.linspace(0.0, max, 8)))  # Less radial ticks
    ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
    ax.grid(True)
    ax.set_xticklabels(['{}'.format(labels_plot[0]), '', '{}'.format(labels_plot[1]), '', '', '', '', ''])

    ax.set_title(title, va='bottom')
    plt.show()
