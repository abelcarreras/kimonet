import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from kimonet.analysis.trajectory_graph import TrajectoryGraph as Trajectory
from kimonet.analysis.trajectory_analysis import TrajectoryAnalysis


def visualize_system(system, dipole=None):

    ndim = system.molecules[0].get_dim()
    #fig, ax = plt.subplots()
    fig = plt.figure()

    fig.suptitle('Orientation' if dipole is None else 'TDM {}'.format(dipole))
    if ndim == 3:
        ax = fig.gca(projection='3d')
        ax.set_zlabel('Z')
    else:
        ax = fig.gca()

    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    # plt.xlim([0, np.dot([1, 0], system.supercell[0])])
    # plt.ylim([0, np.dot([0, 1], system.supercell[1])])


    # define color by state
    colors = {'gs': 'red',
              's1': 'blue',
              's2': 'green',
              't1': 'orange'}

    for i, molecule in enumerate(system.molecules):
        c = molecule.coordinates
        if dipole is None:
            o = molecule.get_orientation_vector()
        else:
            o = molecule.get_transition_moment(to_state=dipole)

        if ndim == 1:
            ax.quiver(c[0], 0, o[0], 0, color=colors[molecule.state])
        if ndim == 2:
            ax.quiver(c[0], c[1], o[0], o[1], color=colors[molecule.state])
            ax.text(c[0], c[1], '{}'.format(i), fontsize=12)
        if ndim == 3:
            ax.quiver(c[0], c[1], c[2], o[0], o[1], o[2], normalize=False, length=5, color=colors[molecule.state])
            # ax.quiver(c[0], c[1], c[2], o[0], o[1], o[2], length=0.1, normalize=True)
            ax.text(c[0], c[1], c[2], '{}'.format(i), fontsize=12)

    # Plot lattice vectors
    for lattice_vector in system.supercell:
        ax.plot(*np.array([[0]*ndim, lattice_vector]).T)

    # ax.quiverkey(q, X=0.3, Y=1.1, U=10,
    #               label='Quiver key, length = 10', labelpos='E')

    plt.show()


# THIS IS OLD and not used, to be replaced in the near future
def merge_json_files(file1, file2):
    """
    :param file1: name (string) of the first file in .json format
    :param file2: the same for the second file
    :return:
        if the headings of both files (system_information and molecule_information) are equal, then
        the trajectories are merged. A single file with the same heading and with the merged trajectories is returned.
        in other case, None is returned and a a warning message is printed.
    """

    # reading of the files
    with open(file1, 'r') as read_file1:
        data_1 = json.load(read_file1)

    with open(file2, 'r') as read_file2:
        data_2 = json.load(read_file2)

    heading_1 = data_1['system_information']
    heading_2 = data_2['system_information']

    if heading_1 == heading_2:
        trajectories_1 = data_1['trajectories']
        trajectories_2 = data_2['trajectories']

        merged_trajectories = trajectories_1 + trajectories_2

        data_1['trajectories'] = merged_trajectories

        # data_1 has in its 'trajectories' entrance the merged information of both files,
        # while keeping the system information (that is equal in both files).
        with open(file1, 'w') as write_file:
            json.dump(data_1, write_file)

        print('Trajectories has been merged in ' + file1)

        return

    else:
        print('Trajectories taken in different conditions.')
        return


def get_l2_t_from_file(diffusion_file):
    """
    :param diffusion_file: json file with diffusion results
    :return: The function takes from the file the experimental values of D and lifetime, It returns a list
    with values of time from 0 to lifetime. With D computes the respective values of l².
    The aim of this function is to get this sets of data to plot several of them and be able to compare the slopes.
    As well, the value of the lifetime and diffusion length will be seen in this plot (the last point of each line).
    """

    with open(diffusion_file, 'r') as readfile:
        data = json.load(readfile)

    diffusion_constant = data['experimental']['diffusion_constant']
    lifetime = data['experimental']['lifetime']

    time_set = np.arange(0, lifetime, 0.1)

    squared_distance_set = (2 * diffusion_constant * time_set)
    # the 2 factor stands for the double of the dimensionality

    changed_parameter = data['changed_parameter']

    return time_set, squared_distance_set, diffusion_constant, changed_parameter


def get_diffusion_vs_mu(diffusion_file):
    """
    :param diffusion_file: json file with diffusion results
    :return: Reads the file and returns:
        D, lifetime, L_d, mu
    """

    with open(diffusion_file, 'r') as readfile:
        data = json.load(readfile)

    diffusion_constant = data['experimental']['diffusion_constant']
    lifetime = data['experimental']['lifetime']
    diffusion_length = data['experimental']['diffusion_length']

    mu = np.linalg.norm(np.array(data['conditions']['transition_moment']))

    return diffusion_constant, lifetime, diffusion_length, mu

