import json
import numpy as np
import matplotlib.pyplot as plt
from kimonet import all_none
from scipy import stats
from mpl_toolkits.mplot3d import Axes3D  # Actually needed


class Trajectory:
    def __init__(self, system):
        """
        trajectory: dictionary with the system trajectory (time, number of excitons, positions and process occurred)
        """

        inipos = [[list(system.molecules[center].get_coordinates()),
                   system.molecules[center].state,
                   list(system.molecules[center].cell_state)] for center in system.centers]

        self.trajectory = {'time': [0],
                           'n': [system.get_number_of_excitations()],
                           'positions': [inipos],
                           'process': [],
                           'exciton_altered': [],
                           'supercell': system.supercell}

    def add(self, change_step, time_step, system):
        """
        :param change_step: process occured during time_step: {donor, process, acceptor}
        :param time_step: duration of the chosen process
        :param system: dictionary with the information of the system
        No return function, only updates the dictionary trajectory
        """

        # time update:
        if len(self.trajectory['time']) == 0:
            self.trajectory['time'].append(time_step)
        else:
            self.trajectory['time'].append(self.trajectory['time'][-1] + time_step)

        # excitons positions and quantity update
        exciton_positions = []      # list with all the positions of the excited molecules (simultaneous)
        n = 0                       # number of excitons

        if all_none(system.centers) is True:
            # if there is not any excited molecule saves the position of the last excited molecule.

            # This will be the last position of the trajectory:
            # The time will be the decay time
            # The exciton position could not be saved since it gives no different information from the previous point
            # (during the decay, the exciton does not change its position). n = 0

            last_excited = change_step['acceptor']
            exciton_coordinates = list(system.molecules[last_excited].get_coordinates())   # cartesian coordinates
            excited_state = system.molecules[last_excited].electronic_state()                    # electronic state
            cell_state = system.molecules[last_excited].cell_state

            # a tetra vector is built ([x,y,z], state)
            exciton_point = [exciton_coordinates, excited_state, list(cell_state)]

            # collector of the positions of all excitons transferring.
            exciton_positions.append(exciton_point)

        else:
            for centre in system.centers:
                if type(centre) == int:
                    exciton_coordinates = list(system.molecules[centre].get_coordinates())      # cartesian coordinates
                    excited_state = system.molecules[centre].electronic_state()                       # electronic state
                    cell_state = system.molecules[centre].cell_state

                    # a tetra vector is built ([x,y,z], state)
                    exciton_point = [exciton_coordinates, excited_state, list(cell_state)]

                    # collector of the positions of all excitons transferring.
                    exciton_positions.append(exciton_point)
                    n = n + 1

        self.trajectory['positions'].append(exciton_positions)
        self.trajectory['n'].append(n)

        # process occurred.

        self.trajectory['process'].append(change_step['process'])

        # exciton that suffered the process

        self.trajectory['exciton_altered'].append(change_step['index'])

        # No return function: only updates trajectory.

    def get_data(self):
        return self.trajectory

    def plot_2d(self):

        # types = [pos[0][1] for pos in self.trajectory['positions']]

        vector = [pos[0][0] for pos in self.trajectory['positions']]
        vector = np.array(vector).T
        plt.xlim([0, 20])
        plt.ylim([0, 20])

        plt.plot(vector[0], vector[1], '-o')

        return plt

    def plot_distances(self):

        initial = np.array(self.trajectory['positions'][0][0][0])

        vector = []
        lattice = np.zeros_like(initial)
        for pos in self.trajectory['positions']:
            lattice += np.dot(self.trajectory['supercell'], pos[0][2])
            vector.append(np.array(pos[0][0]) - lattice - initial)

        #vector = [np.array(pos[0][0]) - initial for pos in self.trajectory['positions']]

        vector = np.array(vector).T
        vector = np.linalg.norm(vector, axis=0)

        plt.plot(self.trajectory['time'], vector)

        return plt

    def get_diffusion(self):

        initial = np.array(self.trajectory['positions'][0][0][0])

        vector = []
        lattice = np.zeros_like(initial)
        for pos in self.trajectory['positions']:
            lattice += np.dot(self.trajectory['supercell'], pos[0][2])
            vector.append(np.array(pos[0][0]) - lattice - initial)

        #vector = [np.array(pos[0][0]) - initial for pos in self.trajectory['positions']]

        vector = np.array(vector).T
        n_dim = len(vector)

        vector2 = np.linalg.norm(vector, axis=0)**2/n_dim

        t = np.array(self.trajectory['time'])
        slope, intercept, r_value, p_value, std_err = stats.linregress(t, vector2)

        return slope

    def get_diffusion_tensor(self):

        initial = np.array(self.trajectory['positions'][0][0][0])

        vector = []
        lattice = np.zeros_like(initial)
        for pos in self.trajectory['positions']:
            lattice += np.dot(self.trajectory['supercell'], pos[0][2])
            vector.append(np.array(pos[0][0]) - lattice - initial)

        t = np.array(self.trajectory['time'])

        vector = np.array(vector).T

        tensor_x = []
        for v1 in vector:
            tensor_y = []
            for v2 in vector:
                vector2 = v1*v2
                slope, intercept, r_value, p_value, std_err = stats.linregress(t, vector2)
                tensor_y.append(slope)
            tensor_x.append(tensor_y)

        return np.array(tensor_x)

    def get_lifetime(self):

        if len(self.trajectory['time']) > 0:
            return self.trajectory['time'][-1]
        else:
            return 0


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


def visualize_system(system):

    ndim = system.molecules[0].get_dim()
    #fig, ax = plt.subplots()
    fig = plt.figure()

    fig.suptitle('Transition moment')
    if ndim == 3:
        ax = fig.gca(projection='3d')
    else:
        ax = fig.gca()

    # plt.xlim([0, np.dot([1, 0], system.supercell[0])])
    # plt.ylim([0, np.dot([0, 1], system.supercell[1])])

    # define color by state
    colors = {'gs': 'red',
              's0': 'blue',
              's1': 'green',
              't1': 'orange'}

    for molecule in system.molecules:
        c = molecule.coordinates
        o = molecule.get_transition_moment()
        if ndim == 1:
            ax.quiver(c[0], 0, o[0], 0, color=colors[molecule.state])
        if ndim == 2:
            ax.quiver(c[0], c[1], o[0], o[1], color=colors[molecule.state])
        if ndim == 3:
            ax.quiver(c[0], c[1], c[2], o[0], o[1], o[2], normalize=True, color=colors[molecule.state])
            # ax.quiver(c[0], c[1], c[2], o[0], o[1], o[2], length=0.1, normalize=True)

    # ax.quiverkey(q, X=0.3, Y=1.1, U=10,
    #               label='Quiver key, length = 10', labelpos='E')

    plt.show()
