import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from mpl_toolkits.mplot3d import Axes3D  # Actually needed


class Node:
    def __init__(self, state, coordinates, supercell):
        self.state = state
        self.coordinates = list(coordinates)
        self.supercell = list(supercell)


class Trajectory:
    def __init__(self, system):
        """
        Stores and analyzes the information of a kinetic MC trajectory

        system: initial system
        """

        #inipos = [[list(system.molecules[center].get_coordinates()),
        #           system.molecules[center].state,
        #           list(system.molecules[center].cell_state)] for center in system.centers]

        self.centers = [{'coordinates': [list(system.molecules[center].get_coordinates())],
                         'state': [system.molecules[center].state],
                         'cell_state': [list(system.molecules[center].cell_state)]} for center in system.centers]


        self.times = [0]
        self.process = []
        self.exciton_altered = []
        self.supercell = system.supercell
        self.system = system

        self.center_track = {}
        # for i, center in enumerate(self.system.centers):
        #    self.center_track['{}'.format(i)] = center

        for i, center in enumerate(self.system.centers):
            self.center_track['{}'.format(center)] = i

        self.n_dim = len(self.centers[0]['coordinates'][0])
        self.n_centers = len(self.centers)

    def add(self, change_step, time_step):
        """
        Adds trajectory step

        :param change_step: process occurred during time_step: {donor, process, acceptor}
        :param time_step: duration of the chosen process
        """

        self.times.append(self.times[-1] + time_step)
        self.exciton_altered.append(change_step['index'])
        self.process.append(change_step['process'])

        # key = next(key for key, value in self.center_track.items() if value == change_step['donor'])
        # self.center_track[key] = change_step['acceptor']

        self.center_track['{}'.format(change_step['acceptor'])] = self.center_track.pop('{}'.format(change_step['donor']))

        # print('dict', self.center_track, self.center_track['{}'.format(self.system.centers[0])])

        for center in self.system.centers:
            i = self.center_track['{}'.format(center)]
            exciton_coordinates = list(self.system.molecules[center].get_coordinates())  # cartesian coordinates
            excited_state = self.system.molecules[center].electronic_state()  # electronic state
            cell_state = list(self.system.molecules[center].cell_state)

            self.centers[i]['coordinates'].append(exciton_coordinates)
            self.centers[i]['state'].append(excited_state)
            self.centers[i]['cell_state'].append(cell_state)

        return

    def get_number_of_centers(self):
        return self.n_centers

    def get_dimension(self):
        return self.n_dim

    def plot_2d(self, icenter, supercell_only=False):

        if self.get_dimension() != 2:
            raise Exception('plot_2d can only be used in 2D systems')

        if supercell_only:
            #  plt.xlim([0, 20])
            #  plt.ylim([0, 20])
            vector = np.array(self.centers[icenter]['coordinates'])
        else:
            initial = np.array(self.centers[icenter]['coordinates'][0])

            cell_states = self.centers[icenter]['cell_state']
            coordinates = self.centers[icenter]['coordinates']

            vector = []
            lattice = np.zeros_like(initial)
            for cell_state, coordinate in zip(cell_states, coordinates):
                lattice += np.dot(self.supercell, cell_state)
                vector.append(np.array(coordinate) - lattice - initial)

        vector = np.array(vector).T

        plt.plot(vector[0], vector[1], '-o')

        return plt

    def plot_distances(self, icenter):

        initial = np.array(self.centers[icenter]['coordinates'][0])

        cell_states = self.centers[icenter]['cell_state']
        coordinates = self.centers[icenter]['coordinates']

        vector = []
        lattice = np.zeros_like(initial)
        for cell_state, coordinate in zip(cell_states, coordinates):
            lattice += np.dot(self.supercell, cell_state)
            vector.append(np.array(coordinate) - lattice - initial)

        vector.append(vector[-1])

        vector = np.array(vector).T
        n_dim, n_length = vector.shape
        vector = np.linalg.norm(vector, axis=0)
        t = np.array(self.times)[:n_length]

        plt.plot(t, vector)

        return plt

    def get_diffusion(self, icenter):

        initial = np.array(self.centers[icenter]['coordinates'][0])

        cell_states = self.centers[icenter]['cell_state']
        coordinates = self.centers[icenter]['coordinates']

        vector = []
        lattice = np.zeros_like(initial)
        for cell_state, coordinate in zip(cell_states, coordinates):
            lattice += np.dot(self.supercell, cell_state)
            vector.append(np.array(coordinate) - lattice - initial)

        vector.append(vector[-1])
        vector = np.array(vector).T
        n_dim, n_length = vector.shape

        vector2 = np.linalg.norm(vector, axis=0)**2  # emulate dot product in axis 0

        t = np.array(self.times)[:n_length]

        slope, intercept, r_value, p_value, std_err = stats.linregress(t, vector2)

        return slope/(2*n_dim)

    def get_diffusion_tensor(self, icenter):

        initial = np.array(self.centers[icenter]['coordinates'][0])

        cell_states = self.centers[icenter]['cell_state']
        coordinates = self.centers[icenter]['coordinates']

        vector = []
        lattice = np.zeros_like(initial)
        for cell_state, coordinate in zip(cell_states, coordinates):
            lattice += np.dot(self.supercell, cell_state)
            vector.append(np.array(coordinate) - lattice - initial)

        vector.append(vector[-1])
        vector = np.array(vector).T
        n_dim, n_length = vector.shape
        t = np.array(self.times)[:n_length]

        tensor_x = []
        for v1 in vector:
            tensor_y = []
            for v2 in vector:
                vector2 = v1*v2
                slope, intercept, r_value, p_value, std_err = stats.linregress(t, vector2)
                tensor_y.append(slope)
            tensor_x.append(tensor_y)

        return np.array(tensor_x)/2

    def get_lifetime(self, icenter):
        n_length = len(self.centers[icenter]['coordinates'])
        return self.times[n_length]

    def get_diffusion_length_square(self, icenter):

        initial = np.array(self.centers[icenter]['coordinates'][0])

        cell_states = self.centers[icenter]['cell_state']
        coordinates = self.centers[icenter]['coordinates']

        lattice = np.zeros_like(initial)
        for cell_state, coordinate in zip(cell_states, coordinates):
            lattice += np.dot(self.supercell, cell_state)

        vector = np.array(self.centers[icenter]['coordinates'][-1]) - lattice - initial

        return np.dot(vector, vector)

    def get_diffusion_length_square_tensor(self, icenter):

        initial = np.array(self.centers[icenter]['coordinates'][0])

        cell_states = self.centers[icenter]['cell_state']
        coordinates = self.centers[icenter]['coordinates']

        lattice = np.zeros_like(initial)
        for cell_state, coordinate in zip(cell_states, coordinates):
            lattice += np.dot(self.supercell, cell_state)

        vector = np.array(self.centers[icenter]['coordinates'][-1]) - lattice - initial

        tensor_x = []
        for v1 in vector:
            tensor_y = []
            for v2 in vector:
                tensor_y.append(v1*v2)
            tensor_x.append(tensor_y)

        return np.array(tensor_x)


class TrajectoryAnalysis:

    def __init__(self, trajectories):
        self.trajectories = trajectories
        self.n_centers = trajectories[0].get_number_of_centers()
        self.n_dim = trajectories[0].get_dimension()
        self.n_traj = len(trajectories)

    def __str__(self):

        txt_data = '\nTrajectory Analysis\n'
        txt_data += '------------------------------\n'
        txt_data += 'Number of trajectories: {}\n'.format(self.n_traj)
        txt_data += 'Dimension: {}\n'.format(self.n_dim)
        txt_data += 'Number of centers: {}\n'.format(self.n_centers)

        return txt_data

    def diffusion_coeff_tensor(self):
        """
        calculate the average diffusion tensor defined as:

        DiffTensor = 1/2 * <DiffLen^2> / <time>

        :param trajectories: list of Trajectory
        :return:
        """
        return np.average([traj.get_diffusion_tensor(0) for traj in self.trajectories], axis=0)

    def diffusion_length_tensor(self):
        """
        calculate the average diffusion length tensor defined as:

        DiffLenTen = SQRT( 2 * DiffTensor * lifetime)

        :param trajectories: list of Trajectory
        :return:
        """
        dl_tensor = np.average([traj.get_diffusion_length_square_tensor(0) for traj in self.trajectories], axis=0)

        return np.sqrt(dl_tensor)

    def diffusion_coefficient(self):
        """
        Return the average diffusion coefficient defined as:

        DiffCoeff = 1/(2*z) * <DiffLen^2>/<time>

        :return:
        """
        return np.average([traj.get_diffusion(0) for traj in self.trajectories])

    def lifetime(self):
        return np.average([traj.get_lifetime(0) for traj in self.trajectories])

    def diffusion_length(self):
        """
        Return the average diffusion coefficient defined as:

        DiffLen = SQRT(2 * z * DiifCoeff * LifeTime)

        :return:
        """
        length2 = np.average([traj.get_diffusion_length_square(0) for traj in self.trajectories])
        return np.sqrt(length2)

    def plot_2d(self):
        plt = None
        for traj in self.trajectories:
            plt = traj.plot_2d(0)
        return plt

    def plot_distances(self):
        plt = None
        for traj in self.trajectories:
            plt = traj.plot_distances(0)
        return plt


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
            ax.quiver(c[0], c[1], c[2], o[0], o[1], o[2], normalize=False, color=colors[molecule.state])
            # ax.quiver(c[0], c[1], c[2], o[0], o[1], o[2], length=0.1, normalize=True)

    # ax.quiverkey(q, X=0.3, Y=1.1, U=10,
    #               label='Quiver key, length = 10', labelpos='E')

    plt.show()
