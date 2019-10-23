import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from mpl_toolkits.mplot3d import Axes3D  # Actually needed!


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
        self.labels = {}

    def get_ranges_from_label(self, label):
        dtype = [('center', int), ('position', int)]
        self.labels[label] = []
        for i, center in enumerate(self.centers):
            for j, state in enumerate(center['state']):
                if state == label:
                    self.labels[label].append((i, j))

        ordered_points = np.sort(np.array(self.labels[label], dtype=dtype), order='center')
        continuous_sections = np.split(ordered_points, np.where(np.diff(ordered_points['position']) != 1)[0] + 1)
        if len(ordered_points) == 0:
            return [(np.array((0, 0), dtype=dtype), np.array((0, 0), dtype=dtype))]
        return [(seq[0], seq[-1]) for seq in continuous_sections]

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

        # print(self.centers[0]['cell_state'][-1])
        # print('t:', len(self.times), len(self.centers[0]['state']))
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
            for cell_state, coordinate in zip(cell_states, coordinates):
                lattice = np.dot(self.supercell, cell_state)
                vector.append(np.array(coordinate) - lattice - initial)

        vector = np.array(vector).T

        plt.plot(vector[0], vector[1], '-o')

        return plt

    def plot_distances(self, icenter):

        initial = np.array(self.centers[icenter]['coordinates'][0])

        cell_states = self.centers[icenter]['cell_state']
        coordinates = self.centers[icenter]['coordinates']

        vector = []
        #lattice = np.zeros_like(initial)
        for cell_state, coordinate in zip(cell_states, coordinates):
            lattice = np.dot(self.supercell, cell_state)
            vector.append(np.array(coordinate) - lattice - initial)

        # vector.append(vector[-1])

        vector = np.array(vector).T
        n_dim, n_length = vector.shape
        vector = np.linalg.norm(vector, axis=0)
        t = np.array(self.times)[:n_length]

        plt.plot(t, vector)

        return plt

    def _vector_list(self, state):

        t = []
        coordinates = []
        sections = self.get_ranges_from_label(state)
        for ini, fin in sections:
            icenter = ini['center']

            initial = np.array(self.centers[icenter]['coordinates'][ini['position']])

            cell_states = self.centers[icenter]['cell_state'][ini['position']: fin['position']+1]
            coordinate_range = self.centers[icenter]['coordinates'][ini['position']: fin['position']+1]

            coord_per = []
            for cell_state, coordinate in zip(cell_states, coordinate_range):
                lattice = np.dot(self.supercell, cell_state) - np.dot(self.supercell, cell_states[0])
                coord_per.append(np.array(coordinate) - lattice - initial)

            coordinates += coord_per

            t += list(np.array(self.times[ini['position']:fin['position']+1]) - self.times[ini['position']])

        return np.array(coordinates).T, t

    def get_diffusion(self, state):

        vector, t = self._vector_list(state)

        if len(vector) == 0:
            return np.nan

        n_dim, n_length = vector.shape

        vector2 = np.linalg.norm(vector, axis=0)**2  # emulate dot product in axis 0

        with np.errstate(invalid='ignore'):
            slope, intercept, r_value, p_value, std_err = stats.linregress(t, vector2)

        return slope/(2*n_dim)

    def get_diffusion_tensor(self, state):

        vector, t = self._vector_list(state)

        tensor_x = []
        for v1 in vector:
            tensor_y = []
            for v2 in vector:
                vector2 = v1*v2
                with np.errstate(invalid='ignore'):
                    slope, intercept, r_value, p_value, std_err = stats.linregress(t, vector2)
                tensor_y.append(slope)
            tensor_x.append(tensor_y)

        return np.array(tensor_x)/2

    def get_lifetime(self, state):

        sections = self.get_ranges_from_label(state)

        t = []
        for ini, fin in sections:
            t.append(self.times[fin['position']+1] - self.times[ini['position']])

        return np.average(t)

    def get_diffusion_length_square(self, state):

        distances = []
        sections = self.get_ranges_from_label(state)
        for ini, fin in sections:
            icenter = ini['center']

            cell_states = self.centers[icenter]['cell_state'][ini['position']: fin['position']+1]
            initial = np.array(self.centers[icenter]['coordinates'][ini['position']])

            lattice_diff = np.dot(self.supercell, cell_states[-1]) - np.dot(self.supercell, cell_states[0])
            distance_vector = self.centers[icenter]['coordinates'][fin['position']] - lattice_diff - initial

            distances.append(distance_vector)

        return np.average([np.dot(vector, vector) for vector in distances])

    def get_diffusion_length_square_tensor(self, state):

        distances = []
        sections = self.get_ranges_from_label(state)
        for ini, fin in sections:
            icenter = ini['center']

            cell_states = self.centers[icenter]['cell_state'][ini['position']: fin['position']+1]
            initial = np.array(self.centers[icenter]['coordinates'][ini['position']])

            lattice_diff = np.dot(self.supercell, cell_states[-1]) - np.dot(self.supercell, cell_states[0])
            distance_vector = self.centers[icenter]['coordinates'][fin['position']] - lattice_diff - initial

            distances.append(distance_vector)

        tensor = []
        for vector in distances:
            tensor_x = []
            for v1 in vector:
                tensor_y = []
                for v2 in vector:
                    tensor_y.append(v1*v2)
                tensor_x.append(tensor_y)

            tensor.append(np.array(tensor_x))

        return np.average(tensor, axis=0)
