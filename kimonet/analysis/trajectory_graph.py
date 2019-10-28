from kimonet.core.processes import Transfer
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings


class TrajectoryGraph:
    def __init__(self, system):
        """
        Stores and analyzes the information of a kinetic MC trajectory
        system: system
        """

        self.node_count = len(system.centers)

        self.graph = nx.DiGraph()

        for i, center in enumerate(system.centers):
            self.graph.add_node(i,
                                coordinates=[list(system.molecules[center].get_coordinates())],
                                state=system.molecules[center].state,
                                cell_state=[list(system.molecules[center].cell_state)],
                                time=[0],
                                event_time=0,
                                index=[center]
                                )
            # print(system.molecules[center].state)

        self.supercell = system.supercell
        self.system = system

        self.n_dim = len(system.molecules[0].get_coordinates())
        self.n_centers = len(system.centers)
        self.labels = {}
        self.times = [0]

        self.states = set()
        for center in system.centers:
            self.states.add(system.molecules[center].state)

    def _close_donor(self, change_step, node):
        # Close donor
        index = (change_step['donor'])
        node['index'].append(index)
        node['coordinates'].append(list(self.system.molecules[index].get_coordinates()))
        node['time'].append(self.times[-1] - node['event_time'])
        node['cell_state'].append(node['cell_state'][-1])
        node['finished'] = True

    def add(self, change_step, time_step):
        """
        Adds trajectory step

        :param change_step: process occurred during time_step: {donor, process, acceptor}
        :param time_step: duration of the chosen process
        """

        self.times.append(self.times[-1] + time_step)

        end_points = [node for node in self.graph.nodes
                      if len(list(self.graph.successors(node))) == 0 and not 'finished' in self.graph.nodes[node]]

        process = change_step['process']
        for i in end_points:
            # print('-> ', self.G.nodes[i]['index'][-1], change_step['donor'])
            node = self.graph.nodes[i]
            if node['index'][-1] == change_step['donor']:

                # print(process[0], process[1])
                # print('>--<', process[0][0], process[1][1])
                if type(process.initial) != str:
                    # Intermolecular process
                    if process.initial[0] != process.final[1] and process.final[1] != 'gs':
                        # s1, X  -> s2, X

                        # Close donor
                        self._close_donor(change_step, node)

                        index = change_step['acceptor']

                        self.graph.add_edge(i, self.node_count)
                        self.graph.add_node(self.node_count,
                                            coordinates=[list(self.system.molecules[index].get_coordinates())],
                                            state=self.system.molecules[index].state,
                                            cell_state=[list(self.system.molecules[index].cell_state)],
                                            #cell_state=[[0, 0]],
                                            time=[0],
                                            event_time=self.times[-1],
                                            index=[index]
                                            )
                        self.node_count += 1

                    if process.initial[0] != process.final[0] and process.final[0] != 'gs':
                        # s1, X  -> X, s2

                        # Close donor
                        self._close_donor(change_step, node)

                        index = change_step['acceptor']

                        self.graph.add_edge(i, self.node_count)
                        self.graph.add_node(self.node_count,
                                            coordinates=[list(self.system.molecules[index].get_coordinates())],
                                            state=self.system.molecules[index].state,
                                            cell_state=[list(self.system.molecules[index].cell_state)],
                                            #cell_state=[[0, 0]],
                                            time=[0],
                                            event_time=self.times[-1],
                                            index=[index]
                                            )
                        self.node_count += 1

                    if process.initial[0] == process.final[1] and process.final[1] != 'gs':
                        # s1, X  -> X, s1
                        # print('->', process.initial[0], process.final[1])
                        # print(process)
                        index = (change_step['acceptor'])
                        node['index'].append(index)
                        node['coordinates'].append(list(self.system.molecules[index].get_coordinates()))
                        # nodei['state'].append(self.system.molecules[index].electronic_state())
                        node['cell_state'].append(list(self.system.molecules[index].cell_state))
                        node['time'].append(self.times[-1] - node['event_time'])
                        # print('->', i, list(self.system.molecules[index].cell_state))

                        # print('jeje', np.linalg.norm(np.array(node['cell_state'][-1]) - np.array(node['cell_state'][-2])))
                        #if np.linalg.norm(np.array(node['cell_state'][-1]) - np.array(node['cell_state'][-2])) > 2:
                        #    print('here:', node['cell_state'][-1], node['cell_state'][-2])
                        #    print(process, node['state'])
                        #    exit()
                else:
                    # Close donor
                    self._close_donor(change_step, node)

                    # Intramolecular conversion
                    if process.initial != process.final and process.final != 'gs':
                        index = change_step['acceptor']
                        # print(process.initial, '>>>', process.final)
                        self.graph.add_edge(i, self.node_count)
                        self.graph.add_node(self.node_count,
                                            coordinates=[list(self.system.molecules[index].get_coordinates())],
                                            state=self.system.molecules[index].state,
                                            cell_state=[list(self.system.molecules[index].cell_state)],
                                            # cell_state=[[0, 0]],
                                            event_time=self.times[-1],
                                            time=[0],
                                            index=[index]
                                            )
                        self.node_count += 1

    def get_states(self):
        return self.states

    def get_dimension(self):
        return self.n_dim

    def get_graph(self):
        return self.graph

    def _vector_list(self, state):
        node_list = [node for node in self.graph.nodes if self.graph.nodes[node]['state'] == state]

        vector = []
        t = []
        for node in node_list:
            t += self.graph.nodes[node]['time']
            # print('node**', node)

            initial = np.array(self.graph.nodes[node]['coordinates'][0])
            cell_state_i = self.graph.nodes[node]['cell_state'][0]

            # print('cell**', self.G.nodes[node]['cell_state'], len(self.G.nodes[node]['cell_state']))
            for coordinate, cell_state in zip(self.graph.nodes[node]['coordinates'], self.graph.nodes[node]['cell_state']):
                lattice = np.dot(self.supercell, cell_state) - np.dot(self.supercell, cell_state_i)
                vector.append(np.array(coordinate) - lattice - initial)

        vector = np.array(vector).T
        return vector, t

    def get_diffusion(self, state):

        vector, t = self._vector_list(state)

        if not np.array(t).any():
            return 0

        n_dim, n_length = vector.shape

        vector2 = np.linalg.norm(vector, axis=0)**2  # emulate dot product in axis 0

        # plt.plot(t, vector2, 'o')
        # plt.show()
        with np.errstate(invalid='ignore'):
            slope, intercept, r_value, p_value, std_err = stats.linregress(t, vector2)

        return slope/(2*n_dim)

    def get_diffusion_tensor(self, state):

        vector, t = self._vector_list(state)

        if not np.array(t).any():
            return np.zeros((self.n_dim, self.n_dim))

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

    def get_number_of_centers(self):
        return 1

    def plot_2d(self, state=None, supercell_only=False):

        if state is None:
            node_list = [node for node in self.graph.nodes]
        else:
            node_list = [node for node in self.graph.nodes if self.graph.nodes[node]['state'] == state]

        t = []
        coordinates = []
        for node in node_list:
            t += [self.graph.nodes[node]['time'] for node in node_list]

            if supercell_only:
                coordinates += [self.graph.nodes[node]['coordinates'] for node in node_list]

            else:
                vector = []
                initial = self.graph.nodes[node]['coordinates'][0]
                for cell_state, coordinate in zip(self.graph.nodes[node]['cell_state'], self.graph.nodes[node]['coordinates']):
                    lattice = np.dot(self.supercell, cell_state)
                    vector.append(np.array(coordinate) - lattice - initial)
                coordinates += vector

        if self.get_dimension() != 2:
            raise Exception('plot_2d can only be used in 2D systems')

        coordinates = np.array(coordinates).T

        if len(coordinates) == 0:
            warnings.warn('No data for state {}'.format(state))
            return plt

        plt.plot(coordinates[0], coordinates[1], '-o')

        return plt

    def plot_distances(self, state=None):

        if state is None:
            node_list = [node for node in self.graph.nodes]
        else:
            node_list = [node for node in self.graph.nodes if self.graph.nodes[node]['state'] == state]

        t = []
        coordinates = []
        for node in node_list:
            t += self.graph.nodes[node]['time']

            vector = []
            initial = self.graph.nodes[node]['coordinates'][0]
            cell_state_i = self.graph.nodes[node]['cell_state'][0]

            for cell_state, coordinate in zip(self.graph.nodes[node]['cell_state'], self.graph.nodes[node]['coordinates']):
                lattice = np.dot(self.supercell, cell_state) - np.dot(self.supercell, cell_state_i)
                vector.append(np.array(coordinate) - lattice - initial)
                # print('lattice: ', lattice)

            # print('->', [np.linalg.norm(v, axis=0) for v in vector])
            # print('->', t)
            # plt.plot(self.graph.nodes[node]['time'], [np.linalg.norm(v, axis=0) for v in vector], '-o')

            coordinates += vector

        vector = np.array(coordinates).T

        if len(coordinates) == 0:
            warnings.warn('No data for state {}'.format(state))
            return plt

        vector = np.linalg.norm(vector, axis=0)

        # print(t)
        plt.plot(t, vector, '.')

        return plt

    def get_lifetime(self, state):

        node_list = [node for node in self.graph.nodes if self.graph.nodes[node]['state'] == state]

        if len(node_list) == 0:
            return 0

        t = [self.graph.nodes[node]['time'][-1] for node in node_list]

        return np.average(t)

    def get_lifetime_ratio(self, state):

        t_tot = self.times[-1]

        return self.get_lifetime(state)/t_tot

    def get_diffusion_length_square(self, state):

        node_list = [node for node in self.graph.nodes if self.graph.nodes[node]['state'] == state]

        dot_list = []
        for node in node_list:
            # print('node', node)


            coordinates_i = np.array(self.graph.nodes[node]['coordinates'][0])
            cell_state_i = np.array(self.graph.nodes[node]['cell_state'][0])

            coordinates_f = np.array(self.graph.nodes[node]['coordinates'][-1])
            cell_state_f = np.array(self.graph.nodes[node]['cell_state'][-1])

            # print('cell', cell_state_f, cell_state_i)
            lattice_diff = np.dot(self.supercell, cell_state_f) - np.dot(self.supercell, cell_state_i)

            vector = coordinates_f - lattice_diff - coordinates_i

            dot_list.append(np.dot(vector, vector))

        if len(dot_list) == 0:
            return np.nan

        # print(dot_list)
        # exit()
        return np.average(dot_list)

    def get_diffusion_length_square_tensor(self, state):

        node_list = [node for node in self.graph.nodes if self.graph.nodes[node]['state'] == state]

        distances = []
        for node in node_list:
            coordinates_i = np.array(self.graph.nodes[node]['coordinates'][0])
            cell_state_i = np.array(self.graph.nodes[node]['cell_state'][0])

            coordinates_f = np.array(self.graph.nodes[node]['coordinates'][-1])
            cell_state_f = np.array(self.graph.nodes[node]['cell_state'][-1])

            lattice_diff = np.dot(self.supercell, cell_state_f) - np.dot(self.supercell, cell_state_i)

            vector = coordinates_f - lattice_diff - coordinates_i

            distances.append(vector)

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

