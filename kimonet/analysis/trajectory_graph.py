from kimonet.core.processes import Transfer
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import stats
import warnings

_ground_state_ = 'gs'


def count_keys_dict(dictionary, key):
    if key in dictionary:
        dictionary[key] += 1
    else:
        dictionary[key] = 1


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
                                index=[center],
                                finished=False,
                                )

        self.supercell = system.supercell
        self.system = system

        self.n_dim = len(system.molecules[0].get_coordinates())
        self.n_centers = len(system.centers)
        self.labels = {}
        self.times = [0]

        self.states = set()
        ce = {}
        for center in system.centers:
            state = system.molecules[center].state
            self.states.add(state)
            count_keys_dict(ce, state)
        self.current_excitons = [ce]

    def _finish_node(self, inode):

        node = self.graph.nodes[inode]
        if not node['finished']:
            # index = change_step['donor']
            node['index'].append(node['index'][-1])
            node['coordinates'].append(node['coordinates'][-1])
            node['time'].append(self.times[-1] - node['event_time'])
            node['cell_state'].append(node['cell_state'][-1])
            node['finished'] = True

    def _add_node(self, from_node, new_on_molecule, process_label=None):

        if self.system.molecules[new_on_molecule].state == _ground_state_:
            print('Error in state: ', self.system.molecules[new_on_molecule].state)
            exit()

        self.graph.add_edge(from_node, self.node_count, process_label=process_label)
        self.graph.add_node(self.node_count,
                            coordinates=[list(self.system.molecules[new_on_molecule].get_coordinates())],
                            state=self.system.molecules[new_on_molecule].state,
                            cell_state=[list(self.system.molecules[new_on_molecule].cell_state)],
                            # cell_state=[[0, 0]],
                            time=[0],
                            event_time=self.times[-1],
                            index=[new_on_molecule],
                            finished=False
                            )
        self.node_count += 1

    def _append_node(self, from_node, link_to_molecule):
        node = self.graph.nodes[from_node]

        node['index'].append(link_to_molecule)
        node['coordinates'].append(list(self.system.molecules[link_to_molecule].get_coordinates()))
        node['cell_state'].append(list(self.system.molecules[link_to_molecule].cell_state))
        node['time'].append(self.times[-1] - node['event_time'])

    def add_step(self, change_step, time_step):
        """
        Adds trajectory step

        :param change_step: process occurred during time_step: {donor, process, acceptor}
        :param time_step: duration of the chosen process
        """
        # print(change_step)
        # print(self.system.molecules[change_step['donor']].get_coordinates(), self.system.molecules[change_step['acceptor']].get_coordinates())
        # print(self.system.molecules[change_step['donor']].cell_state, self.system.molecules[change_step['acceptor']].cell_state)

        self.times.append(self.times[-1] + time_step)

        end_points = [node for node in self.graph.nodes
                      if len(list(self.graph.successors(node))) == 0 and not self.graph.nodes[node]['finished']]

        node_link = {'donor': None, 'acceptor': None}
        for inode in end_points:
            node = self.graph.nodes[inode]
            if node['index'][-1] == change_step['donor']:
                node_link['donor'] = inode
            if node['index'][-1] == change_step['acceptor']:
                node_link['acceptor'] = inode

        process = change_step['process']

        if change_step['donor'] == change_step['acceptor']:
            # Intramolecular conversion
            self._finish_node(node_link['donor'])

            # Check if not ground state
            final_state = self.system.molecules[change_step['acceptor']].state
            if final_state != _ground_state_:
                self._add_node(from_node=node_link['donor'],
                               new_on_molecule=change_step['acceptor'],
                               process_label=process.description)

        else:
            # Intermolecular process
            if process.initial[0] == process.final[1] and process.final[1] != _ground_state_:
                # s1, X  -> X, s1
                # Simple transfer
                # print('C1')
                self._append_node(from_node=node_link['donor'],
                                  link_to_molecule=change_step['acceptor'])

            if (process.initial[0] != process.final[1]
                    and process.initial[0] != _ground_state_ and process.final[1] != _ground_state_
                    and process.final[0] == _ground_state_ and process.initial[1] == _ground_state_):
                # s1, X  -> X, s2
                # Transfer with change
                # print('C2')
                self._finish_node(node_link['donor'])

                self._add_node(from_node=node_link['donor'],
                               new_on_molecule=change_step['acceptor'],
                               process_label=process.description)

            if (process.initial[0] != process.final[0] and process.initial[0] != process.final[1]
                    and process.initial[0] != _ground_state_
                    and process.final[0] != _ground_state_
                    and process.final[1] != _ground_state_
                    and process.initial[1] == _ground_state_):
                # s1, X  -> s2, s3
                # Exciton splitting
                # print('C3')
                self._finish_node(node_link['donor'])

                self._add_node(from_node=node_link['donor'],
                               new_on_molecule=change_step['donor'],
                               process_label=process.description)

                self._add_node(from_node=node_link['donor'],
                               new_on_molecule=change_step['acceptor'],
                               process_label=process.description)

            if (process.initial[0] != process.final[1] and process.initial[1] != process.final[1]
                    and process.initial[0] != _ground_state_
                    and process.initial[1] != _ground_state_
                    and process.final[0] == _ground_state_
                    and process.final[1] != _ground_state_):
                # s1, s2  ->  X, s3
                # Exciton merge type 1
                # print('C4')
                self._finish_node(node_link['donor'])
                self._finish_node(node_link['acceptor'])

                self._add_node(from_node=node_link['donor'],
                               new_on_molecule=change_step['acceptor'],
                               process_label=process.description)

                self.graph.add_edge(node_link['acceptor'], self.node_count-1, process_label=process.description)

            if (process.initial[0] != process.final[0] and process.initial[1] != process.final[0]
                    and process.initial[0] != _ground_state_
                    and process.initial[1] != _ground_state_
                    and process.final[0] != _ground_state_
                    and process.final[1] == _ground_state_):
                # s1, s2  ->  s3, X
                # Exciton merge type 2
                # print('C5')
                self._finish_node(node_link['donor'])
                self._finish_node(node_link['acceptor'])

                self._add_node(from_node=node_link['donor'],
                               new_on_molecule=change_step['donor'],
                               process_label=process.description)

                self.graph.add_edge(node_link['acceptor'], self.node_count-1, process_label=process.description)

        ce = {}
        for center in self.system.centers:
            state = self.system.molecules[center].state
            self.states.add(state)
            count_keys_dict(ce, state)

        self.current_excitons.append(ce)

    def plot_graph(self):

        # cmap = cm.get_cmap('Spectral')
        # default matplotlib color cycle list
        color_list = [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd', u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf']

        colors_map = {}
        node_map = {}
        for i, state in enumerate(self.get_states()):
            colors_map[state] = np.roll(color_list, -i)[0]
            node_map[state] = []

        for node in self.graph:
            state = self.graph.nodes[node]['state']
            node_map[state].append(node)

        #pos = nx.spring_layout(self.graph)
        pos = nx.drawing.nx_agraph.graphviz_layout(self.graph, prog='dot')
        for state in self.get_states():
            nx.draw_networkx_nodes(self.graph,
                                   pos=pos,
                                   nodelist=node_map[state],
                                   node_color=colors_map[state],
                                   label=state)
        nx.draw_networkx_edges(self.graph, pos=pos)
        nx.draw_networkx_labels(self.graph, pos=pos)
        plt.legend()

        plt.show()

    def get_states(self):
        return self.states

    def get_dimension(self):
        return self.n_dim

    def get_graph(self):
        return self.graph

    def get_times(self):
        return self.times

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
                lattice = np.dot(self.supercell.T, cell_state) - np.dot(self.supercell.T, cell_state_i)
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

    def get_number_of_cumulative_excitons(self, state=None):
        time = []
        node_count = []
        print('THis is wrong!!, accumulated')
        for node in self.graph.nodes:
            time.append(self.graph.nodes[node]['event_time'])
            if state is not None:
                if self.graph.nodes[node]['event_time'] == state:
                    node_count.append(node_count[-1]+1)
            else:
                node_count.append(node)
        return time, node_count

    def get_number_of_excitons(self, state=None):
        excitations_count = []
        for t, status in zip(self.times, self.current_excitons):
            if state is None:
                excitations_count.append(np.sum(list(status.values())))
            else:
                if state in status:
                    excitations_count.append(status[state])
                else:
                    excitations_count.append(0)

        return excitations_count

    def plot_number_of_cumulative_excitons(self, state=None):
        t, n = self.get_number_of_cumulative_excitons(state)
        plt.plot(t, n, '-o')
        return plt

    def plot_number_of_excitons(self, state=None):
        n = self.get_number_of_excitons(state)
        plt.plot(self.times, n, '-o')
        return plt


    def get_number_of_nodes(self):
        return self.graph.number_of_nodes()

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
                    lattice = np.dot(self.supercell.T, cell_state)
                    vector.append(np.array(coordinate) - lattice)
                    # print(lattice)
                coordinates += vector
                # print(vector)
                plt.plot(np.array(vector).T[0], np.array(vector).T[1], '-o')
                # plt.show()

        if self.get_dimension() != 2:
            raise Exception('plot_2d can only be used in 2D systems')

        coordinates = np.array(coordinates).T

        if len(coordinates) == 0:
            # warnings.warn('No data for state {}'.format(state))
            return plt

        # plt.plot(coordinates[0], coordinates[1], '-o')
        plt.title('exciton trajectories ({})'.format(state))

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
                lattice = np.dot(self.supercell.T, cell_state) - np.dot(self.supercell.T, cell_state_i)
                vector.append(np.array(coordinate) - lattice - initial)
                # print('lattice: ', lattice)

            # print('->', [np.linalg.norm(v, axis=0) for v in vector])
            # print('->', t)
            # plt.plot(self.graph.nodes[node]['time'], [np.linalg.norm(v, axis=0) for v in vector], '-o')

            coordinates += vector

        vector = np.array(coordinates).T

        if len(coordinates) == 0:
            # warnings.warn('No data for state {}'.format(state))
            return plt

        vector = np.linalg.norm(vector, axis=0)

        # print(t)
        plt.title('diffusion distances ({})'.format(state))
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
            lattice_diff = np.dot(self.supercell.T, cell_state_f) - np.dot(self.supercell.T, cell_state_i)

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

            lattice_diff = np.dot(self.supercell.T, cell_state_f) - np.dot(self.supercell.T, cell_state_i)

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

        # If no data for this state in this particular trajectory return nan matrix
        if len(tensor) == 0:
            return np.nan

        return np.average(tensor, axis=0)

