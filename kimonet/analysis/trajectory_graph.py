# from kimonet.core.processes import Transfer
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import stats
import warnings
import os
from copy import deepcopy
from kimonet.system.state import ground_state as _GS_


def count_keys_dict(dictionary, key):
    if key in dictionary:
        dictionary[key] += 1
    else:
        dictionary[key] = 1


class StateData:
    def __init__(self, state):
        self.label = state.label
        self.multiplicity = state.multiplicity
        self.size = state.size
        self.energy = state.energy

    def __repr__(self):
        return '<{}>'.format(self.label)

    def __hash__(self):
        return hash((self.label,
                     # self.energy,
                     self.multiplicity,
                     self.size))

    def __eq__(self, other):
        return hash(self) == hash(other)


class ArrayHandler():
    def __init__(self, arraystructure, data):
        self.arraystructure = arraystructure

        if data in self.arraystructure.dtype.fields:
            self.data = data
        else:
            raise Exception('Data not in array')

        # determine data position
        for i, element in enumerate(self.arraystructure.dtype.fields):
            if self.data == element:
                self.index = i

        # set initial data len
        self.data_len = len(arraystructure)

    def __str__(self):
        return str(self.arraystructure[self.data][:self.data_len])

    def __getitem__(self, item):
        return self.arraystructure[self.data][:self.data_len][item]

    def __len__(self):
        return self.data_len

    def append(self, value):

        if self.data_len >= len(self.arraystructure):
            self.arraystructure.resize((len(self.arraystructure) + 1,), refcheck=False)
            new_element = [None] * len(self.arraystructure.dtype.fields)
        else:
            new_element = list(self.arraystructure[self.data_len])

        new_element[self.index] = deepcopy(value)
        self.arraystructure[self.data_len] = tuple(new_element)
        self.data_len += 1


class TrajectoryGraph:
    def __init__(self, system):
        """
        Stores and analyzes the information of a kinetic MC trajectory
        system: system
        """

        self.node_count = len(system.get_states())

        self.graph = nx.DiGraph()

        for inode, state in enumerate(system.get_states()):
            # print('add_node', inode, state.label, state)

            self.graph.add_node(inode,
                                coordinates=[list(state.get_coordinates())],
                                state=state.label,
                                time=[0],
                                event_time=0,
                                index=[id(state)],
                                finished=False,
                                )

        self.supercell = np.array(system.supercell)
        self.system = system

        self.n_dim = len(system.molecules[0].get_coordinates())
        self.n_excitons = len(system.get_states())
        self.labels = {}
        self.times = [0]

        self.states = set()
        ce = {}
        for state in self.system.get_states():
            self.states.add(StateData(state))
            count_keys_dict(ce, state.label)

        self.current_excitons = [ce]

        # Calculations
        self._diff_coefficient = {}
        self._diff_tensor = {}
        self._diff_length_square = {}
        self._diff_length_tensor = {}
        self._coordinates_dict = {}
        self._times_dict = {}

    def _reset_calculation_data(self):
        self._diff_coefficient = {}
        self._diff_tensor = {}
        self._diff_length_square = {}
        self._diff_length_tensor = {}
        self._vector_list_info = {}
        self._coordinates_dict = {}
        self._times_dict = {}

    def _finish_node(self, inode):

        node = self.graph.nodes[inode]
        if not node['finished']:
            # index = change_step['donor']
            node['index'].append(node['index'][-1])
            node['coordinates'].append(node['coordinates'][-1])
            node['time'].append(self.times[-1] - node['event_time'])
            node['finished'] = True

    def _add_node(self, from_node, new_on_state, process_label=None):

        # if self.system.molecules[new_on_molecule].set_state(_GS_):
        #     print('Error in state: ', self.system.molecules[new_on_molecule].state.label)

        self.graph.add_edge(from_node, self.node_count, process_label=process_label)
        self.graph.add_node(self.node_count,
                            coordinates=[list(new_on_state.get_coordinates_absolute())],
                            state=new_on_state.label,
                            time=[0],
                            event_time=self.times[-1],
                            index=[id(new_on_state)],
                            finished=False
                            )
        self.node_count += 1
        return self.node_count - 1

    def _append_to_node(self, on_node, add_state):
        node = self.graph.nodes[on_node]

        node['index'].append(id(add_state))
        node['coordinates'].append(list(add_state.get_coordinates_absolute()))
        node['time'].append(self.times[-1] - node['event_time'])

    def add_step(self, process, time_step):
        """
        Adds trajectory step

        :param process: process occurred during time_step: {donor, process, acceptor}
        :param time_step: duration of the chosen process
        """

        # print('-------------------------------')
        # print('process: ', process.description)

        self.times.append(self.times[-1] + time_step)

        end_points = [node for node in self.graph.nodes
                      if len(list(self.graph.successors(node))) == 0 and not self.graph.nodes[node]['finished']]

        # print('end_points', end_points)
        # print('initial: ', process.initial)
        node_links = {}
        created_nodes = {}
        for state in process.initial:
            for inode in end_points:
                node = self.graph.nodes[inode]
                if node['index'][-1] == id(state):
                    node_links[state] = inode
                    break

        # print('node_links', node_links)
        if len(node_links) == 0:
            exit()

        for initial_state, inode in node_links.items():

            if initial_state in process.get_transport_connections():
                for final_state in process.get_transport_connections()[initial_state]:
                    # Transfer
                    # print('append:', inode, final_state.label, final_state)
                    self._append_to_node(on_node=inode,
                                         add_state=final_state)
            else:
                # splitting & merging
                self._finish_node(inode)

                for final_state in process.get_transition_connections()[initial_state]:
                    if final_state in created_nodes:
                        # print('add edge: ', inode, '-', created_nodes[final_state], final_state.label, final_state)
                        self.graph.add_edge(inode,
                                            created_nodes[final_state],
                                            process_label=process.description)
                    else:
                        # print('add_node: ', inode, '-', self.node_count, final_state.label, final_state)
                        created_nodes[final_state] = self._add_node(from_node=inode,
                                                                    new_on_state=final_state,
                                                                    process_label=process.description)

        # print('------', initial_state, [state.label for state in process.initial], [state.label for state in process.final])
        ce = {}
        for state in self.system.get_states():
            self.states.add(StateData(state))
            count_keys_dict(ce, state.label)

        self.current_excitons.append(ce)

        # Reset data
        self._reset_calculation_data()

    def _start_point_nodes(self):

        #print()
        #print('edges', list(self.graph.nodes))
        start_points = []
        #transposed_edges_list = list(zip(*self.graph.out_edges))
        for node in self.graph.nodes:
            if len(self.graph.in_edges(node)) == 0:
                start_points.append(node)

        return start_points

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

        # pos = nx.spring_layout(self.graph)
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

        return plt

    def get_states(self):
        return [s.label for s in self.states]

    def get_states_full(self):
        return list(self.states)

    def get_dimension(self):
        return self.n_dim

    def get_graph(self):
        return self.graph

    def get_simulation_times(self):
        return self.times

    def get_times(self, state=None):

        if state in self._times_dict:
            return self._times_dict[state]

        if state is None:
            self._times_dict[state] = self._vector_list_global()[1]
        else:
            self._times_dict[state] = self._vector_list(state)[1]

        return self._times_dict[state]

    def get_coordinates(self, state=None):

        if state in self._coordinates_dict:
            return self._coordinates_dict[state]

        if state is None:
            self._coordinates_dict[state] = np.array([np.array(ar).T.tolist() for ar in self._vector_list_global()[0]],
                                                     dtype=object).tolist()
            # print('--> none shape', np.array(self._coordinates_dict[state]).shape)
            # print(self._coordinates_dict[state][0])
            # print(self._coordinates_dict[state][1])

        else:
            self._coordinates_dict[state] = [np.array(ar).T.tolist() for ar in self._vector_list(state)[0]]
            # print('--> state shape', np.array(self._coordinates_dict[state]).shape)
            # print(self._coordinates_dict[state][0])
            # print(self._coordinates_dict[state][1])

        return self._coordinates_dict[state]

    def get_distances(self):
        return [np.linalg.norm(coor, axis=1).tolist() for coor in self.get_coordinates()]
        # return np.linalg.norm(self.get_coordinates(), axis=2).tolist()

    def _vector_list(self, state):

        node_list = [node for node in self.graph.nodes if self.graph.nodes[node]['state'] == state]

        vector = []
        times = []
        for node in node_list:

            if len(self.graph.nodes[node]['time']) == 1:
                continue

            times.append(self.graph.nodes[node]['time'])

            initial = np.array(self.graph.nodes[node]['coordinates'][0])
            vector.append(np.array([np.array(coordinate) - initial for coordinate in self.graph.nodes[node]['coordinates']]).T.tolist())

        return [vector, times]

    def _vector_list_global(self):

        vector_list = []
        times_list = []
        for node_list in self._trajectory_node_path():

            vector = [[0.0] * self.get_dimension()]
            times = [0]
            t_ini = self.graph.nodes[0]['time'][0]
            coor_ini = np.array([0.0] * self.get_dimension())

            for node in node_list:
                # print('t_ini', t_ini)
                times += list(np.array(self.graph.nodes[node]['time'][1:]) + t_ini)
                initial = np.array(self.graph.nodes[node]['coordinates'][0])
                # print('v ', [np.array(coordinate) - initial for coordinate in self.graph.nodes[node]['coordinates']])
                vector += [np.array(coordinate) - initial - np.array(coor_ini) for coordinate in self.graph.nodes[node]['coordinates'][1:]]
                t_ini = times[-1]
                coor_ini = vector[-1]

            vector = np.array(vector).T.tolist()

            if len(times) == 1:
                continue

            vector_list.append(vector)
            times_list.append(times)

        return vector_list, times_list

    def _trajectory_node_path(self):

        trajectory_path = []
        for ini_node in self._start_point_nodes():
            same_initial_list = []

            def _recursive(inode, internal_list):
                end_of_trajectory = True
                for e in self.graph.out_edges(inode):
                    if e[0] == inode:
                        end_of_trajectory = False
                        internal_list = list(internal_list)
                        internal_list.append(e[1])
                        _recursive(e[1], tuple(internal_list))
                        internal_list = internal_list[:-1]
                if end_of_trajectory:
                    internal_list = list(internal_list)
                    same_initial_list.append(tuple(internal_list))

            _recursive(ini_node, [ini_node])
            trajectory_path += same_initial_list

        return trajectory_path

    def get_n_subtrajectories(self, state=None):
        return len(self.get_times(state))

    def get_n_segments(self, state=None):
        # get number of subtrajectories

        if state is None:
            state_list = self.get_states()
        else:
            state_list = [state]

        n_segments = 0
        for s in state_list:
            node_list = [node for node in self.graph.nodes if self.graph.nodes[node]['state'] == s]
            n_segments += len(node_list)
        return n_segments

    def get_diffusion(self, state):
        """
        Return the average diffusion coefficient defined as:

        DiffCoeff = 1/(2*z) * <DiffLen^2>/<time>

        :return: the diffusion coefficient
        """

        if state in self._diff_coefficient:
            return self._diff_coefficient[state]

        slope_list = []
        for v, t in zip(self.get_coordinates(state), self.get_times(state)):

            vector2 = np.linalg.norm(v, axis=1)**2  # emulate dot product in axis 0
            #vector2 = np.diag(np.dot(vector.T, vector))

            #plt.plot(t, vector2, 'o', label=state)
            #plt.show()
            with np.errstate(invalid='ignore'):
                slope, intercept, r_value, p_value, std_err = stats.linregress(t, vector2)

            slope_list.append(slope)

        # plt.show()
        if len(slope_list) == 0:
            self._diff_coefficient[state] = None
        else:
            self._diff_coefficient[state] = np.mean(slope_list)/(2 * self.get_dimension())

        return self._diff_coefficient[state]

    def get_diffusion_tensor(self, state):

        if state in self._diff_tensor:
            return self._diff_tensor[state]

        tensor_x_list = []
        for v, t in zip(self.get_coordinates(state), self.get_times(state)):
            vector = np.array(v).T

            tensor_x = []
            for v1 in vector:
                tensor_y = []
                for v2 in vector:
                    #print('v1, v2:', v1, v2)
                    vector2 = np.multiply(v1, v2)
                    with np.errstate(invalid='ignore'):
                        slope, intercept, r_value, p_value, std_err = stats.linregress(t, vector2)
                    tensor_y.append(slope)
                tensor_x.append(tensor_y)

            tensor_x_list.append(np.array(tensor_x))

        if len(tensor_x_list) == 0:
            self._diff_tensor[state] = None
        else:
            self._diff_tensor[state] = np.mean(tensor_x_list, axis=0) / 2

        return self._diff_tensor[state]

    def get_number_of_cumulative_excitons(self, state=None):
        time = []
        node_count = []
        print('This is wrong!!, accumulated')
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

    def plot_2d(self, state=None, supercell_only=False, show_warnings=True):

        if state is None:
            node_list = [node for node in self.graph.nodes]
        else:
            node_list = [node for node in self.graph.nodes if self.graph.nodes[node]['state'] == state]

        if len(node_list) == 0 and show_warnings:
            warnings.warn('Exciton {} not found for plot'.format(state))

        t = []
        coordinates = []
        for node in node_list:
            t += [self.graph.nodes[node]['time'] for node in node_list]

            if supercell_only:
                coordinates += [self.graph.nodes[node]['coordinates'] for node in node_list]

            else:
                vector = []
                for coordinate in self.graph.nodes[node]['coordinates']:
                    vector.append(np.array(coordinate))
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
        plt.title('exciton trajectories ({})'.format(state if state is not None else 'All'))

        return plt

    def get_distances_vs_times(self, state=None):

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

            for coordinate in self.graph.nodes[node]['coordinates']:
                vector.append(np.array(coordinate) - initial)

            # print('->', [np.linalg.norm(v, axis=0) for v in vector])
            # print('->', t)
            # plt.plot(self.graph.nodes[node]['time'], [np.linalg.norm(v, axis=0) for v in vector], '-o')

            coordinates += vector

        vector = np.array(coordinates).T

        if len(coordinates) == 0:
            # warnings.warn('No data for state {}'.format(state))
            return [], []

        vector = np.linalg.norm(vector, axis=0)

        return vector, t

    def get_max_distances_vs_times(self, state):

        if state is None:
            node_list = [node for node in self.graph.nodes]
        else:
            node_list = [node for node in self.graph.nodes if self.graph.nodes[node]['state'] == state]

        t = []
        coordinates = []
        for node in node_list:
            t += self.graph.nodes[node]['time']

            vector = []
            initial = np.array(self.graph.nodes[node]['coordinates'][0])
            final = np.array(self.graph.nodes[node]['coordinates'][-1])
            vector.append(final - initial)

            coordinates += vector

        vector = np.array(coordinates).T

        if len(coordinates) == 0:
            # warnings.warn('No data for state {}'.format(state))
            return [], []

        vector = np.linalg.norm(vector, axis=0)

        return vector, t

    def plot_distances(self, state=None):

        vector, t = self.get_distances_vs_times(state)

        # print(t)
        plt.title('diffusion distances ({})'.format(state if state is not None else 'All'))
        plt.plot(t, vector, '.')
        plt.xlabel('Time (ns)')
        plt.ylabel('Distance (Angs)')

        return plt

    def get_lifetime(self, state):

        if len(self.get_times(state)) == 0:
            return None

        if state is None:
            return self.get_simulation_times()[-1]

        return np.mean([t[-1] for t in self.get_times(state)])

    def get_time_ratio(self, state):

        t_state = np.sum([t[-1] for t in self.get_times(state)])
        t_tot = np.sum([t[-1] for t in self.get_times(None)])

        return t_state/t_tot

    def get_diffusion_length_square(self, state):

        if state in self._diff_length_square:
            return self._diff_length_square[state]

        vector_final = [v[-1] for v in self.get_coordinates(state)]

        if len(vector_final) == 0:
            return None

        dot_list = np.linalg.norm(vector_final, axis=1)**2

        self._diff_length_square[state] = np.average(dot_list)
        return self._diff_length_square[state]

    def get_diffusion_length_square_tensor(self, state):

        if state in self._diff_length_tensor:
            return self._diff_length_tensor[state]

        vector_final = [v[-1] for v in self.get_coordinates(state)]

        tensor = []
        for vector in vector_final:
            tensor_x = []
            for v1 in vector:
                tensor_y = []
                for v2 in vector:
                    tensor_y.append(v1*v2)
                tensor_x.append(tensor_y)

            tensor.append(np.array(tensor_x))

        if len(tensor) == 0:
            return None

        self._diff_length_tensor[state] = np.average(tensor, axis=0)*self.get_dimension()
        return self._diff_length_tensor[state]


class TrajectoryGraph2(TrajectoryGraph):
    def __init__(self, system):
        """
        Stores and analyzes the information of a kinetic MC trajectory
        system: system
        """

        exit()

        self.node_count = len(system.get_states())

        self.graph = nx.DiGraph()

        if not os.path.exists('test_map'):
            os.mkdir('test_map')

        self.mapped_list = []
        for i, state in enumerate(system.get_states()):

            mem_array = np.require(np.memmap('test_map/array_{}_{}_{}'.format(id(self), os.getpid(), i),
                                             dtype=[('coordinates', object),
                                                    ('cell_state', object),
                                                    ('time', object),
                                                    ('index', object)],
                                             mode='w+', shape=(1,)),
                                   requirements=['O'])

            mem_array[:] = (
                            #state.get_center().get_coordinates(),
                            #state.get_center().cell_state,
                            0.0,
                            system.get_molecule_index(state.get_center()))

            self.graph.add_node(i,
                                coordinates=ArrayHandler(mem_array, 'coordinates'),
                                state=state.label,
                                cell_state=ArrayHandler(mem_array, 'cell_state'),
                                time=ArrayHandler(mem_array, 'time'),
                                event_time=0,
                                index=ArrayHandler(mem_array, 'index'),
                                finished=False,
                                )
            self.mapped_list.append((mem_array, 'test_map/array_{}_{}_{}'.format(id(self), os.getpid(), i)))

        self.supercell = system.supercell
        self.system = system

        self.n_dim = len(system.molecules[0].get_coordinates())
        self.n_excitons = len(system.get_states())
        self.labels = {}

        mem_array_t = np.require(np.memmap('test_map/array_{}_{}_{}'.format(id(self), os.getpid(), 't'),
                                           dtype=[('times', object)],
                                           mode='w+', shape=(1,)),
                                 requirements=['O'])

        mem_array_t[:] = (0)
        self.times = ArrayHandler(mem_array_t, 'times')
        self.mapped_list.append((mem_array_t, 'test_map/array_{}_{}_{}'.format(id(self), os.getpid(),'t')))

        self.states = set()
        ce = {}
        for state in self.system.get_states():
            self.states.add(state.label)
            count_keys_dict(ce, state.label)
        self.current_excitons = [ce]

    def _add_node(self, from_node, new_on_molecule, process_label=None):

        if self.system.molecules[new_on_molecule].set_state(_GS_):
            print('Error in state: ', self.system.molecules[new_on_molecule].state.label)
            exit()

        mem_array = np.require(np.memmap('test_map/array_{}_{}_{}'.format(id(self), os.getpid(), self.node_count),
                                         dtype=[('coordinates', object),
                                                ('cell_state', object),
                                                ('time', object),
                                                ('index', object)],
                                         mode='w+', shape=(1,)),
                               requirements=['O'])

        mem_array[:] = (self.system.molecules[new_on_molecule].get_coordinates(),
                        self.system.molecules[new_on_molecule].cell_state,
                        0.0,
                        new_on_molecule)

        self.graph.add_edge(from_node, self.node_count, process_label=process_label)
        self.graph.add_node(self.node_count,
                            coordinates=ArrayHandler(mem_array, 'coordinates'),
                            state=self.system.molecules[new_on_molecule].state.label,
                            cell_state=ArrayHandler(mem_array, 'cell_state'),
                            time=ArrayHandler(mem_array, 'time'),
                            event_time=self.times[-1],
                            index=ArrayHandler(mem_array, 'index'),
                            finished=False
                            )

        self.mapped_list.append((mem_array, 'test_map/array_{}_{}_{}'.format(id(self), os.getpid(), self.node_count)))

        self.node_count += 1

    def __del__(self):
        for mapped_array, filename in self.mapped_list:
            del mapped_array
            try:
                os.remove(filename)
            except FileNotFoundError:
                pass
