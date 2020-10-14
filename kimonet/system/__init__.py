import numpy as np
import itertools
import copy
from scipy.spatial import distance
from kimonet.utils import distance_vector_periodic
from kimonet.system.state import ground_state as _GS_


class System:
    def __init__(self,
                 molecules,
                 conditions,
                 supercell,
                 transfers=None,
                 decays=None,
                 cutoff_radius=10):

        self.molecules = molecules
        self.conditions = conditions
        self.supercell = supercell
        self.neighbors = {}
        self.is_finished = False

        self.transfer_scheme = transfers if transfers is not None else {}
        self.decay_scheme = decays if decays is not None else {}

        self._cutoff_radius = cutoff_radius

        # search for states
        self._states = []
        for molecule in self.molecules:
            if molecule.state.label != _GS_.label:
                self._states.append(molecule.state)

    def get_states(self):
        return self._states

    @property
    def cutoff_radius(self):
        return self._cutoff_radius

    @cutoff_radius.setter
    def cutoff_radius(self, cutoff):
        self._cutoff_radius = cutoff

    def get_neighbours_num(self, center):

        radius = self.cutoff_radius
        center_position = self.molecules[center].get_coordinates()

        def get_supercell_increments(supercell, radius):
            # TODO: This function can be optimized as a function of the particular molecule coordinates
            v = np.array(radius/np.linalg.norm(supercell, axis=1), dtype=int) + 1  # here extensive approximation
            return list(itertools.product(*[range(-i, i+1) for i in v]))

        cell_increments = get_supercell_increments(self.supercell, radius)

        if not '{}_{}'.format(center, radius) in self.neighbors:
            neighbours = []
            jumps = []
            for i, molecule in enumerate(self.molecules):
                coordinates = molecule.get_coordinates()
                for cell_increment in cell_increments:
                    r_vec = distance_vector_periodic(coordinates - center_position, self.supercell, cell_increment)
                    if 0 < np.linalg.norm(r_vec) < radius:
                        neighbours.append(i)
                        jumps.append(cell_increment)

            neighbours = np.array(neighbours)
            jumps = np.array(jumps)

            self.neighbors['{}_{}'.format(center, radius)] = [neighbours, jumps]

        return self.neighbors['{}_{}'.format(center, radius)]

    def get_neighbours(self, ref_mol):

        radius = self.cutoff_radius
        center_position = ref_mol.get_coordinates()

        def get_supercell_increments(supercell, radius):
            # TODO: This function can be optimized as a function of the particular molecule coordinates
            v = np.array(radius/np.linalg.norm(supercell, axis=1), dtype=int) + 1  # here extensive approximation
            return list(itertools.product(*[range(-i, i+1) for i in v]))

        cell_increments = get_supercell_increments(self.supercell, radius)

        if not '{}_{}'.format(ref_mol, radius) in self.neighbors:
            neighbours = []
            jumps = []
            for molecule in self.molecules:
                coordinates = molecule.get_coordinates()
                for cell_increment in cell_increments:
                    r_vec = distance_vector_periodic(coordinates - center_position, self.supercell, cell_increment)
                    if 0 < np.linalg.norm(r_vec) < radius:
                        neighbours.append(molecule)
                        jumps.append(list(cell_increment))

            #jumps = np.array(jumps)

            self.neighbors['{}_{}'.format(ref_mol, radius)] = [neighbours, jumps]

        return self.neighbors['{}_{}'.format(ref_mol, radius)]

    def get_state_neighbors_copy(self, ref_state):

        radius = self.cutoff_radius
        center_position = ref_state.get_coordinates_relative(self.supercell)

        def get_supercell_increments(supercell, radius):
            # TODO: This function can be optimized as a function of the particular molecule coordinates
            v = np.array(radius/np.linalg.norm(supercell, axis=1), dtype=int) + 1  # here extensive approximation
            return list(itertools.product(*[range(-i, i+1) for i in v]))

        cell_increments = get_supercell_increments(self.supercell, radius)

        neighbours = []
        for state in self.get_ground_states():
            coordinates = state.get_coordinates()
            for cell_increment in cell_increments:
                r_vec = distance_vector_periodic(coordinates - center_position, self.supercell, cell_increment)
                if 0 < np.linalg.norm(r_vec) < radius:
                    state = state.copy()
                    state.cell_state = ref_state.cell_state + cell_increment
                    neighbours.append(state)

        return neighbours

    def get_state_neighbors(self, ref_state):

        radius = self.cutoff_radius
        center_position = ref_state.get_coordinates_relative(self.supercell)

        def get_supercell_increments(supercell, radius):
            # TODO: This function can be optimized as a function of the particular molecule coordinates
            v = np.array(radius/np.linalg.norm(supercell, axis=1), dtype=int) + 1  # here extensive approximation
            return list(itertools.product(*[range(-i, i+1) for i in v]))

        cell_increments = get_supercell_increments(self.supercell, radius)

        state_neighbors = []
        state_cell_incr = []
        for state in self.get_ground_states():
            coordinates = state.get_coordinates()
            for cell_increment in cell_increments:
                r_vec = distance_vector_periodic(coordinates - center_position, self.supercell, cell_increment)
                if 0 < np.linalg.norm(r_vec) < radius:
                    state_neighbors.append(state)
                    state_cell_incr.append(list(cell_increment))

        return state_neighbors, state_cell_incr

    def get_ground_states(self):

        gs_list = []
        for mol in self.molecules:
            if (mol.state not in gs_list) and mol.state.label == _GS_.label:
                gs_list.append(mol.state)

        return gs_list

    def get_molecule_index(self, molecule):
        return self.molecules.index(molecule)

    def reset(self):
        for molecule in self.molecules:
            molecule.set_state(_GS_)
            molecule.cell_state = np.zeros(molecule.get_dim())
        # self._centers = []
        self.is_finished = False

    def copy(self):
        return copy.deepcopy(self)

    def get_num_molecules(self):
        return len(self.molecules)

    def get_number_of_excitations(self):
        return len(self._states)

    def add_excitation_index(self, type, index, do_copy=True):
        if do_copy:
            type = type.copy()

        if type.label == _GS_.label:
            self._states.remove(self.molecules[index].state)
        else:
            self._states.append(type)

        self.molecules[index].set_state(type)

    def remove_exciton(self, exciton):
        if exciton.label != _GS_.label:
            for mol in exciton.get_molecules():
                mol.set_state(_GS_)
            self._states.remove(exciton)

    def add_exciton(self, exciton):
        if exciton.label != _GS_.label:
            for mol in exciton.get_molecules():
                mol.set_state(exciton)

            self._states.append(exciton)
        else:
            exciton.reset_molecules()

    def add_excitation_random(self, type, n):
        for i in range(n):
            while True:
                num = np.random.randint(0, self.get_num_molecules())
                if self.molecules[num].state == _GS_:
                    # self.molecules[num] = type
                    self.add_excitation_index(type, num)
                    break

    def add_excitation_center(self, type):
        center_coor = np.diag(self.supercell)/2
        min = np.linalg.norm(self.supercell[0])
        index = 0
        for i, molecule in enumerate(self.molecules):
            dist = distance.euclidean(center_coor, molecule.get_coordinates())
            if dist < min:
                min = dist
                index = i

        self.add_excitation_index(type, index)

    def get_volume(self):
        return np.abs(np.linalg.det(self.supercell))

    def update(self, process):
        for initial, final in zip(process.initial, process.final_test):
            self.remove_exciton(initial)
            for mol in final.get_molecules():
                if final.label != _GS_.label:
                    mol.cell_state = process.cell_states[mol]
                mol.set_state(final)
            self.add_exciton(final)

        process.reset_cell_states()

if __name__ == '__main__':
    from kimonet.system.state import State
    from kimonet.system.molecule import Molecule

    s1 = State(label='s1', energy=1.0, multiplicity=1)
    molecule = Molecule()

    molecule1 = molecule.copy()
    molecule1.set_coordinates([0])
    molecule1.name = 'TypeA'

    molecule2 = molecule.copy()
    molecule2.set_coordinates([1])
    molecule2.name = 'TypeB'

    molecule3 = molecule.copy()
    molecule3.set_coordinates([2])
    molecule3.name = 'TypeC'

    # setup system
    system = System(molecules=[molecule1, molecule2, molecule3],
                    conditions={'custom_constant': 1},
                    supercell=[[3]])

    system.add_excitation_index(s1, 2)
    print(system.get_ground_states())

    s1 = system.get_states()[0]
    print(s1, s1.label)
    s_list = system.get_state_neighbors_copy(s1)
    for s in s_list:
        print(s.label, s.get_coordinates_absolute(system.supercell))
        # print(s.label, s.get_coordinates_absolute(system.supercell) - s1.get_center().get_coordinates())

    print('----')


    s_list, c_list = system.get_state_neighbors(s1)
    for s, c in zip(s_list, c_list):
        print(s.label, s.get_coordinates_absolute(system.supercell) - np.dot(np.array(system.supercell).T, c), state.get_center().name)
        # print(s.label, s.get_coordinates_absolute(system.supercell) - s1.get_center().get_coordinates())


    print('----')
    m_list, c_list = system.get_neighbours(s1.get_center())
    print('m_list', m_list)
    for m, c in zip(m_list, c_list):
        if m.state.label != 's1':
            # print(np.array(system.supercell).T, c, np.dot(np.array(system.supercell).T, c))
            print(m.state.label, m.state.get_coordinates() - np.dot(np.array(system.supercell).T, c), m.state.get_center().name)
