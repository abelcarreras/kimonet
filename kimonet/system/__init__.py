import numpy as np
import itertools
import copy
from kimonet.utils import distance_vector_periodic
from kimonet.system.state import ground_state as _GS_
from kimonet.utils.combinations import get_molecules_centered_in_mol


class System(object):
    def __init__(self,
                 molecules,
                 supercell,
                 transfers=None,
                 decays=None,
                 cutoff_radius=10):

        self.molecules = molecules
        self.supercell = supercell
        self.neighbors = {}
        self.is_finished = False

        self._transfer_scheme = transfers if transfers is not None else []
        self._decay_scheme = decays if decays is not None else []

        self._cutoff_radius = cutoff_radius

        # search for states
        self._states = []
        for molecule in self.molecules:
            if molecule.state.label != _GS_.label:
                self._states.append(molecule.state)

        # speed up purpose savings
        self._gs_list = None
        self._state_neighbors = {}

        # set supercell to all states in the system
        for molecule in self.molecules:
            molecule.state.supercell = supercell

    @property
    def transfer_scheme(self):
        return self._transfer_scheme

    @transfer_scheme.setter
    def transfer_scheme(self, transfers):
        for transfer in transfers:
            transfer.supercell = self.supercell
        self._transfer_scheme = transfers

    @property
    def decay_scheme(self):
        return self._decay_scheme

    @decay_scheme.setter
    def decay_scheme(self, decays):
        for decay in decays:
            decay.supercell = self.supercell
        self._decay_scheme = decays

    @property
    def process_scheme(self):
        return self._decay_scheme + self._transfer_scheme

    @process_scheme.setter
    def process_scheme(self, processes):
        for process in processes:
            process.supercell = self.supercell
            if len(process.initial) == 1:
                self._decay_scheme.append(process)
            elif len(process.initial) == 2:
                self._transfer_scheme.append(process)
            else:
                raise Exception('Only processes involving 1 or 2 states are currently supported')

    def _reset_data(self):
        self._gs_list = None
        self._state_neighbors = {}

    def get_states(self):
        return self._states

    @property
    def cutoff_radius(self):
        return self._cutoff_radius

    @cutoff_radius.setter
    def cutoff_radius(self, cutoff):
        self._cutoff_radius = cutoff
        self._reset_data()

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
        """
        Not used for now. To be deprecated in the future

        :param ref_state:
        :return:
        """

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

        if ref_state not in self._state_neighbors:

            radius = self.cutoff_radius
            center_position = ref_state.get_coordinates_relative(self.supercell)

            def get_supercell_increments(supercell, radius):
                # TODO: This function can be optimized as a function of the particular molecule coordinates
                v = np.array(radius/np.linalg.norm(supercell, axis=1), dtype=int) + 1  # here extensive approximation
                return list(itertools.product(*[range(-i, i+1) for i in v]))

            cell_increments = get_supercell_increments(self.supercell, radius)

            # Get neighbor non ground states
            state_neighbors = []
            state_cell_incr = []
            # for state in self.get_ground_states() + self.get_states():
            for state in self.get_states():
                coordinates = state.get_coordinates()
                for cell_increment in cell_increments:
                    r_vec = distance_vector_periodic(coordinates - center_position, self.supercell, cell_increment)
                    if 0 < np.linalg.norm(r_vec) < radius:
                        state_neighbors.append(state)
                        state_cell_incr.append(list(cell_increment))

            # Include neighbor ground states
            state_neighbors_gs, state_cell_incr_gs = self.get_ground_states_improved(ref_state)
            state_cell_incr = state_cell_incr_gs + state_cell_incr
            state_neighbors = state_neighbors_gs + state_neighbors

            self._state_neighbors[ref_state] = [state_neighbors, state_cell_incr]

        return self._state_neighbors[ref_state]

    def get_ground_states(self):
        if self._gs_list is None:
            self._gs_list = []
            for mol in self.molecules:
                if (mol.state not in self._gs_list) and mol.state.label == _GS_.label:  # can be an indicator of all ok
                    # TODO: check if this works: if mol.state.label == _GS_.label:
                    self._gs_list.append(mol.state)

        return self._gs_list

    def get_ground_states_improved(self, ref_state):

        mol_list, cell_incr_list = self.get_neighbours(ref_state.get_center())

        gs_list = []
        gs_cell = []
        for mol, cell_incr in zip(mol_list, cell_incr_list):
            if mol.state.label == _GS_.label:
                gs_list.append(mol.state)
                gs_cell.append(cell_incr)

        return gs_list, gs_cell

    def get_molecule_index(self, molecule):
        return self.molecules.index(molecule)

    def reset(self):
        for molecule in self.molecules:
            molecule.set_state(_GS_.copy())
            molecule.state.supercell = self.supercell
            molecule.cell_state = np.zeros(molecule.get_dim())
        # self._centers = []
        self._reset_data()
        self.is_finished = False


    def copy(self):
        return copy.deepcopy(self)

    def get_num_molecules(self):
        return len(self.molecules)

    def get_number_of_excitations(self):
        return len(self._states)

    def add_excitation_index(self, state, index):

        state = state.copy()
        state.supercell = self.supercell

        gs_list = [s.get_center() for s in self.get_ground_states()]
        mol_list = get_molecules_centered_in_mol(self.molecules[index], gs_list, self.supercell,
                                                 size=state.size,
                                                 connected_distance=state.connected_distance)
        if mol_list is None:
            raise Exception('Not enough space is system')

        if state.label == _GS_.label:
            self._states.remove(self.molecules[index].state)
        else:
            self._states.append(state)

        for molecule in mol_list:
            molecule.set_state(state)

        self._reset_data()

    def remove_exciton(self, exciton):
        if exciton.label != _GS_.label:
            self._states.remove(exciton)
            for mol in exciton.get_molecules():
                mol.set_state(_GS_.copy())
                mol.state.supercell = self.supercell
            # if exciton in self._states:
#        else:
#            exciton.cell_state *= 0
#            exciton.reset_molecules()

        self._reset_data()

    def add_exciton(self, exciton):
        if exciton.label != _GS_.label:
            for mol in exciton.get_molecules():
                mol.set_state(exciton)

            self._states.append(exciton)
        else:
            exciton.cell_state *= 0
            exciton.reset_molecules()

        self._reset_data()

    def add_excitation_random(self, exciton_ref, n_states):
        max_cycles = len(self.molecules) * 2
        for i in range(n_states):
            for nc in range(max_cycles):

                gs_list = [s.get_center() for s in self.get_ground_states()]
                choice_molecule = np.random.choice(gs_list, 1)[0]

                exciton = exciton_ref.copy()
                exciton.supercell = self.supercell

                mol_list = get_molecules_centered_in_mol(choice_molecule, gs_list, self.supercell,
                                                         size=exciton.size,
                                                         connected_distance=exciton_ref.connected_distance)

                if mol_list is None:
                    if nc == max_cycles - 1:
                        raise Exception('Not enough space in system')
                    continue

                self._states.append(exciton)

                for molecule in mol_list:
                    molecule.set_state(exciton)

                self._reset_data()
                break

    def get_volume(self):
        return np.abs(np.linalg.det(self.supercell))

    def update(self, process):
        for initial in process.initial:
            self.remove_exciton(initial)

        for final in process.final:
            for mol in final.get_molecules():
                if final.label != _GS_.label:
                    mol.cell_state = process.cell_states[mol]
                mol.set_state(final)
            self.add_exciton(final)

        process.reset_cell_states()
        self._reset_data()

    def print_status(self):
        for i, mol in enumerate(self.molecules):
            print(i, mol.state.label, mol.state, mol.state.cell_state, mol.name, mol)
        print('----')
        for state in self.get_states():
            print(state.label, state, state.get_molecules())


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
                    supercell=[[3]])

    system.add_excitation_index(s1, 2)
    print(system.get_ground_states())

    s1 = system.get_states()[0]
    print(s1, s1.label)
    s_list = system.get_state_neighbors_copy(s1)
    for s in s_list:
        print(s.label, s.get_coordinates_absolute())
        # print(s.label, s.get_coordinates_absolute(system.supercell) - s1.get_center().get_coordinates())

    print('----')

    s_list, c_list = system.get_state_neighbors(s1)
    for s, c in zip(s_list, c_list):
        print(s.label, s.get_coordinates_absolute() - np.dot(np.array(system.supercell).T, c))
        # print(s.label, s.get_coordinates_absolute(system.supercell) - s1.get_center().get_coordinates())


    print('----')
    m_list, c_list = system.get_neighbours(s1.get_center())
    print('m_list', m_list)
    for m, c in zip(m_list, c_list):
        if m.state.label != 's1':
            # print(np.array(system.supercell).T, c, np.dot(np.array(system.supercell).T, c))
            print(m.state.label, m.state.get_coordinates() - np.dot(np.array(system.supercell).T, c), m.state.get_center().name)
