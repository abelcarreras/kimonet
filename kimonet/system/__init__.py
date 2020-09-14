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
                 cutoff_radius=10):

        self.molecules = molecules
        self.conditions = conditions
        self.supercell = supercell
        self.neighbors = {}
        self.is_finished = False

        self.transfer_scheme = transfers if transfers is not None else {}
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
                        jumps.append(cell_increment)

            jumps = np.array(jumps)

            self.neighbors['{}_{}'.format(ref_mol, radius)] = [neighbours, jumps]

        return self.neighbors['{}_{}'.format(ref_mol, radius)]

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
