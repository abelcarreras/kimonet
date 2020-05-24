import numpy as np
from scipy.spatial import distance
import itertools
import warnings
import copy
from kimonet.utils import distance_vector_periodic

_ground_state = 'gs'

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
        self.cutoff_radius = cutoff_radius

        # search centers
        self.centers = []
        for i, molecule in enumerate(self.molecules):
            if molecule.state != _ground_state:
                self.centers.append(i)

    def get_neighbours(self, center):

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

    def reset(self):
        for molecule in self.molecules:
            molecule.state = _ground_state
            molecule.cell_state = np.zeros(molecule.get_dim())
        self.centers = []
        self.is_finished = False

    def copy(self):
        return copy.deepcopy(self)

    def get_num_molecules(self):
        return len(self.molecules)

    def get_number_of_excitations(self):
        return len(self.centers)

    def add_excitation_index(self, type, index):
        self.molecules[index].state = type
        if type == _ground_state:
            try:
                self.centers.remove(index)
            except ValueError:
                pass
        else:
            if not index in self.centers:
                self.centers.append(index)

    def add_excitation_random(self, type, n):
        for i in range(n):
            while True:
                num = np.random.randint(0, self.get_num_molecules())
                if self.molecules[num].state == _ground_state:
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
