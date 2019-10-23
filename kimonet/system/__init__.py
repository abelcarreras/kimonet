import numpy as np
from scipy.spatial import distance
import itertools
import warnings
import copy


class System:
    def __init__(self,
                 molecules,
                 conditions,
                 supercell):

        self.molecules = molecules
        self.conditions = conditions
        self.supercell = supercell
        self.neighbors = {}
        self.is_finished = False

        # search centers
        self.centers = []
        for i, molecule in enumerate(self.molecules):
            if molecule.state != 'gs':
                self.centers.append(i)

    def get_neighbours(self, center):
        radius = self.conditions['cutoff_radius']
        center_position = self.molecules[center].get_coordinates()

        from kimonet.utils import minimum_distance_vector

        if not '{}_{}'.format(center, radius) in self.neighbors:
            neighbours = []
            jumps = []
            for i, molecule in enumerate(self.molecules):
                coordinates = molecule.get_coordinates()
                r_vec, cell_vec = minimum_distance_vector(coordinates - center_position, self.supercell)

                if 0 < np.linalg.norm(r_vec) < radius:
                    neighbours.append(i)
                    jumps.append(cell_vec)

            indexes = np.unique(neighbours, return_index=True)[1]
            neighbours = np.array(neighbours)[indexes]
            jumps = np.array(jumps)[indexes]

            self.neighbors['{}_{}'.format(center, radius)] = [neighbours, jumps]

        return self.neighbors['{}_{}'.format(center, radius)]

    def reset(self):
        for molecule in self.molecules:
            molecule.state = 'gs'
            molecule.cell_state = np.zeros(molecule.get_dim())
        self.centers = []
        self.is_finished = False

    def copy(self):
        return copy.deepcopy(self)

    def get_num_molecules(self):
        return len(self.molecules)

    def get_data(self):
        return {'molecules': self.molecules,
                'conditions': self.conditions,
                'supercell': self.supercell,
                'centres': self.get_centers()}

    def get_number_of_excitations(self):
        return len(self.centers)

    def add_excitation_index(self, type, index):
        self.molecules[index].state = type
        if type == 'gs':
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
                if self.molecules[num].state == 'gs':
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
