from pymatgen import Lattice, Structure, Molecule
import numpy as np
from scipy.optimize import fmin
from kimonet.utils.rotation import rotate_vector
from kimonet.system.molecule import Molecule
from kimonet.system.vibrations import MarcusModel
from kimonet.system.generators import crystal_system
from kimonet.analysis import Trajectory, visualize_system
import csv


lattice = Lattice.from_parameters(a=7.6778,
                                  b=5.7210,
                                  c=8.395,
                                  alpha=90.0,
                                  beta=124.55,
                                  gamma=90.0)

with open('Atoms.csv', newline='') as csvfile:
    spamreader = csv.DictReader(csvfile, delimiter=',')
    scaled_coord = []
    for row in spamreader:
        # print(row['Label'], row['Xfrac + ESD'].split('(')[0], row['Yfrac + ESD'].split('(')[0], row['Zfrac + ESD'].split('(')[0])
        scaled_coord.append([float(row['Xfrac + ESD'].split('(')[0]), float(row['Yfrac + ESD'].split('(')[0]), float(row['Zfrac + ESD'].split('(')[0])])

print(scaled_coord)


print('lattice')
print(lattice.matrix)

#     8.2592000000       0.0000000000       0.0000000000
#     0.0000000000       5.9835000000       0.0000000000
#    -4.6806096939       0.0000000000       7.3045323700

symbols_monomer = ['C', 'H', 'C', 'H', 'C', 'C', 'H', 'C', 'H',
                   'C', 'H', 'C', 'H', 'C', 'C', 'H', 'C', 'H']

#symbols_monomer = ['C', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H',
#                   'C', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H']


coor_mol1 = np.array(scaled_coord)[:18]
coor_mol2 = np.array(scaled_coord)[-36:-18]

coor_mol = np.vstack([coor_mol1, coor_mol2])



struct1 = Structure(lattice, symbols_monomer, coor_mol1, coords_are_cartesian=False)
struct2 = Structure(lattice, symbols_monomer, coor_mol2, coords_are_cartesian=False)
#struct_c = Structure(lattice, symbols_monomer * 2, coor_mol, coords_are_cartesian=True)
struct_c = Structure(lattice, symbols_monomer * 2, coor_mol, coords_are_cartesian=False)

def xyz_file(coordinates, symbols):
    # print(symbols_monomer)
    print(len(symbols))
    print('')
    for s, c in zip(symbols, coordinates):
        print(' {} '.format(s) + '{} {} {}'.format(*c))


def rotation(r, x1, x2, x3, y1, y2, y3, z1, z2, z3):
    rx, ry, rz = r
    objective_x = x1, x2, x3
    objective_y = y1, y2, y3
    objective_z = z1, z2, z3

    return np.dot(rotate_vector([0, 0, 1], [rx, ry, rz]), objective_x) * \
           np.dot(rotate_vector([0, 1, 0], [rx, ry, rz]), objective_y) * \
           np.dot(rotate_vector([1, 0, 0], [rx, ry, rz]), objective_z)

def get_rotation_angles(orientation_matrix):
    x_orientation = np.array(orientation_matrix)[0]
    y_orientation = np.array(orientation_matrix)[1]
    z_orientation = np.array(orientation_matrix)[2]

    return fmin(rotation, [0.0, 0.0, 0.0], args=tuple(list(x_orientation) + list(y_orientation) + list(z_orientation)), disp=False)


def get_inertia(coordinates, masses):
    """
    return inertia moments and main axis of inertia (in rows)
    """

    coordinates = np.array(coordinates)

    cm = np.average(coordinates, axis=0, weights=masses)

    inertia_tensor = np.zeros((3, 3))
    for c, m in zip(coordinates, masses):
        inertia_tensor += m * (np.dot(c-cm, c-cm) * np.identity(3) - np.outer(c-cm, c-cm))

    eval, ev = np.linalg.eigh(inertia_tensor)

    return cm, eval, ev.T


def get_fragment_position_and_orientation(coordinates, masses):
    coordinates_cart = np.array(coordinates.cart_coords)
    masses = np.array(masses)
    cm, eval, ev = get_inertia(coordinates_cart, masses)
    cm = np.average(coordinates.frac_coords, axis=0, weights=masses)

    print(eval)
    print('\nBasis crystal\n', ev)
    print(cm)

    return cm, ev


def get_dipole_in_basis(dipole, basis_dipole, basis_new):

    return np.dot(np.array(basis_new).T, np.dot(basis_dipole, dipole))


dipole = [0.0749, 1.7657, 0.0134]

ev_dipole = [[ 9.99999137e-01, -1.30587822e-03, -1.43334210e-04],
             [ 1.30597043e-03,  9.99998939e-01,  6.45107322e-04],
             [ 1.42491626e-04, -6.45293955e-04,  9.99999782e-01]]


dipole2 = [0.0130, -0.0750, -1.7570]
# dipole2 = [-0.0130, 0.0750, 1.7570]


ev_dipole2 = [[ 1.433e-04,  1.000e+00, -1.306e-03],
              [ 6.451e-04, -1.306e-03, -1.000e+00],
              [-1.000e+00,  1.425e-04, -6.453e-04]]

dipole = get_dipole_in_basis(dipole, ev_dipole, np.identity(3))
dipole2 = get_dipole_in_basis(dipole2, ev_dipole2, np.identity(3))

print('\ndipole', dipole)
print('dipole2', dipole2)

#for s,c in zip(struct1.atomic_numbers, struct1.cart_coords):
#    print(s, '{:10.5f} {:10.5f} {:10.5f}'.format(*c))

for i, struct in enumerate([struct1, struct2]):
    position, ev = get_fragment_position_and_orientation(struct, [1] * 18)

    print('Molecule {}\n---------------'.format(i+1))
    params = get_rotation_angles(ev)

    print('orientation: {} {} {}'.format(*params))
    print('position', position)

    struct_c.to(filename='naphtalene.cif')
    struct_c.to(filename='POSCAR')

    molecule = Molecule(state_energies={'gs': 0, 's1': 1},
                        vibrations=MarcusModel(),
                        transition_moment={('s1', 'gs'): dipole,
                                           ('s2', 'gs'): dipole},  # transition dipole moment of the molecule (Debye)
                        )

    system = crystal_system(conditions={},
                            molecule=molecule,
                            scaled_coordinates=[position],
                            unitcell=lattice.matrix,
                            dimensions=[1, 1, 1],
                            orientations=[params])

    visualize_system(system)
    visualize_system(system, dipole='s1')

