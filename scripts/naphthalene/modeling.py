from pymatgen import Lattice, Structure, Molecule
import numpy as np
from scipy.optimize import fmin
from kimonet.utils.rotation import rotate_vector
from kimonet.system.generators import crystal_system
from kimonet.analysis import visualize_system
from kimonet.system.molecule import Molecule
from kimonet.utils.units import DEBYE_TO_ANGS_EL
DEBYE_TO_AU = 0.393430


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

    return np.dot(rotate_vector([1, 0, 0], [rx, ry, rz]), objective_x)**2 * \
           np.dot(rotate_vector([0, 1, 0], [rx, ry, rz]), objective_y)**2 * \
           np.dot(rotate_vector([0, 0, 1], [rx, ry, rz]), objective_z)**2 * -1.0


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


def print_xyz(coordinates, symbols):
    print('{}\n'.format(len(coordinates)))
    for s,c in zip(symbols, coordinates):
        print('{:3}'.format(s) + '{:10.5f} {:10.5f} {:10.5f}'.format(*c))

# define lattice
lattice = Lattice.from_parameters(a=7.6778,
                                  b=5.7210,
                                  c=8.395,
                                  alpha=90.0,
                                  beta=124.55,
                                  gamma=90.0)

lattice = Lattice.from_parameters(a=8.69,
                                  b=6.01,
                                  c=8.29,
                                  alpha=90.0,
                                  beta=124.55,
                                  gamma=90.0)


#import csv
#with open('molecules.csv', newline='') as csvfile:
#    spamreader = csv.DictReader(csvfile, delimiter=',')
#    scaled_coord = []
#    for row in spamreader:
#        scaled_coord.append([float(row['Xfrac + ESD'].split('(')[0]), float(row['Yfrac + ESD'].split('(')[0]), float(row['Zfrac + ESD'].split('(')[0])])
#
#print(np.array(scaled_coord))

# coordinates of the molecules in scaled coordinates (10 molecules)
scaled_coord = [[ 0.0842 , 0.0203 , 0.3373],
                [ 0.1272 , 0.063  , 0.4615],
                [ 0.1147 , 0.1718 , 0.228 ],
                [ 0.179  , 0.3157 , 0.2791],
                [ 0.0486 , 0.1098 , 0.0371],
                [ 0.0773 , 0.263  ,-0.0797],
                [ 0.1412 , 0.4077 ,-0.0311],
                [-0.0121 ,-0.1998 , 0.2623],
                [-0.031  ,-0.3016 , 0.3377],
                [-0.0842 ,-0.0203 ,-0.3373],
                [-0.1272 ,-0.063  ,-0.4615],
                [-0.1147 ,-0.1718 ,-0.228 ],
                [-0.179  ,-0.3157 ,-0.2791],
                [-0.0486 ,-0.1098 ,-0.0371],
                [-0.0773 ,-0.263  , 0.0797],
                [-0.1412 ,-0.4077 , 0.0311],
                [ 0.0121 , 0.1998 ,-0.2623],
                [ 0.031  , 0.3016 ,-0.3377],
                [ 0.0842 , 0.0203 , 1.3373],
                [ 0.1272 , 0.063  , 1.4615],
                [ 0.1147 , 0.1718 , 1.228 ],
                [ 0.179  , 0.3157 , 1.2791],
                [ 0.0486 , 0.1098 , 1.0371],
                [ 0.0773 , 0.263  , 0.9203],
                [ 0.1412 , 0.4077 , 0.9689],
                [-0.0121 ,-0.1998 , 1.2623],
                [-0.031  ,-0.3016 , 1.3377],
                [-0.0842 ,-0.0203 , 0.6627],
                [-0.1272 ,-0.063  , 0.5385],
                [-0.1147 ,-0.1718 , 0.772 ],
                [-0.179  ,-0.3157 , 0.7209],
                [-0.0486 ,-0.1098 , 0.9629],
                [-0.0773 ,-0.263  , 1.0797],
                [-0.1412 ,-0.4077 , 1.0311],
                [ 0.0121 , 0.1998 , 0.7377],
                [ 0.031  , 0.3016 , 0.6623],
                [ 0.0842 , 1.0203 , 0.3373],
                [ 0.1272 , 1.063  , 0.4615],
                [ 0.1147 , 1.1718 , 0.228 ],
                [ 0.179  , 1.3157 , 0.2791],
                [ 0.0486 , 1.1098 , 0.0371],
                [ 0.0773 , 1.263  ,-0.0797],
                [ 0.1412 , 1.4077 ,-0.0311],
                [-0.0121 , 0.8002 , 0.2623],
                [-0.031  , 0.6984 , 0.3377],
                [-0.0842 , 0.9797 ,-0.3373],
                [-0.1272 , 0.937  ,-0.4615],
                [-0.1147 , 0.8282 ,-0.228 ],
                [-0.179  , 0.6843 ,-0.2791],
                [-0.0486 , 0.8902 ,-0.0371],
                [-0.0773 , 0.737  , 0.0797],
                [-0.1412 , 0.5923 , 0.0311],
                [ 0.0121 , 1.1998 ,-0.2623],
                [ 0.031  , 1.3016 ,-0.3377],
                [ 0.0842 , 1.0203 , 1.3373],
                [ 0.1272 , 1.063  , 1.4615],
                [ 0.1147 , 1.1718 , 1.228 ],
                [ 0.179  , 1.3157 , 1.2791],
                [ 0.0486 , 1.1098 , 1.0371],
                [ 0.0773 , 1.263  , 0.9203],
                [ 0.1412 , 1.4077 , 0.9689],
                [-0.0121 , 0.8002 , 1.2623],
                [-0.031  , 0.6984 , 1.3377],
                [-0.0842 , 0.9797 , 0.6627],
                [-0.1272 , 0.937  , 0.5385],
                [-0.1147 , 0.8282 , 0.772 ],
                [-0.179  , 0.6843 , 0.7209],
                [-0.0486 , 0.8902 , 0.9629],
                [-0.0773 , 0.737  , 1.0797],
                [-0.1412 , 0.5923 , 1.0311],
                [ 0.0121 , 1.1998 , 0.7377],
                [ 0.031  , 1.3016 , 0.6623],
                [ 1.0842 , 0.0203 , 0.3373],
                [ 1.1272 , 0.063  , 0.4615],
                [ 1.1147 , 0.1718 , 0.228 ],
                [ 1.179  , 0.3157 , 0.2791],
                [ 1.0486 , 0.1098 , 0.0371],
                [ 1.0773 , 0.263  ,-0.0797],
                [ 1.1412 , 0.4077 ,-0.0311],
                [ 0.9879 ,-0.1998 , 0.2623],
                [ 0.969  ,-0.3016 , 0.3377],
                [ 0.9158 ,-0.0203 ,-0.3373],
                [ 0.8728 ,-0.063  ,-0.4615],
                [ 0.8853 ,-0.1718 ,-0.228 ],
                [ 0.821  ,-0.3157 ,-0.2791],
                [ 0.9514 ,-0.1098 ,-0.0371],
                [ 0.9227 ,-0.263  , 0.0797],
                [ 0.8588 ,-0.4077 , 0.0311],
                [ 1.0121 , 0.1998 ,-0.2623],
                [ 1.031  , 0.3016 ,-0.3377],
                [ 1.0842 , 0.0203 , 1.3373],
                [ 1.1272 , 0.063  , 1.4615],
                [ 1.1147 , 0.1718 , 1.228 ],
                [ 1.179  , 0.3157 , 1.2791],
                [ 1.0486 , 0.1098 , 1.0371],
                [ 1.0773 , 0.263  , 0.9203],
                [ 1.1412 , 0.4077 , 0.9689],
                [ 0.9879 ,-0.1998 , 1.2623],
                [ 0.969  ,-0.3016 , 1.3377],
                [ 0.9158 ,-0.0203 , 0.6627],
                [ 0.8728 ,-0.063  , 0.5385],
                [ 0.8853 ,-0.1718 , 0.772 ],
                [ 0.821  ,-0.3157 , 0.7209],
                [ 0.9514 ,-0.1098 , 0.9629],
                [ 0.9227 ,-0.263  , 1.0797],
                [ 0.8588 ,-0.4077 , 1.0311],
                [ 1.0121 , 0.1998 , 0.7377],
                [ 1.031  , 0.3016 , 0.6623],
                [ 1.0842 , 1.0203 , 0.3373],
                [ 1.1272 , 1.063  , 0.4615],
                [ 1.1147 , 1.1718 , 0.228 ],
                [ 1.179  , 1.3157 , 0.2791],
                [ 1.0486 , 1.1098 , 0.0371],
                [ 1.0773 , 1.263  ,-0.0797],
                [ 1.1412 , 1.4077 ,-0.0311],
                [ 0.9879 , 0.8002 , 0.2623],
                [ 0.969  , 0.6984 , 0.3377],
                [ 0.9158 , 0.9797 ,-0.3373],
                [ 0.8728 , 0.937  ,-0.4615],
                [ 0.8853 , 0.8282 ,-0.228 ],
                [ 0.821  , 0.6843 ,-0.2791],
                [ 0.9514 , 0.8902 ,-0.0371],
                [ 0.9227 , 0.737  , 0.0797],
                [ 0.8588 , 0.5923 , 0.0311],
                [ 1.0121 , 1.1998 ,-0.2623],
                [ 1.031  , 1.3016 ,-0.3377],
                [ 1.0842 , 1.0203 , 1.3373],
                [ 1.1272 , 1.063  , 1.4615],
                [ 1.1147 , 1.1718 , 1.228 ],
                [ 1.179  , 1.3157 , 1.2791],
                [ 1.0486 , 1.1098 , 1.0371],
                [ 1.0773 , 1.263  , 0.9203],
                [ 1.1412 , 1.4077 , 0.9689],
                [ 0.9879 , 0.8002 , 1.2623],
                [ 0.969  , 0.6984 , 1.3377],
                [ 0.9158 , 0.9797 , 0.6627],
                [ 0.8728 , 0.937  , 0.5385],
                [ 0.8853 , 0.8282 , 0.772 ],
                [ 0.821  , 0.6843 , 0.7209],
                [ 0.9514 , 0.8902 , 0.9629],
                [ 0.9227 , 0.737  , 1.0797],
                [ 0.8588 , 0.5923 , 1.0311],
                [ 1.0121 , 1.1998 , 0.7377],
                [ 1.031  , 1.3016 , 0.6623],
                [ 0.4158 , 0.5203 ,-0.3373],
                [ 0.3728 , 0.563  ,-0.4615],
                [ 0.3853 , 0.6718 ,-0.228 ],
                [ 0.321  , 0.8157 ,-0.2791],
                [ 0.4514 , 0.6098 ,-0.0371],
                [ 0.4227 , 0.763  , 0.0797],
                [ 0.3588 , 0.9077 , 0.0311],
                [ 0.5121 , 0.3002 ,-0.2623],
                [ 0.531  , 0.1984 ,-0.3377],
                [ 0.5842 , 0.4797 , 0.3373],
                [ 0.6272 , 0.437  , 0.4615],
                [ 0.6147 , 0.3282 , 0.228 ],
                [ 0.679  , 0.1843 , 0.2791],
                [ 0.5486 , 0.3902 , 0.0371],
                [ 0.5773 , 0.237  ,-0.0797],
                [ 0.6412 , 0.0923 ,-0.0311],
                [ 0.4879 , 0.6998 , 0.2623],
                [ 0.469  , 0.8016 , 0.3377],
                [ 0.4158 , 0.5203 , 0.6627],
                [ 0.3728 , 0.563  , 0.5385],
                [ 0.3853 , 0.6718 , 0.772 ],
                [ 0.321  , 0.8157 , 0.7209],
                [ 0.4514 , 0.6098 , 0.9629],
                [ 0.4227 , 0.763  , 1.0797],
                [ 0.3588 , 0.9077 , 1.0311],
                [ 0.5121 , 0.3002 , 0.7377],
                [ 0.531  , 0.1984 , 0.6623],
                [ 0.5842 , 0.4797 , 1.3373],
                [ 0.6272 , 0.437  , 1.4615],
                [ 0.6147 , 0.3282 , 1.228 ],
                [ 0.679  , 0.1843 , 1.2791],
                [ 0.5486 , 0.3902 , 1.0371],
                [ 0.5773 , 0.237  , 0.9203],
                [ 0.6412 , 0.0923 , 0.9689],
                [ 0.4879 , 0.6998 , 1.2623],
                [ 0.469  , 0.8016 , 1.3377]]


print('lattice')
print(lattice.matrix)

symbols_monomer = ['C', 'H', 'C', 'H', 'C', 'C', 'H', 'C', 'H',
                   'C', 'H', 'C', 'H', 'C', 'C', 'H', 'C', 'H']

coor_mol1 = np.array(scaled_coord)[:18]
coor_mol2 = np.array(scaled_coord)[-36:-18]

coor_mol = np.vstack([coor_mol1, coor_mol2])

# define structures
struct1 = Structure(lattice, symbols_monomer, coor_mol1, coords_are_cartesian=False)
struct2 = Structure(lattice, symbols_monomer, coor_mol2, coords_are_cartesian=False)
struct_c = Structure(lattice, symbols_monomer * 2, coor_mol, coords_are_cartesian=False)


# define molecule
coordinates_monomer = [[ 2.4610326539,  0.7054950347, -0.0070507104],
                       [ 1.2697800226,  1.4213478618,  0.0045894884],
                       [ 0.0071248839,  0.7134976955,  0.0071917580],
                       [-1.2465927908,  1.4207541565,  0.0039025332],
                       [ 2.4498274919, -0.7358510124,  0.0046346543],
                       [ 3.2528295760,  1.2280710625, -0.0312673955],
                       [ 1.3575083440,  2.3667492466,  0.0220260183],
                       [-1.2932627225,  2.3688000888, -0.0152164523],
                       [ 3.2670227933, -1.2176289251,  0.0251089819],
                       [-2.4610326539, -0.7054950347,  0.0070507104],
                       [-1.2697800226, -1.4213478618, -0.0045894884],
                       [-0.0071248839, -0.7134976955, -0.0071917580],
                       [ 1.2465927908, -1.4207541565, -0.0039025332],
                       [-2.4498274919,  0.7358510124, -0.0046346543],
                       [-3.2528295760, -1.2280710625,  0.0312673955],
                       [-1.3575083440, -2.3667492466, -0.0220260183],
                       [ 1.2932627225, -2.3688000888,  0.0152164523],
                       [-3.2670227933,  1.2176289251, -0.0251089819]]

symbols_monomer = ['C', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H',
                   'C', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H']

#transition dipole moment of state1
# dipole = [0.0892, -0.0069, -0.000]
# dipole = [0.0749, 1.7657, 0.0134]
dipole = [0.2673, -0.0230, 0.0001]

# reference dipole
ev_dipole = [[ 9.99999137e-01, -1.30587822e-03, -1.43334210e-04],
             [ 1.30597043e-03,  9.99998939e-01,  6.45107322e-04],
             [ 1.42491626e-04, -6.45293955e-04,  9.99999782e-01]]

# transition dipole moment of state2
dipole2 = [0.0130, -0.0750, -1.7570]
# dipole2 = [-0.0130, 0.0750, 1.7570]

# reference dipole2
ev_dipole2 = [[ 1.433e-04,  1.000e+00, -1.306e-03],
              [ 6.451e-04, -1.306e-03, -1.000e+00],
              [-1.000e+00,  1.425e-04, -6.453e-04]]

dipole = get_dipole_in_basis(dipole, ev_dipole, np.identity(3))
dipole2 = get_dipole_in_basis(dipole2, ev_dipole2, np.identity(3))

# print dipole
print('\ndipole (debye)', np.array(dipole)/DEBYE_TO_AU)
print('dipole2 (debye)', np.array(dipole2)/DEBYE_TO_AU)

# plot data for visual inspection
for i, struct in enumerate([struct1, struct2]):
    position, ev = get_fragment_position_and_orientation(struct, [1] * 18)

    print_xyz(coordinates=get_dipole_in_basis(np.array(coordinates_monomer).T, ev_dipole, ev).T + np.array([position]*18),
              symbols=symbols_monomer)

    print('Molecule {}\n---------------'.format(i+1))
    params = get_rotation_angles(ev)

    print('orientation: {} {} {}'.format(*params))
    print('position', position)

    # store structure in file
    struct_c.to(filename='naphtalene.cif')
    struct_c.to(filename='POSCAR')

    scale_factor = 50
    molecule = Molecule(state_energies={'gs': 0, 's1': 1},
                        transition_moment={('s1', 'gs'): dipole*scale_factor,
                                           ('s2', 'gs'): dipole2*scale_factor},  # transition dipole moment of the molecule (Debye)
                        )

    system = crystal_system(conditions={},
                            molecule=molecule,
                            scaled_coordinates=[position],
                            unitcell=lattice.matrix,
                            dimensions=[1, 1, 1],
                            orientations=[params])

    print('TM: {}'.format(system.molecules[0].get_transition_moment(to_state='s1')))
    print('TM_test: {}'.format(get_dipole_in_basis(np.array(dipole)*DEBYE_TO_ANGS_EL*scale_factor, ev_dipole, ev)))

    visualize_system(system)
    visualize_system(system, dipole='s1')
