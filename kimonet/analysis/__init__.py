import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from kimonet.analysis.trajectory_graph import TrajectoryGraph as Trajectory
from kimonet.analysis.trajectory_analysis import TrajectoryAnalysis
from kimonet.system.state import ground_state as _GS_
import warnings


def visualize_system(system, length=1):

    ndim = system.molecules[0].get_dim()
    #fig, ax = plt.subplots()
    fig = plt.figure()

    fig.suptitle('Orientation')
    if ndim == 3:
        ax = fig.gca(projection='3d')
        ax.set_zlabel('Z')
    else:
        ax = fig.gca()

    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    # plt.xlim([0, np.dot([1, 0], system.supercell[0])])
    # plt.ylim([0, np.dot([0, 1], system.supercell[1])])

    list_state_labels = np.unique([s.label for s in system.get_states()] + [_GS_.label])
    color_list = ['red', 'blue', 'green', 'orange', 'grey']

    colors = {l: np.roll(color_list, -i)[0] for i, l in enumerate(list_state_labels)}

    for i, molecule in enumerate(system.molecules):
        c = molecule.get_coordinates()
        o = molecule.get_orientation_vector()*length

        if np.linalg.norm(o) == 0:
            continue

        if ndim == 1:
            ax.quiver(c[0]-o[0]/2, 0, o[0], 0, color=colors[molecule.state.label],
                      scale=1, scale_units='xy', angles='xy')
        if ndim == 2:
            ax.quiver(c[0]-o[0]/2, c[1]-o[1]/2, o[0], o[1], color=colors[molecule.state.label],
                      scale=1, scale_units='xy', angles='xy')
            ax.text(c[0]+0.1, c[1]+0.1, '{}'.format(i), fontsize=12)
            ax.plot(c[0], c[1], 'o', color=colors[molecule.state.label])
        if ndim == 3:
            ax.quiver(c[0]-o[0]/2, c[1]-o[1]/2, c[2]-o[2]/2, o[0], o[1], o[2],
                      scale=1, scale_units='xy', angles='xy',
                      color=colors[molecule.state.label])
            # ax.quiver(c[0], c[1], c[2], o[0], o[1], o[2], length=0.1, normalize=True)
            ax.text(c[0], c[1], c[2], '{}'.format(i), fontsize=12)

    # Plot lattice vectors
    if ndim > 1:
        for lattice_vector in system.supercell:
            ax.plot(*np.array([[0]*ndim, lattice_vector]).T)

    for i in range(ndim):
        for j in range(i+1, ndim):
            ax.plot(*np.array([system.supercell[i], system.supercell[i] + system.supercell[j]]).T, color='black')
            ax.plot(*np.array([system.supercell[j], system.supercell[j] + system.supercell[i]]).T, color='black')
            if ndim == 3:
                ax.plot(*np.array([system.supercell[i]+system.supercell[j],
                                   system.supercell[0]+system.supercell[1]+system.supercell[2]]).T, color='black')

    # ax.quiverkey(q, X=0.3, Y=1.1, U=10,
    #               label='Quiver key, length = 10', labelpos='E')

    plt.show()


def plot_polar_plot(tensor_full, plane=(0, 1), title='', max=None, crystal_labels=False):

    tensor = np.array(tensor_full)[np.array(plane)].T[np.array(plane)].T

    r = []
    theta = []
    for i in np.arange(0, np.pi*2, 0.01):
        unit_vector = np.array([np.cos(i), np.sin(i)])
        r.append(np.dot(unit_vector, np.dot(tensor, unit_vector)))  # <n|D|n>
        theta.append(i)

    if max is None:
        max = np.max(np.nan_to_num(r)) * 1.2

    labels = {'cartesian': ['x', 'y', 'z'],
              'crystal': ['a', 'b', 'c']}

    if crystal_labels:
        labels_plot = [labels['crystal'][i] for i in plane]
    else:
        labels_plot = [labels['cartesian'][i] for i in plane]

    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.05, 0.8, 0.8], polar=True)
    #ax = plt.subplot(111, projection='polar')
    ax.arrow(0., 0., np.pi, max,  edgecolor='black', lw=1, zorder=5)
    ax.arrow(0., 0., 3./2*np.pi, max,  edgecolor='black', lw=1, zorder=5)
    ax.annotate("", xy=(0, max), xytext=(0, 0), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(np.pi/2, max), xytext=(0, 0), arrowprops=dict(arrowstyle="->"))
    ax.plot(theta, r)
    #plt.polar(theta, r)
    ax.set_rmax(max)
    ax.set_rticks(list(np.linspace(0.0, max, 8)))  # Less radial ticks
    ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
    ax.grid(True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ax.set_xticklabels(['{}'.format(labels_plot[0]), '', '{}'.format(labels_plot[1]), '', '', '', '', ''])

    ax.set_title(title, va='bottom')
    plt.show()
