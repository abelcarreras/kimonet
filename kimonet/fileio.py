import h5py
from kimonet.analysis.trajectory_graph import TrajectoryGraph
import pickle
import numpy as np


def store_trajectory_list(trajectory_list, filename):
    f = h5py.File(filename, 'w')

    for i, trajectory in enumerate(trajectory_list):
        grp = f.create_group('{}'.format(i))
        grp.create_dataset('graph', data=np.void(pickle.dumps(trajectory.graph)))
        grp.create_dataset('node_count', data=trajectory.node_count)
        grp.create_dataset('system', data=np.void(pickle.dumps(trajectory.system)))
        grp.create_dataset('n_dim', data=trajectory.n_dim)
        grp.create_dataset('times', data=trajectory.times)
        grp.create_dataset('states', data=np.void(pickle.dumps(trajectory.states)))
        grp.create_dataset('current_excitons', data=np.void(pickle.dumps(trajectory.current_excitons)))

    f.close()

    pass


def load_trajectory_list(filename):
    f = h5py.File(filename, 'r')

    trajectory_list = []
    for dataset in f:
        graph = pickle.loads(f[dataset]['graph'][()].tostring())
        node_count = f[dataset]['node_count'][()]
        times = f[dataset]['times'][()]
        ndim = f[dataset]['n_dim'][()]
        system = pickle.loads(f[dataset]['system'][()].tostring())
        states = pickle.loads(f[dataset]['states'][()].tostring())
        current_excitons = pickle.loads(f[dataset]['current_excitons'][()].tostring())

        trajectory = TrajectoryGraph(system)
        trajectory.graph = graph
        trajectory.node_count = node_count
        trajectory.states = states
        trajectory.current_excitons = current_excitons
        trajectory.supercell = np.array(system.supercell)
        trajectory.n_excitons = len(system.get_states())
        trajectory.times = times
        trajectory.n_dim = ndim

        trajectory_list.append(trajectory)

    return trajectory_list
