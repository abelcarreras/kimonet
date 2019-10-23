import json
import numpy as np


class TrajectoryAnalysis:

    def __init__(self, trajectories):
        self.trajectories = trajectories
        self.n_centers = trajectories[0].get_number_of_centers()
        self.n_dim = trajectories[0].get_dimension()
        self.n_traj = len(trajectories)

    def __str__(self):

        txt_data = '\nTrajectory Analysis\n'
        txt_data += '------------------------------\n'
        txt_data += 'Number of trajectories: {}\n'.format(self.n_traj)
        txt_data += 'Dimension: {}\n'.format(self.n_dim)
        txt_data += 'Number of centers: {}\n'.format(self.n_centers)

        return txt_data

    def diffusion_coeff_tensor(self):
        """
        calculate the average diffusion tensor defined as:

        DiffTensor = 1/2 * <DiffLen^2> / <time>

        :param trajectories: list of Trajectory
        :return:
        """
        return np.nanmean([traj.get_diffusion_tensor('s1') for traj in self.trajectories], axis=0)

    def diffusion_length_tensor(self):
        """
        calculate the average diffusion length tensor defined as:

        DiffLenTen = SQRT( |2 * DiffTensor * lifetime| )

        :param trajectories: list of Trajectory
        :return:
        """
        dl_tensor = np.average([traj.get_diffusion_length_square_tensor('s1') for traj in self.trajectories], axis=0)

        return np.sqrt(np.abs(dl_tensor))

    def diffusion_coefficient(self):
        """
        Return the average diffusion coefficient defined as:

        DiffCoeff = 1/(2*z) * <DiffLen^2>/<time>

        :return:
        """
        return np.nanmean([traj.get_diffusion('s1') for traj in self.trajectories])

    def lifetime(self):
        return np.average([traj.get_lifetime('s1') for traj in self.trajectories])

    def diffusion_length(self):
        """
        Return the average diffusion coefficient defined as:

        DiffLen = SQRT(2 * z * DiifCoeff * LifeTime)

        :return:
        """
        length2 = np.nanmean([traj.get_diffusion_length_square('s1') for traj in self.trajectories])
        return np.sqrt(length2)

    def plot_2d(self):
        plt = None
        for traj in self.trajectories:
            plt = traj.plot_2d(0)
        return plt

    def plot_distances(self):
        plt = None
        for traj in self.trajectories:
            plt = traj.plot_distances(0)
        return plt
