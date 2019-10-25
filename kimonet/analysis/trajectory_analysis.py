import numpy as np


class TrajectoryAnalysis:

    def __init__(self, trajectories):
        self.trajectories = trajectories
        self.n_centers = trajectories[0].get_number_of_centers()
        self.n_dim = trajectories[0].get_dimension()
        self.n_traj = len(trajectories)

        self.states = set()
        for traj in trajectories:
            self.states |= traj.get_states()
        self.states = self.states

    def __str__(self):

        txt_data = '\nTrajectory Analysis\n'
        txt_data += '------------------------------\n'
        txt_data += 'Number of trajectories: {}\n'.format(self.n_traj)
        txt_data += 'Dimension: {}\n'.format(self.n_dim)
        txt_data += 'Number of centers: {}\n'.format(self.n_centers)
        txt_data += 'States: {}\n'.format(self.states)

        return txt_data

    def get_states(self):
        return self.states

    def get_lifetime_ratio(self, state):
        return np.average([traj.get_lifetime_ratio(state) for traj in self.trajectories])

    def diffusion_coeff_tensor(self, state):
        """
        calculate the average diffusion tensor defined as:

        DiffTensor = 1/2 * <DiffLen^2> / <time>

        :param state: electronic state to analyze
        :return:
        """
        return np.nanmean([traj.get_diffusion_tensor(state) for traj in self.trajectories], axis=0)

    def diffusion_length_tensor(self, state):
        """
        calculate the average diffusion length tensor defined as:

        DiffLenTen = SQRT( |2 * DiffTensor * lifetime| )

        :param state: electronic state to analyze
        :return:
        """
        dl_tensor = np.average([traj.get_diffusion_length_square_tensor(state) for traj in self.trajectories], axis=0)

        return np.sqrt(np.abs(dl_tensor))

    def diffusion_coefficient(self, state=None):
        """
        Return the average diffusion coefficient defined as:

        DiffCoeff = 1/(2*z) * <DiffLen^2>/<time>

        :return:
        """

        sum_diff = 0
        if state is None:
            for s in self.get_states():
                diffusion_list = [traj.get_diffusion(s) for traj in self.trajectories]
                if not np.isnan(diffusion_list).all():
                    sum_diff += np.nanmean(diffusion_list) * self.get_lifetime_ratio(s)
            return sum_diff

        return np.nanmean([traj.get_diffusion(state) for traj in self.trajectories])

    def lifetime(self, state=None):

        sum_diff = 0
        if state is None:
            for s in self.get_states():
                diffusion_list = [traj.get_lifetime(s) for traj in self.trajectories]
                if not np.isnan(diffusion_list).all():
                    sum_diff += np.nanmean(diffusion_list) * self.get_lifetime_ratio(s)
            return sum_diff
        return np.average([traj.get_lifetime(state) for traj in self.trajectories])

    def diffusion_length(self, state=None):
        """
        Return the average diffusion coefficient defined as:

        DiffLen = SQRT(2 * z * DiifCoeff * LifeTime)

        :return:
        """

        sum_diff = 0
        if state is None:
            for s in self.get_states():
                diffusion_list = [traj.get_diffusion_length_square(s) for traj in self.trajectories]
                if not np.isnan(diffusion_list).all():
                    sum_diff += np.nanmean(diffusion_list) * self.get_lifetime_ratio(s)

            return np.sqrt(sum_diff)

        length2 = np.nanmean([traj.get_diffusion_length_square(state) for traj in self.trajectories])
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
