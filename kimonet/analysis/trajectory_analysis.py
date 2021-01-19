import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def normalize_cell(supercell):
    normalize = []
    for r in np.array(supercell):
        normalize.append(r/np.linalg.norm(r))
    return np.array(normalize)


class TrajectoryAnalysis:

    def __init__(self, trajectories):
        self.trajectories = trajectories
        self.n_dim = trajectories[0].get_dimension()
        self.n_traj = len(trajectories)

        self.states = set()
        for traj in trajectories:
            self.states |= traj.get_states()
        self.states = self.states
        self._points_ratio = {}
        self._segment_ratio = {}

    def __str__(self):

        txt_data = '\nTrajectory Analysis\n'
        txt_data += '------------------------------\n'
        txt_data += 'Number of trajectories: {}\n'.format(self.n_traj)
        txt_data += 'Dimension: {}\n'.format(self.n_dim)
        txt_data += 'Number of nodes: {}\n'.format(self.get_number_of_nodes())
        txt_data += 'States: {}\n'.format(self.states)

        return txt_data

    def get_number_of_nodes(self):
        return len([traj.get_number_of_nodes() for traj in self.trajectories])

    def get_states(self):
        return self.states

    def get_lifetime_ratio(self, state):
        return np.average([traj.get_lifetime_ratio(state) for traj in self.trajectories])

    def get_points_ration(self, state=None):
        if state not in self._points_ratio:
            sum_t = np.nansum([traj.get_n_points(state) for traj in self.trajectories])
            self._points_ratio[state] = [traj.get_n_points(state)/float(sum_t) for traj in self.trajectories]
        return self._points_ratio[state]

    def get_segment_ration(self, state=None):
        if state not in self._segment_ratio:
            sum_t = np.nansum([traj.get_n_segments(state) for traj in self.trajectories])
            self._segment_ratio[state] = [traj.get_n_segments(state)/float(sum_t) for traj in self.trajectories]
        return self._segment_ratio[state]


    def diffusion_coeff_tensor(self, state, unit_cell=None):
        """
        calculate the average diffusion tensor defined as:

        DiffTensor = 1/2 * <DiffLen^2> / <time>

        :param state: electronic state to analyze
        :return:
        """
        #tensor = np.nanmean([traj.get_diffusion_tensor(state) for traj in self.trajectories
        #                     if traj.get_diffusion_tensor(state) is not None], axis=0)

        tensor_list = [traj.get_diffusion_tensor(state)*s for traj, s in
                       zip(self.trajectories, self.get_points_ration(state))
                       if not np.isnan(traj.get_diffusion_tensor(state)).any()]
        tensor = np.sum(tensor_list, axis=0)

        if unit_cell is not None:
            trans_mat = normalize_cell(unit_cell)
            mat_inv = np.linalg.inv(trans_mat)
            # tensor = np.dot(mat_inv.T, tensor)
            tensor = np.dot(np.dot(mat_inv.T, tensor), mat_inv)

        return tensor

    def diffusion_length_square_tensor(self, state, unit_cell=None):
        """
        calculate the average diffusion length tensor defined as:

        DiffLenTen = 2 * DiffTensor * lifetime

        :param state: electronic state to analyze
        :return:
        """
        #dl_tensor_list = [traj.get_diffusion_length_square_tensor(state) for traj in self.trajectories
        #                  if not np.isnan(traj.get_diffusion_length_square_tensor(state)).any()]

        dl_tensor_list = [traj.get_diffusion_length_square_tensor(state)*s for traj, s in
                          zip(self.trajectories, self.get_segment_ration(state))
                          if not np.isnan(traj.get_diffusion_length_square_tensor(state)).any()]

        #diffusion_list = [traj.get_diffusion(s) * s for traj, s in
        #                  zip(self.trajectories, self.get_points_ration(state))]

        # tensor = np.average(dl_tensor_list, axis=0)
        tensor = np.sum(dl_tensor_list, axis=0)

        if unit_cell is not None:
            trans_mat = normalize_cell(unit_cell)
            mat_inv = np.linalg.inv(trans_mat)
            # tensor = np.dot(mat_inv.T, tensor)
            tensor = np.dot(np.dot(mat_inv.T, tensor), mat_inv)

        return tensor

    def diffusion_coefficient(self, state=None):
        """
        Return the average diffusion coefficient defined as:

        DiffCoeff = 1/(2*z) * <DiffLen^2>/<time>

        :return:
        """

        sum_diff = 0
        if state is None:
            for s in self.get_states():
                diffusion_list = [traj.get_diffusion(s)*s for traj, s in zip(self.trajectories, self.get_points_ration(state))]
                if not np.isnan(diffusion_list).all():
                    sum_diff += np.nansum(diffusion_list) * self.get_lifetime_ratio(s)
                    # sum_diff += np.nanmean(diffusion_list) * self.get_lifetime_ratio(s)
            return sum_diff

        return np.nansum([traj.get_diffusion(state)*s for traj, s in zip(self.trajectories, self.get_points_ration(state))])
        # return np.nanmean([traj.get_diffusion(state) for traj in self.trajectories])

    def diffusion_coefficient2(self, state=None):
        """
        Return the average diffusion coefficient defined as:

        DiffCoeff = 1/(2*z) * <DiffLen^2>/<time>

        :return:
        """

        vector = []
        times = []
        for traj in self.trajectories:
            v, ti = traj.get_vector_list(state)
            vector += list(v.T)
            times += list(ti)

        vector = np.array(vector).T
        print(vector.shape)

        n_dim, n_length = vector.shape

        vector2 = np.diag(np.dot(vector.T, vector))

        with np.errstate(invalid='ignore'):
            slope, intercept, r_value, p_value, std_err = stats.linregress(times, vector2)

        return slope/(2*n_dim)

    def lifetime(self, state=None):

        sum_diff = 0
        if state is None:
            for s in self.get_states():
                lifetime_list = [traj.get_lifetime(s)*s for traj, s in zip(self.trajectories, self.get_segment_ration(state))]
                # diffusion_list = [traj.get_lifetime(s) for traj in self.trajectories]
                if not np.isnan(lifetime_list).all():
                    # sum_diff += np.nanmean(diffusion_list) * self.get_lifetime_ratio(s)
                    sum_diff += np.nansum(lifetime_list) * self.get_lifetime_ratio(s)
            return sum_diff

        return np.nansum([traj.get_lifetime(state)*s for traj, s in zip(self.trajectories, self.get_segment_ration(state))])
        # return np.average([traj.get_lifetime(state) for traj in self.trajectories])

    def diffusion_length(self, state=None):
        """
        Return the average diffusion coefficient defined as:

        DiffLen = SQRT(2 * z * DiffCoeff * LifeTime)

        :return:
        """

        sum_diff = 0
        if state is None:
            for s in self.get_states():
                d_length_list = [traj.get_diffusion_length_square(s)*s for traj, s in zip(self.trajectories, self.get_segment_ration(state))]
                # d_length_list = [traj.get_diffusion_length_square(s) for traj in self.trajectories]
                if not np.isnan(d_length_list).all():
                    sum_diff += np.nansum(d_length_list) * self.get_lifetime_ratio(s)
                    #sum_diff += np.nanmean(d_length_list) * self.get_lifetime_ratio(s)

            return np.sqrt(sum_diff)

        # length2 = np.nanmean([traj.get_diffusion_length_square(state) for traj in self.trajectories])
        length2 =  np.nansum([traj.get_diffusion_length_square(state)*s for traj, s in zip(self.trajectories, self.get_segment_ration(state))])

        return np.sqrt(length2)

    def plot_2d(self, state=None):
        plt = None
        for traj in self.trajectories:
            plt = traj.plot_2d(state)
        return plt

    def plot_distances(self, state=None):
        plt = None
        for traj in self.trajectories:
            plt = traj.plot_distances(state)
        return plt

    def plot_exciton_density(self, state=None):

        time_max = np.max([traj.get_times()[-1] for traj in self.trajectories]) * 1.1
        t_range = np.linspace(0, time_max, 100)

        ne_interp = []
        for traj in self.trajectories:
            ne = traj.get_number_of_excitons(state)
            t = traj.get_times()
            ne_interp.append(np.interp(t_range, t, ne, right=0))

        plt.title('Averaged exciton number ({})'.format('' if state is None else state))
        plt.ylim(bottom=0, top=np.max(ne_interp))
        plt.xlim(left=0, right=time_max)
        plt.xlabel('time (ns)')
        plt.ylabel('# of excitons in supercell')
        plt.plot(t_range, np.average(ne_interp, axis=0), label='Total' if state is None else state)
        plt.legend()
        return plt

    def plot_histogram(self, state=None, normalized=False, bins=None):

        distances = []
        for traj in self.trajectories:
            d, _ = traj.get_max_distances_vs_times(state)
            distances += list(d)

        plt.title('Distances histogram  ({})'.format('' if state is None else state))
        plt.xlabel('Distance (Angs)')
        if normalized:
            plt.ylabel('Probability density (Angs^-1)')
        else:
            plt.ylabel('# of occurrences')
        try:
            plt.hist(distances, density=normalized, bins=bins)
        except AttributeError:
            plt.hist(distances, normed=normalized, bins=bins)
        return plt


def _get_diffusion_helper(trajectory, state):
    return trajectory.get_diffusion(state)


class TrajectoryAnalysisParallel(TrajectoryAnalysis):
    def __init__(self, trajectories, processors=2):
        super().__init__(trajectories)
        import concurrent.futures as futures
        self._executor = futures.ProcessPoolExecutor(max_workers=processors)
        # self._executor = futures.ThreadPoolExecutor(max_workers=processors)

    def diffusion_coefficient(self, state=None):
        """
        Return the average diffusion coefficient defined as:
        1/(2*z) * <DiffLen^2>/<time>
        *Parallel version*

        :param state: state label
        :return:
        """
        import concurrent.futures as futures

        if state is None:
            sum_diff = 0
            for s in self.get_states():
                futures_list = []
                for trajectory in self.trajectories:
                    futures_list.append(self._executor.submit(_get_diffusion_helper, trajectory, s))

                diffusion_list = []
                for f in futures.as_completed(futures_list):
                    diffusion_list.append(f.result())

                if not np.isnan(diffusion_list).all():
                    sum_diff += np.nanmean(diffusion_list) * self.get_lifetime_ratio(s)
            return sum_diff

        futures_list = []
        for trajectory in self.trajectories:
            futures_list.append(self._executor.submit(_get_diffusion_helper, trajectory, state))

        diffusion_list = []
        for f in futures.as_completed(futures_list):
            diffusion_list.append(f.result())

        return np.nanmean(diffusion_list)
