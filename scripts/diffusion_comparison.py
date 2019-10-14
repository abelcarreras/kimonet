import matplotlib.pyplot as plt
from kimonet.analysis import get_l2_t_from_file

"""
In this script the user can get the plot of l² vs t for different diffusion situations.
As default there is a reference set of values corresponding to the values introduced in reference_simulation in 
example_simulation directory.

Given the exciton lifetime a set of equidistant time points between 0 and the lifetime is
constructed. Then with the diffusion constant we compute a set for l². We plot both sets.
The slope of the line will be D and the final point will be (lifetime, diffusion_length).

For running this script the files and the same script has to be in the same directory.
"""

# names of the files with diffusion results
reference_file = 'diffusion_results_1d.json'
file_1 = 'diffusion_results_1d_mu_1-1.json'
file_2 = 'diffusion_results_1d_mu_0-9.json'
file_3 = 'diffusion_results_1d_mu_0-8.json'
file_4 = 'diffusion_results_1d_mu_0-7.json'
file_5 = 'diffusion_results_1d_mu_0-6.json'
file_6 = 'diffusion_results_1d_mu_0-5.json'
file_7 = 'diffusion_results_1d_mu_0-4.json'

# the files are collected in a list
file_list = [reference_file, file_1, file_2, file_3, file_4, file_5, file_6, file_7]

# a list for every studied parameter is built
time_set_list = []
squared_distance_set_list = []
diffusion_constant_list = []
changes_list = []                   # changes tells which parameter is different from the reference simulation

for file in file_list:

    time_set, squared_distance_set, diffusion_constant, changed_parameter = get_l2_t_from_file(file)
    # with get_l2_t_from_file function we read the file and get the desired parameters

    time_set_list.append(time_set)
    squared_distance_set_list.append(squared_distance_set)
    diffusion_constant_list.append(diffusion_constant)
    changes_list.append(changed_parameter)


# PLOTS l² vs t for all cases
for i in range(len(time_set_list)):

    plt.plot(time_set_list[i], squared_distance_set_list[i], label=changes_list[i]+'  $D = %.3f$ '
                                                                    % diffusion_constant_list[i])

plt.xlabel('$ Time, ns $')
plt.ylabel('$ l^2, nm $')
plt.legend()
plt.show()




