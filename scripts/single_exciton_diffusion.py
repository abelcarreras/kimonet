import json
from kimonet.analysis.diffusion import statistical_diffusion_study, diffusion_parameters
from kimonet.analysis.theorethical_functions import theoretical_diffusion_values


input_file_name = 'example_1d_simulation.json'              # name of the file with the simulation data (.json)
changed_parameter = 'mu = 0.6'                             # says which parameter is different from the reference set
output_name = 'diffusion_results_1d.json'

# the input file has to be in the same directory of the script.
# The new file (output) will be created in this same directory

with open(input_file_name, 'r') as read_file:                   # reading of the file (.json)
    simulation_data = json.load(read_file)


#   Information about the simulation conditions (defined molecule and system)
system_information = simulation_data['system_information']


#   Data from the simulation: trajectories of the system
trajectories = simulation_data['trajectories']


################################################################################################################

# theoretical diffusion values in a dictionary (only computed if the system type allows it)
diffusion_theoretical = theoretical_diffusion_values(system_information)

################################################################################################################

# results obtained from the simulation (last positions of each trajectory)
# offers 3 plots:
#       convergence of the diffusion length and exciton values to the theoretical values
#       histogram of the final distances of the exciton
diffusion_experimental = diffusion_parameters(trajectories, diffusion_theoretical, system_information)

################################################################################################################

# calculation of the diffusion constant from the statistical study of the points of the trajectory
# uses the theoretical value of the lifetime to compute the diffusion length
# plot offered:
#       points (<r²>, <t>) over the trajectories at each step
#       lineal regression of these set of values: the slope of the line is the diffusion constant.

diffusion_statistical = statistical_diffusion_study(trajectories, diffusion_theoretical, system_information)


################################################################################################################

#   The results are saved in a json format file as a nested dictionary.
#   A pretty print is also offered

print('Theorethical values:')
print(diffusion_theoretical)
print("""--------------------
        Simulation values:""")
print(diffusion_experimental)
print("""--------------------
        Statistical values:""")
print(diffusion_statistical)


diffusion_results_1d = {'conditions': system_information, 'changed_parameter': changed_parameter,
                        'theorical': diffusion_theoretical, 'experimental': diffusion_experimental,
                        'statistical': diffusion_statistical}

with open(output_name, 'w') as write_file:
    json.dump(diffusion_results_1d, write_file)






