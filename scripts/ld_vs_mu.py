import matplotlib.pyplot as plt
from kimonet.analysis import get_diffusion_vs_mu

"""
In this script the user can get the plots of the diffusion length, constant and exciton lifetime vs the 
modulus of the transition dipole moment.

The inputs of this script are the name of the files with the diffusion results for each case.
The script reads them and returns the 4 desired values. Collecting each magnitude independently plots the
dependencies. 

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
diffusion_constant_list = []
lifetime_list = []
diffusion_length_list = []
mu_list = []

for file in file_list:
    diffusion_constant, lifetime, diffusion_length, mu = get_diffusion_vs_mu(file)
    # with get_diffusion_vs_mu function we read the file and get the desired parameters

    diffusion_constant_list.append(diffusion_constant)
    lifetime_list.append(lifetime)
    diffusion_length_list.append(diffusion_length)
    mu_list.append(mu)

##################################################
#              PLOTS
##################################################

# Ld vs mu
plt.plot(mu_list, diffusion_length_list, 'ro')
plt.xlabel('$ mu , au $')
plt.ylabel('$L_D, nm $')
plt.title('Diffusion length vs transition moment')
plt.show()

# D vs mu
plt.plot(mu_list, diffusion_constant_list, 'bo')
plt.xlabel(' $ mu, au $')
plt.ylabel('$D, nm^2 ns^{-1}$')
plt.title('Diffusion constant vs transition moment')
plt.show()

# Lifetime vs mu
plt.plot(mu_list, lifetime_list, 'go')
plt.xlabel(' $ mu, au $')
plt.ylabel('$ t, ns $')
plt.title('Exciton lifetime vs transition moment')
plt.show()
