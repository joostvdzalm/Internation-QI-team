import numpy as np
from transfer_matrix_tools import  calculate_transfer_matrix,calculate_gate_fidelity

def exp_value_braket(statevector,operator):
    '''
    Function that returns mathematically calculated expectation value for given statevector and operator.
    '''
    return np.dot(  np.conjugate(statevector)   , np.dot ( operator   , statevector) )
	
	
def get_bootstrap_standard_error(number_of_qubits , number_of_shots, bootstrap_iterations , input_expected_values , output_expected_values , sv_expected_values, exp_input_expected_values): 
	'''
	Returns the standard error for the calculation of the transfer matrix.
	'''
	
	ideal_transfer_matrix=calculate_transfer_matrix(input_expected_values, sv_expected_values)
	std= 1/np.sqrt(number_of_shots)  #standard error of each exp_value  / std deviation for the bootstrap gaussian
	
	new_expected_values = np.zeros((6**number_of_qubits,2**(2*number_of_qubits)))  #new possible measurements will be added here for each iteration
	new_input_expected_values = np.zeros((6**number_of_qubits,2**(2*number_of_qubits)))
	fidelities=np.zeros(bootstrap_iterations)
	


	for n in range(bootstrap_iterations):

		for i in range(6**number_of_qubits):
			for j in range(2**(2*number_of_qubits)):
				m = output_expected_values[i,j]   # the average value of the gaussian
				new_expected_values[i,j]=  np.random.normal(m,std) 

				m = exp_input_expected_values[i,j]   # the average value of the gaussian
				new_input_expected_values[i,j]=  np.random.normal(m,std) 

		# after getting another possible realisation of the experiment we calculate the new tr_matrix and fidelity
		new_transfer_matrix = calculate_transfer_matrix(new_input_expected_values, new_expected_values) #calculate transfer matrix
		fidelities[n] = calculate_gate_fidelity(number_of_qubits , ideal_transfer_matrix , new_transfer_matrix )     #get corresponding fidelity

	return  np.std(fidelities) # std dev of realisations/ std error of fidelities 



