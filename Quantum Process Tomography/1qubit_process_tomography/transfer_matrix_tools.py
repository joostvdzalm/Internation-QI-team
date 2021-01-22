import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

#1qubit process 
new_map  =   ['I','X','Y','Z']





#gate fidelity calculation
def calculate_gate_fidelity(n, Transfer_matrix_ideal,Transfer_matrix):
    d= 2 ** n
    fpro = (np.trace(np.dot(np.matrix(Transfer_matrix_ideal).transpose() , Transfer_matrix))) / d**2
    gate_fidelity= (fpro*d+1)/(d+1)
    return gate_fidelity


#transfer matrix calculation    
def calculate_transfer_matrix(input_expected_values,output_expected_values):
    inverse_input_matrix = np.linalg.pinv(np.matrix(input_expected_values.transpose()))
    tr_matrix= np.matrix(output_expected_values).transpose() * np.matrix(inverse_input_matrix)
    
    return tr_matrix


#plot transfer matrix
def plot_transfer_matrix(Transfer_matrix, plot_title):
    
    labelling=new_map
    
    
    fig, ax = plt.subplots()
    im = ax.imshow(Transfer_matrix, vmin=-1, vmax=1, interpolation='nearest', cmap=cm.RdBu )
    # We want to show all ticks...
    ax.set_xticks(np.arange(len(labelling)))
    ax.set_yticks(np.arange(len(labelling)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(labelling)
    ax.set_yticklabels(labelling)

    # Rotate the tick labels and set their alignment.
    ax.set_title(plot_title)
    plt.show()