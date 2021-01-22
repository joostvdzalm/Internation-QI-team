import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

#mapping according to Leo
new_map  =   ['II','XI','YI','ZI','IX','IY','IZ','XX','XY','XZ','YX','YY','YZ','ZX','ZY','ZZ']

#mapping according to Chow
#new_map  =   ['II','IX','IY','IZ','XI','YI','ZI','XX','XY','XZ','YX','YY','YZ','ZX','ZY','ZZ']

def rearrange_transfer_matrix(tr_matrix):
    '''
    Rearranges matrix if the current mapping is as follows:
    ['II','IX','IY','IZ','XI','XX','XY','XZ','YI','YX','YY','YZ','ZI','ZX','ZY','ZZ']
    
    to the new_map mapping
    '''
    def direction_to_index( direction ):
        if   direction=='I':
            return 0
        elif direction=='X':
            return 1
        elif direction=='Y':
            return 2
        elif direction=='Z':
            return 3

    rearranged_tr_matrix=np.zeros((16,16))
    i=0
    for directionrow in new_map:
        row1= direction_to_index(directionrow[0])
        row2= direction_to_index(directionrow[1])
        j=0
        for directioncol in new_map:
            col1= direction_to_index(directioncol[0])
            col2= direction_to_index(directioncol[1])
            
            rearranged_tr_matrix[i,j] = tr_matrix[4*row1+row2 , 4*col1+col2]
            j+=1
        i+=1
    return rearranged_tr_matrix



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
    return rearrange_transfer_matrix(tr_matrix)

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