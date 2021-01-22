#function to calibrate calculated exp_values
#inputs: 
#A: experimental vector expected_values((1,3))  \\ IZ ZI ZZ
#Bd: calibration data B 3x4
import numpy as np
import qiskit as qk
def calibrate_readout_err(expectation_value_vector , calibration_array_B):
    '''
    Function that returns calibrated expectation value vector.
    '''
    #offset vector
    b0=np.zeros((3,1))
    b0=calibration_array_B[:,0]
    #B square matrix
    B=calibration_array_B[:,1:4]
    

    #calculate calibrated values vector
    calibrated=np.zeros(3)
    calibrated=np.dot( B , np.transpose( expectation_value_vector - b0 ) )

    return calibrated




##function get readout
# CALL ONCE IN THE BEGINNING TO GET MATRIX B 
# FINDS Bd 3x4 which contains matrix B and offset B0 
def get_readout_err(experimental_backend,number_of_shots ):
    '''
    Function that returns the calibration array for a given backend.
    '''
    #DEFINE HELPING ARRAY M_msq = M_lsq = M_corr = M
    M = np.array([[1,1,1,1], [1,1,-1,-1], [1,-1,1,-1], [1,-1,-1,1]])


    results=np.zeros((4,3))
    print('Calibration has begun.\n')
    n=0
    for  i in range(2):
        for j in range(2):
            
            #initialise circuit
            q = qk.QuantumRegister(3)
            c = qk.ClassicalRegister(3)
            circuit = qk.QuantumCircuit(q, c)

            #initialise 00 01 10 11 state
            circuit.initialize(np.array([1,0]), 0)
            circuit.initialize(np.array([1-i,i]), 1)
            circuit.initialize(np.array([1-j,j]), 2)

            circuit.measure(q, c)
            # Define the experiment
            qi_job = qk.execute(circuit, backend=experimental_backend, shots=number_of_shots)
            qi_result = qi_job.result()
            
            #results
            histogram = qi_result.get_counts(circuit)
            #calculate expected_value IZ ZI ZZ
            expected_value=np.zeros(3)
            for state, counts in histogram.items() :
                expected_value[0] += (-1)**(int(state[1]))*int(counts)                 #corresponds to IZ
                expected_value[1] += (-1)**(int(state[0]))*int(counts)                 #corresponds to ZI
                expected_value[2] += (-1)**(int(state[1])+int(state[0]))*int(counts)   #corresponds to ZZ
            
            expected_value=expected_value/number_of_shots
            results[n,:]=expected_value
            n=n+1

    Bd=np.zeros((3,4))  

    for i in range(3):
        Bd[i,:]= np.dot( np.linalg.inv(M), results[:,i] )
    print(Bd)    
        
    B=Bd[:,1:4]
    B = np.linalg.inv(B)
    Bd[:,1:4]=B
    print('Calibration done!\n')
    return Bd