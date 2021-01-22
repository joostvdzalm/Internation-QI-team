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
    
    calibrated= (expectation_value_vector- calibration_array_B[0] ) / calibration_array_B[1]

    return calibrated




##function get readout
# CALL ONCE IN THE BEGINNING TO GET MATRIX B 
# FINDS Bd 3x4 which contains matrix B and offset B0 
def get_readout_err(experimental_backend,number_of_shots ):
    '''
    Function that returns the calibration array for a given backend.
    '''
    #DEFINE HELPING ARRAY M 
    M = np.array([[1,1], [1,-1] ])


    results=np.zeros((2,1))
    print('Calibration has begun.\n')
    n=0
    for  i in range(2):
      
            #initialise circuit
            q = qk.QuantumRegister(1)
            c = qk.ClassicalRegister(1)
            circuit = qk.QuantumCircuit(q, c)

            #initialise 0 1  state
            circuit.initialize(np.array([1-i,i]), 0)


            circuit.measure(q, c)

            # Define the experiment
            qi_job = qk.execute(circuit, backend=experimental_backend, shots=number_of_shots)
            qi_result = qi_job.result()
            
            #results
            histogram = qi_result.get_counts(circuit)

            #calculate expected_value IZ ZI ZZ
            expected_value=0
            for state, counts in histogram.items() :
                expected_value += (-1)**(int(state[0]))*int(counts)              
            
            expected_value=expected_value/number_of_shots

            results[n,:]=expected_value
            n=n+1

    Bd=1/2* np.dot(M,results)
    
    
    print('Calibration done!\n')
    return Bd