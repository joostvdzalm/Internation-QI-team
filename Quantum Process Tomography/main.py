from Quantum_Process_Tomography import *
"""
Now that all functionality is incorporated in the QPT class in the Qubit_Process_Tomography file all that is left to do 
is make an instance with the amount of qubits you need and call compute_pauli_transfer_matrix(). If you do this from a 
terminal, you can get a nice progressbar to show you how far the calculation is.
"""

QPT = QuantumProcessTomography(2)
QPT.compute_pauli_transfer_matrix()
QPT.plot_pauli_transfer_matrix('Two qubits simple hadamard process')

QPT.compute_theoretical_transfer_matrix()
QPT.plot_theoretical_transfer_matrix('Two qubits simple hadamard process (theoretical)')
plt.show()  # This is used so plots stay in place
