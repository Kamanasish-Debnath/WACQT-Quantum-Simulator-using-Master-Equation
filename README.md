Dr. Kamanasish Debnath <br>
Postdoctoral Researcher <br>
Wallenberg Center for Quantum Technology <br>
Chalmers University of Technology <br>
Gothenburg, Sweden. <br>

kamanasish.debnath@chalmers.se <br>

This code can be used to simulate any quantum algorithms using the native gates available at Chalmers <br>
in the presence of noises, decoherences and ZZ interaction between the qubits. <br>
The code takes any quantum circuit as its input and returns out the final state. <br>
Measurement errors are also incorporated in the code and can be used by specifying the <br>
confusion matrix and measurement basis. <br>
  



"Chalmers_Simulator_mesolve.py" is imported as "import Chalmers_Simulator as CS" and the developed functionalities can be used. <br>
Please refer to the example folder for more details. <br>

Quantum gates modelled so far- <br>
a) Single qubit gates- Hadamard, Pauli X, Pauli Y and virtual Pauli Z <br>
b) Two qubit entangling gate - Controlled Z gate <br>
(c) Three qubit entangling gate - Controlled CZS gate

Under construction- <br>
a) iSWAP entangling gate <br>
b) Taking an open QASM file as the input rather than entering the quantum circuit manually.
