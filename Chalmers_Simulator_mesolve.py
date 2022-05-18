'''
Dr. Kamanasish Debnath
Postdoctoral Researcher
Wallenberg Center for Quantum Technology (WACQT)
Chalmers University of Technology
Gothenburg, Sweden
kamanasish.debnath@chalmers.se
April 2022



This code is used to simulate quantum algorithms using the quantum processor at Chalmers.
Internal functions and their usage are described as below-

1. create_system_Hamiltonian(...)
        This function returns the bare system Hamiltonian 1. 
        and list of collapse operators. 
                                                        
2. DRAGX(..)      
        This function returns the Pulse stength when using DRAG.
        This is merged with (A + A.dag()) when applying PX gate. 
        
3. DRAGX_derivative(..):
        This function returns the derivative of the pulse 



'''

import numpy as np
import math
from qutip import*
import matplotlib.pyplot as plt
import itertools as it
from qutip.qip.circuit import Gate
from qutip.qip.circuit import QubitCircuit, Gate
from qutip import tensor, basis
import itertools as it
import time
sqrt = np.sqrt
pi   = np.pi
sin = np.sin
cos = np.cos
exp = np.exp

print("The quantum gates which are modelled in this code and their notations-")
print('-'*80,)
print('Pauli X', '\t\t', 'PX')
print('Pauli Y', '\t\t', 'PY')
print('Pauli Z', '\t\t', 'PZ')
print('Hadamard', '\t\t', 'HD')
print('PI12', '\t\t\t', '1->2 transition')
print('\n')
print('Controlled Z', '\t\t', 'CZ','\t\t', 'Format:Tar_Con=[[control, target]]')
print('Controlled CZS', '\t\t', 'CCZS','\t\t', 'Format:Tar_Con=[[control, target1, target2, phi]]')
print('Sqrt Controlled CZS', '\t', 'SCCZS','\t\t', 'Format:Tar_Con=[[control, target1, target2, phi]]')


def create_system_Hamiltonian(num_qubits, num_levels, Paulis_gt, CZ_gt, CCZS_gt, Alp, Diss=[],
                              Deph =[], Texc=[], ZZ_list=[], ZZ_strength=[]):
    
    '''
    This function creates the main bare Hamiltonian of the system.
    
    Arguments-
    num_qubits      :   Number of qubits
    num_levels      :   Number of levels
    Paulis_gt       :   Single qubit gate time- Pauli X and Pauli Y
    CZ_gt           :   Gate time for two qubit gate, namely Controlled Z gate
    CCZS_gt         :   Gate time for three qubit Controlled CZs gate. Refer PRX QUANTUM 2, 040348 (2021).
    Alp             :   Non linearity in each qubit (Presently only homogeneous qubits are supported)
    Diss            :   An array of qubit lifetime (in seconds) of each qubit. len(Diss) must be equal to Nqubits
    Deph            :   An array of coherence time (in seconds) of each qubit. len(Deph) must be equal to Nqubits
    Texc            :   Operating temp of each qubit in units of (1/Hz). len(Texc) must be equal to Nqubits
    ZZ_list         :   An array of pairs of qubits which interact via cross talk. Example ZZ_list = [[0,2],[2,5]] etc.
    ZZ_stength      :   An array of ZZ strengths for each pair in ZZ_list. len(ZZ_list) = len(ZZ_strength)
    
    Returns-
    Ham             :   Bare system hamiltonian
    c_ops           :   List of collapse operators
    
    '''
    
    # Defining some global variables which will be used in all the functions
    global Nqubits, Nlevels, Nqubits, gate_time_Paulis, gate_time_CZ, gate_time_CCZS, Alpha, B, anihi_oper


    gate_time_Paulis = Paulis_gt
    gate_time_CZ = CZ_gt
    gate_time_CCZS = CCZS_gt
    Alpha = Alp
    Nlevels = num_levels
    Nqubits = num_qubits
    B = pi/gate_time_Paulis


    anihi_oper= []
    
    # We are using the fact here that the dissipation rate from |2> --> |1>
    # occurs at a rate which is double than that of |1> --> |0>. 
    # Reference Chris Warren and new manuscript in preparation.
    
    # anihi10_oper contains the dissipation operator |1> --> |0>
    # anihi21_oper contains the dissipation operator |2> --> |1>
    
    anihi10_oper = []
    anihi21_oper = []
    
    # Check that Alpha is not a list
    if type(Alpha) == list:
        print("Only identical qubits can be modelled as of now. \
        Using the first value from the list for all qubits.")
        Alpha = Alpha[0]
    
    for i in range(Nqubits):
        Operators= []
        Destroy10 = []
        Destroy21 = []
        for j in range(Nqubits):
            if i==j:
                Operators.append(destroy(Nlevels))
                Destroy10.append(basis(Nlevels,0)*basis(Nlevels,1).dag())
                Destroy21.append(basis(Nlevels,1)*basis(Nlevels,2).dag())
            else:
                Operators.append(qeye(Nlevels))
                Destroy10.append(qeye(Nlevels))
                Destroy21.append(qeye(Nlevels))
                
        anihi_oper.append(tensor(Operators))
        anihi10_oper.append(tensor(Destroy10))
        anihi21_oper.append(tensor(Destroy21))

    Hamiltonian = Qobj(np.zeros(anihi_oper[0].shape), dims= anihi_oper[0].dims)
    
    if Nlevels==2:
        print("Controlled Z and CCZS gates use third excited levels and at least 3 energy levels per \
             qubit is necessary for using entangling gate")

   
    
    # Adding the nonlinearity terms to the Hamiltonian
    # Note a factor (1/2) before the nonlinearity terms.
    
    for i in range(Nqubits):
        Hamiltonian= Hamiltonian + 0.5*Alpha*anihi_oper[i].dag()*anihi_oper[i].dag()*anihi_oper[i]*anihi_oper[i] 
        
        
        
    # Now add the ZZ-coupling in the Hamiltonian
    for i in range(len(ZZ_list)):
        ZZ_0 = ZZ_list[i][0]
        ZZ_1 = ZZ_list[i][1]

        A1= anihi_oper[ZZ_0]
        A2= anihi_oper[ZZ_1]
        Hamiltonian= Hamiltonian + ZZ_strength[i]*A1.dag()*A1*A2.dag()*A2

        
    # Add the noises using the collapse operators
    c_ops= []

    if len(Diss) != 0:
        for i in range(Nqubits):
            c_ops.append(sqrt(1/(Diss[i]))*anihi10_oper[i])
            c_ops.append(sqrt(1/(2*Diss[i]))*anihi21_oper[i])
            
    if len(Deph) != 0:
        for i in range(Nqubits):
            c_ops.append(sqrt(1/(Deph[i]))*anihi_oper[i].dag()*anihi_oper[i])
            
    if len(Texc) != 0:
        for i in range(Nqubits):
            c_ops.append(sqrt(1/(Texc[i]))*anihi_oper[i])
        
    return Hamiltonian, c_ops

  
def Pauli_times(angle):
    '''
    For a given angle of a single qubit gates such as Pauli X and Pauli Y, 
    this function returns the gate time.
    Presently this function is used only for the case when the gate time is zero.
    The gate time is determined by area of the pulse.
    Pulse used is 2B sin^2 (Bt), where B is the Rabi frequency, which is determined
    by- B = (pi/Gate time).
    
        
    Arguments-
    angle         :       Angle of the given Pauli gate
    
    Returns-
    gate time     :       Gate time for a given angle  
    
    '''
    t1 = np.linspace(0, 2*gate_time_Paulis, 10000)
    
            
    # Convert the angle between 0 and 2pi.
    if angle>(2*pi):
        angle = np.mod(angle, 2*pi)
        
    # Convert the angle to positive if angle is less than 0.
    if angle<0:
        angle = math.radians(math.degrees(angle) + 360)

    for i in range(len(t1)):
        Ang1 = (B*t1[i]) - 0.5*sin(2*B*t1[i])
        Ang2 = (B*t1[i-1]) - 0.5*sin(2*B*t1[i-1])

        if Ang1>=angle and Ang2<=angle:
            return (t1[i]+t1[i-1])/2


def Omega(ang):
    '''
    This function returns the pulse strength for a particular angle for single qubit gates.
    Note that the pulse shape is always sin^(At) which ranges from 0 to 1. 
    This function returns the prefactor (the amplitude) depending on the required angle for the single 
    qubit gate.
    '''
    A = pi/gate_time_Paulis
    integr = (gate_time_Paulis/2) - (np.sin(2*A*gate_time_Paulis)/(4*A))
    Ome = ang/integr
    return Ome
    
         
        
def DRAGX(t, ang):
    '''
    This function returns the pulse stength at a given time when applying a sigmax() type pulse.
    Here, we are using sin^2 pulse. B is the maximum Rabi frequency
    '''
    A = (pi/gate_time_Paulis)
    return Omega(ang)*(np.sin(A*t)**2)


def DRAGX_derivative(t, ang):
    '''
    This function returns the derivative of the pulse stength 
    at a given time when applying a sigmax() type pulse.
    Here, we are using sin^2 pulse.
    B is the maximum Rabi frequency and Alpha is the nonlinearity of the qubit.
    '''
    A = (pi/gate_time_Paulis)
    return  Omega(ang)*((2*A*np.cos(A*t)*np.sin(A*t))/(-2*Alpha))

    

    
    
        
def DRAGY(t, ang):
    '''
    This function returns the pulse stength 
    at a given time when applying a sigmay() type pulse.
    Here, we are using sin^2 pulse.
    B is the maximum Rabi frequency
    '''
    A = (pi/gate_time_Paulis)
    return  Omega(ang)*(np.sin(A*t)**2)




def DRAGY_derivative(t, ang):
    '''
    This function returns the derivative of the pulse stength 
    at a given time when applying a sigmay() type pulse.
    Here, we are using sin^2 pulse.
    B is the maximum Rabi frequency and Alpha is the nonlinearity of the qubit.
    '''
    A = (pi/gate_time_Paulis)
    return  Omega(ang)*((2*A*np.cos(A*t)*np.sin(A*t))/(-2*Alpha))



def PauliX(target):   
    '''
    This function returns the (A+A.dag()) operator in the total Hilbert space
    for a given target qubit.
    
    Arguments-
    target         :    Target qubit
    
    Returns-
    final_operator :    (A+A.dag()) operator in the total Hilbert space
    '''
    oper= []
    A = destroy(Nlevels)
    for i in range(Nqubits):
        if i==target:
            Operator= (A + A.dag())/2
            oper.append(Operator)        
        else:
            oper.append(qeye(Nlevels))      
    final_operator= tensor(oper)
   
    return final_operator



def PauliY(target):   
    '''
    This function returns the (1j*A - 1j*A.dag()) operator in the total Hilbert space
    for a given target qubit.
    
    Arguments-
    target         :    Target qubit
    
    Returns-
    final_operator :    (1j*A - 1j*A.dag()) operator in the total Hilbert space
    '''
    oper= []
    A = destroy(Nlevels)
    for i in range(Nqubits):
        if i==target:
            Operator= (-1j*A + 1j*A.dag())/2
            oper.append(Operator)    
        else:
            oper.append(qeye(Nlevels))      
    final_operator = tensor(oper)
    return final_operator


def PI12(target):
    oper = [] 
    One = basis(Nlevels, 1)
    Two = basis(Nlevels, 2)
    for i in range(Nqubits):
        if i==target:
            oper.append(-1j*One*Two.dag())
        else:
            oper.append(qeye(Nlevels))
    opera = tensor(oper)
    return opera



def CZ(control, target):
    '''
    This function returns the (|11><20| + hc) operator in the total Hilbert space
    for a given target qubit.
    
    Arguments-
    target         :    Target qubit
    control        :    Control qubit
    
    Returns-
    final_operator :    |11><20| operator in the total Hilbert space
    '''
    
    oper = [] 
    Zero =basis(Nlevels, 0)
    One = basis(Nlevels, 1)
    Two = basis(Nlevels, 2)
    for i in range(Nqubits):
        if i==control:
            oper.append(One*Two.dag())
        elif i==target:
            oper.append(One*Zero.dag())
        else:
            oper.append(qeye(Nlevels))
    opera = tensor(oper)
    return opera


def CCZS(control, target1, target2):
    
    '''
    This function returns the two operators for three qubit gate Controlled CZs 
    on which the drives lambda_1 and lambda_2 can be applied with an exponential 
    factor to account for the nonlinearity of the third energy level. 
    Refer PRX QUANTUM 2, 040348 (2021).
    
    Arguments-
    control        :      control qubit (q0 in the main article)
    target1        :      first target qubit (q1 in the main article)
    target2        :      second target qubit (q2 in the main article)
    
    Returns-
    oper_lambda1   :      |110><200| + |111><201| equivalent of the main article 
                          in the total Hilbert space.
                          
    oper_lambda2   :      |101><200| + |111><210| equivalent of the main article
                          in the total Hilbert space.
       
    '''

    l0 = basis(Nlevels, 0)
    l1 = basis(Nlevels, 1)
    l2 = basis(Nlevels, 2)

    oper_lambda1 = []
    oper_lambda2 = []
    for i in range(Nqubits):

        if i == control:
            o1 = l1*l2.dag() 
            o2 = l1*l2.dag() 
            oper_lambda1.append(o1)
            oper_lambda2.append(o2)


        elif i == target1:
            o1 = l1*l0.dag() 
            o2 = l0*l0.dag() + l1*l1.dag()
            oper_lambda1.append(o1)
            oper_lambda2.append(o2)


        elif i == target2:
            o1 = l0*l0.dag() + l1*l1.dag()
            o2 = l1*l0.dag()
            oper_lambda1.append(o1)
            oper_lambda2.append(o2)


        else:
            oper_lambda1.append((qeye(Nlevels)))
            oper_lambda2.append((qeye(Nlevels)))


    oper_lambda1 = tensor(oper_lambda1)
    oper_lambda2 = tensor(oper_lambda2)
    
    return oper_lambda1, oper_lambda2

            
        
def virtual_Z_gate(dm, angle, target):
    
    '''
    This function applies virtual Pauli Z gate by direct application of a Liouvillian
    to a density matrix.
    
    
    Arguments-
    dm                : Initial state as density matrix
    angle             : Angle of pauli Z gate
    target            : Target qubit
    
    
    Returns-
    final_state       : Final state (density matrix) after applying Pauli Z gate.
    
    '''
    
    qc = QubitCircuit(N=1)
    qc.add_gate("RZ", targets=0, arg_value = angle)
    Z = qc.propagators()[0]
    Z3 = np.zeros((Nlevels,Nlevels), dtype = complex)
    
    Z3[0,0] = Z[0,0]
    Z3[0,1] = Z[0,1]
    Z3[1,0] = Z[1,0]   
    Z3[1,1] = Z[1,1]
    
    for i in range(2,Nlevels):
        Z3[i,i] = 1
    
    Z7 = []
    for i in range(Nqubits):
        if i == target:
            Z7.append(Qobj(Z3))
        else:
            Z7.append(qeye(Nlevels))
    
    if dm.type == 'oper':
        Z3 = to_super(tensor(Z7))

        final_state = operator_to_vector(dm)
        final_state = Z3*final_state
        final_state = vector_to_operator(final_state)
        return final_state
    
    else:
        Z3 = tensor(Z7)*dm
        return (Z3)



def Final_state(dm, oper):
    '''
    This function applies an operator to a density matrix and returns the final density matrix.
    Useful for density matrices when multiple convertion between operators and vectors are required.
    '''
    final_dm= operator_to_vector(dm)
    final_dm = oper*final_state
    final_dm = vector_to_operator(final_state)
    return final_dm



def pulse_hamiltonians(gate, TC, angle, npoints, measurement = False):
    
    '''
    QobjEvo function of qutip does not take different time slices (tlist) as inputs, which 
    makes it difficult to apply different pulses to different qubits simultaneously. 
    
    This function creates Hamiltonians as quantum objects with different time dependent coefficients 
    but same ``tlist``.
    
    ``tlist`` will correspond to 1000 points from t_start= 0 to t_final= maximum gate time.
    
    
    Arguments- 
    gate   :   Array of strings which specify which gates to apply
    TC     :   Array of target and control qubits. 
    angle  :   Angle of the Pauli gates. 
    npoints:   Number of points in the ``tlist``
    
    
    Returns-
    FHam   :   Final Hamiltonian as the form of a qutip Qobj (excluding the bare time independent Hamiltonian)   
    '''
    
    
    
    # Calculate an array of gate times and maximum gate time

    Gate_times = []
    for i in range(len(gate)):
            
        if gate[i] == 'PX' or gate[i]== 'PY':
            Gate_times.append(gate_time_Paulis)
        
        elif gate[i] == 'CZ':
            Gate_times.append(gate_time_CZ)
            
        elif gate[i] == 'HD':
            Gate_times.append(gate_time_Paulis)
            
        elif gate[i] == 'PZ':
            Gate_times.append(Pauli_times(0))
            
        elif gate[i] == 'I':
            Gate_times.append(Pauli_times(0))
            
        elif gate[i] == 'CCZS':
            Gate_times.append(gate_time_CCZS)
            
        elif gate[i] == 'SCCZS':
            Gate_times.append(gate_time_CCZS*0.5)
            
        elif gate[i] == 'PI12':
            Gate_times.append(gate_time_Paulis)
            
        else:
            print("-"*100)
            print("ERROR in the Gate inputs !! Check the Gate Type")
            print("-"*100)
        
    max_gate_time = np.max(Gate_times)
    tlist = np.linspace(0, max_gate_time, npoints)
    CZ_expo = exp(1j*Alpha*tlist)
        
        
     # Calculate and store the time dependent parts as quantum object
    
    FHam = []
    for i in range(len(gate)):
        gate_time = Gate_times[i]
        
        # Pauli X gate
        if gate[i] == 'PX':
            DragPauliX = DRAGX(tlist, angle[i])  
            DragPauliX_der = DRAGX_derivative(tlist, angle[i])
            DragPauliY = DRAGX(tlist, angle[i]) 
            DragPauliY_der = DRAGY_derivative(tlist, angle[i])
            
            TE = gate_time-tlist
            DRAG_X = DragPauliX * np.heaviside(TE, 0)
            DRAG_Y = DragPauliX_der * np.heaviside(TE, 0)
            PX_Hamiltonian =  PauliX(TC[i])
            PY_Hamiltonian =  PauliY(TC[i])
            FHam.append(QobjEvo([[PX_Hamiltonian, DRAG_X],[PY_Hamiltonian, DRAG_Y]], tlist = tlist))
            
            
        # Pauli Y gate    
        elif gate[i] == 'PY':
            DragPauliX = DRAGX(tlist, angle[i])  
            DragPauliX_der = DRAGX_derivative(tlist, angle[i])
            DragPauliY = DRAGX(tlist, angle[i]) 
            DragPauliY_der = DRAGY_derivative(tlist, angle[i])                        
            TE = gate_time-tlist
            DRAG_Y = DragPauliY * np.heaviside(TE, 0)
            DRAG_X = DragPauliY_der * np.heaviside(TE, 0)            
            PX_Hamiltonian =  -PauliX(TC[i])
            PY_Hamiltonian =   PauliY(TC[i])
            FHam.append(QobjEvo([[PY_Hamiltonian, DRAG_Y],[PX_Hamiltonian, DRAG_X]], tlist = tlist))
                
         
        # Controlled Z gate
        elif gate[i] == 'CZ':
            Pulse_strength = (pi/gate_time_CZ) 
            TE = gate_time-tlist
            Expo = Pulse_strength*CZ_expo* np.heaviside(TE, 0)
            ExpoC = Pulse_strength*np.conjugate(CZ_expo)* np.heaviside(TE, 0)
            opera = CZ(TC[i][0],TC[i][1])
            FHam.append(QobjEvo([[opera, Expo], [opera.dag(), ExpoC]], tlist = tlist))
            
           
        # Controlled CZS three qubit gate
        elif gate[i] == 'CCZS' or gate[i] == 'SCCZS':
            Om_CZS = pi/(sqrt(2)*gate_time_CCZS) # Rabi frequency
            TE = gate_time-tlist
            
            Expo = Om_CZS*CZ_expo*np.heaviside(TE, 0)      # CZ_expo is without negative sign so Expo is exp(1j*alpha*t)
            ExpoC = Om_CZS*np.conjugate(CZ_expo)*np.heaviside(TE, 0)
            
            oper1, oper2 = CCZS(TC[i][0],TC[i][1],TC[i][2])
            
            '''
            In the next line, we create the QobjEvo in the following manner-
            oper1 = |110><200| + |111><201|
            oper2 = |101><200| + |111><210|
            
            [exp(1j*alpha*tlist)*oper1] + [exp(-1j*alpha*tlist)*oper1.dag()] ----> levels in which lambda1 acts
            
            
            [exp(i phi)*exp(1j*alpha*tlist)*oper2] + [exp(-i phi)*exp(-1j*alpha*tlist)*oper2.dag()] ----> levels 
            in which lambda2 acts.
            
            Refer to the  PRX QUANTUM 2, 040348 (2021) for more details.
                       
            '''
            phi = TC[i][3]
            phi_CCZS = -np.exp(1j*phi)
            FHam.append(QobjEvo([[oper1, Expo], [oper1.dag(), ExpoC], \
                                 [phi_CCZS*oper2, Expo], [np.conjugate(phi_CCZS)*oper2.dag(), ExpoC]], tlist = tlist))

               
        
        # Hadamard Gate
        elif gate[i] == 'HD':           
            TE = gate_time-tlist
            DragPauliX = DRAGX(tlist, angle[i])  
            DragPauliX_der = DRAGX_derivative(tlist, angle[i])
            DragPauliY = DRAGX(tlist, angle[i]) 
            DragPauliY_der = DRAGY_derivative(tlist, angle[i])   
            
            
            DRAG_Y = DragPauliY * np.heaviside(TE, 0)
            DRAG_X = DragPauliY_der * np.heaviside(TE, 0)
            PX_Hamiltonian =  PauliX(TC[i])
            PY_Hamiltonian =  PauliY(TC[i])
            FHam.append(QobjEvo([[PY_Hamiltonian, DRAG_Y],[PX_Hamiltonian, DRAG_X]], tlist = tlist))

        
        elif gate[i]=='PI12':
            Pulse_strength = (pi/gate_time_Paulis) 
            TE = gate_time-tlist
            Expo = 0.5*Pulse_strength * CZ_expo*np.heaviside(TE, 0)
            ExpoC = 0.5*Pulse_strength * np.conjugate(CZ_expo)*np.heaviside(TE, 0)
            
            oper =  PI12(TC[i])
            FHam.append(QobjEvo([[oper, Expo], [oper.dag(), ExpoC]], tlist = tlist))
            
        
        
        
        # Incase the inputs are unity (only for measurement part)
        elif gate[i] == 'I' and measurement == True:
            unitary_operator = []
            for k in range(Nqubits):
                unitary_operator.append(qeye(Nlevels))
            unitary_operator = tensor(unitary_operator)
            FHam.append(QobjEvo([unitary_operator, np.ones(len(tlist))], tlist = tlist))
            
            
        elif gate[i] == 'PZ' and measurement == True:
            unitary_operator = []
            for k in range(Nqubits):
                unitary_operator.append(qeye(Nlevels))
            unitary_operator = tensor(unitary_operator)
            FHam.append(QobjEvo([unitary_operator, np.ones(len(tlist))], tlist = tlist))
            
    return  FHam, tlist     



def Execute(Hamiltonian, c_ops, Info, Ini):
    '''
    This function executes the quantum circuit and returns the final state.
    
    Arguments- 
    Hamiltonian   : Bare Hamiltonian of the system
    C_ops         : List of collapse operators
    Info          : The class gate which includes the types of gate, target, control, angle etc
    Ini           : The initial state
    
    Returns-
    FState        : The final state
    
    '''
    
    Tsteps = []
    for i in range(len(Info)):
        
        # Get the QobjEvo for each time dependent Hamiltonians
        npoints = 5000
        gate =  np.array(Info[i].name)
        TC   =  np.array(Info[i].Tar_Con)
        angle = np.array(Info[i].angle)
        
        H1, tlist = pulse_hamiltonians(gate, TC, angle, npoints)
        H2 = sum(H1) + Hamiltonian

        final_dm = mesolve(H2, Ini, tlist, c_ops, e_ops = [], options = Options(store_final_state=True, \
                                                                                atol= 1e-10, rtol=1e-10))
        dm = final_dm.final_state
        if 'HD' in gate:
            index_HD = np.where(gate == 'HD')[0]
            for k, index in enumerate(index_HD):
                final_dm = virtual_Z_gate(dm, np.pi, TC[index])
                dm = final_dm
            
        
        if 'PZ' in gate:
            index_PZ = np.where(gate == 'PZ')[0]
            for k, index in enumerate(index_PZ):
                final_dm = virtual_Z_gate(dm, angle[index], TC[index])
                dm = final_dm
            
        
        Ini = dm            
    return Ini

    
    
    
    
def Measurement(Hamiltonian, c_ops, Ini, Info, CM, coeff):
    '''
    This function returns the diagonal elements (population) of the density 
    matrix for a given set of measurement gates and confusion matrix.
    Inputs of 'gate' must be same as in other cases.
    
    Arguments-
    Hamiltonian  :  Bare Hamiltonian of the system
    Ini          :  Initial state before the measurement process begins
    gate         :  List of gate in the same procedure
    CM           :  Confusion matris in the computational subspace
    coeff        :  Coefficients in front of each measurement term in the cost function
    
    
    Returns-
    probabilities:  Diagonal elements of the density matrix subjected to the confusion
                    matrix in the computational subspace.
    '''
 
    
    probabilities = []
    
    for i in range(len(Info)):
        
        gate =  np.array(Info[i].name)
        TC   =  np.array(Info[i].Tar_Con)
        angle = np.array(Info[i].angle)
        npoints = 500
        H1, tlist = pulse_hamiltonians(gate, TC, angle, npoints, measurement = True)
        
        H2 = sum(H1) + Hamiltonian
        final_dm = mesolve(H2, Ini, tlist, c_ops, e_ops = [], options = Options(store_final_state=True))
        
        # This truncates the Hilbert space and returns the final state in the computational subspace. 
        # No normalization is done since populations outside the computational subspace is lost.
        # _3to2levels returns an array of populations in differnt states.
        state_in_comp_space = _3to2levels(final_dm.final_state)
        ps = (CM*state_in_comp_space)
        
        probabilities.append(coeff[i]*final_pop(ps, gate))
        
    return np.sum(np.real(probabilities))  # Since probabilities, so only real part. 
                                           # And summed over all other measurements
        
        
    
    
    
def _3to2levels(dm):
    '''
    This function returns the density matrix in the computational subspace.
    
    Arguments-
    dm          :      Density matrix in the total Hilbert space
    
    Returns-
    Prob_array  :      Array of probabilities in the computational space
    
    '''


    levels= list(map(",".join, it.product(*[map(str, range(Nlevels))])))
    states = ["".join(seq) for seq in it.product(levels, repeat=Nqubits)]
    Prob_array = []
    counter = 0
    for i in states:
        if '2' not in i:
            Prob_array.append(dm[counter, counter])
        counter = counter + 1
    
    Prob_array = np.array(Prob_array)
    norm = 1/np.sum(Prob_array)
    return Prob_array
            

    
    
def final_pop(probs, gate):
    '''
    Depending on the measurements, this function returns the cost function.
    
    
    Example-
    For 2 qubits, the populations in the computational basis is given by
    P00, P01, P10, P11
    
    The following terms are returned depending on the measurement operators
    
    I,I     :     P00 + P01 + P10 + P11
    I,PZ    :     P00 - P01 + P10 - P11
    PZ,I    :     P00 + P01 - P10 - P11
    PZ,PZ   :     P00 - P01 - P10 + P11
    
    and so on....
    
    
    Arguments-
    probs       :      Probabilities in each of the possible states in the computational basis
    gate        :      Measurement gates
    
    
    Returns-
    sum         :      Returns (P00__ ± P01__ ± P10__ ± P11__) (sign depends on the measurment operator)
        
    '''
    
    ops = []
    for i in gate:
        if i=='PX' or i=='PY' or i=='PZ':
            ops.append(sigmaz())
            
        elif i == 'I':
            ops.append(qeye(2))
            
            
    ops= tensor(ops).diag()

    sum = 0
    for i, j in enumerate(ops):
        sum = sum + j*probs[i]

    return sum 
       
    
def Time_dynamics(Hamiltonian, c_ops, Info, Ini, e_ops = []):
    '''
    This function is used when the time dynamics during the quantum algorithm is needed.
    If e_ops is not supplied the function returns the density function at different time steps.
    
    
    Arguments-
    
    Hamiltonian      :     Bare hamiltonian of the system
    c_ops            :     Collapse operators
    Info             :     The class gate which includes the types of gate, target, control, angle etc.
    Ini              :     Initial state of the system (ket or density matrix)
    e_ops            :     Operators whose expectation values are desired.
    
    Returns-
    
    expect           :      Expectation values of the operators supplied
    states           :      If e_ops = [], then it returns the density matrices at different times.
    tlist            :      Time points correspond to the states.
    
    '''
    
    
    
    Tsteps = []
    states = []
    for i in range(len(Info)):
        
        npoints = 2000
        gate =  np.array(Info[i].name)
        TC   =  np.array(Info[i].Tar_Con)
        angle = np.array(Info[i].angle)
        H1, tlist = pulse_hamiltonians(gate, TC, angle, npoints)
        
        H2 = sum(H1) + Hamiltonian
            
        final_dm = mesolve(H2, Ini, tlist, c_ops, e_ops = [], options = Options(store_final_state=True))
        dm = final_dm.final_state
        
        if 'HD' in gate:
            index_HD = np.where(gate == 'HD')[0]
            for k, index in enumerate(index_HD):
                final_dm = virtual_Z_gate(dm, np.pi, TC[index])
                dm = final_dm
            
        
        if 'PZ' in gate:
            index_PZ = np.where(gate == 'PZ')[0]
            for k, index in enumerate(index_PZ):
                final_dm = virtual_Z_gate(dm, angle[index], TC[index])
                dm = final_dm
            
        
        Ini = dm       
        Tsteps.append(tlist)
        states.append(dm)
        
        
    # Tsteps and states has the appended time steps and states. 
    # These has to be merged into a list.  
        
        
    # THIS FUNCTION IS UNDER CONSTRUCTION !!!
    

                       
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
