from qulacs.gate import PauliRotation
from qulacs import ParametricQuantumCircuit
import scipy.optimize




import matplotlib.pyplot as plt
import numpy as np
import time 
import random
import scipy.linalg

from qulacs import QuantumState
from qulacs import QuantumCircuit
from qulacs.gate import DenseMatrix
from qulacs.circuit import QuantumCircuitOptimizer

from qulacs import QuantumState
from qulacs.gate import Identity, X,Y,Z #パウリ演算子
from qulacs.gate import H,S,Sdag, sqrtX,sqrtXdag,sqrtY,sqrtYdag #1量子ビット Clifford演算
from qulacs.gate import T,Tdag #1量子ビット 非Clifford演算
from qulacs.gate import RX,RY,RZ #パウリ演算子についての回転演算
from qulacs.gate import CNOT, CZ, SWAP #2量子ビット演算
from qulacs.gate import U1,U2,U3 #IBM Gate
from qulacs import Observable
import re


def cost_func_ansatz(ansatz_circuit,hamiltonian,para):
    nqubits = ansatz_circuit.get_qubit_count()
    state = QuantumState(nqubits)

    parameter_count = ansatz_circuit.get_parameter_count()



    for i in range(parameter_count):
        ansatz_circuit.set_parameter(i,para[i])
        
    ansatz_circuit.update_quantum_state(state)

    return  hamiltonian.get_expectation_value(state)

def cost_func_state_ansatz(state,ansatz_circuit,hamiltonian,para):
    nqubits = ansatz_circuit.get_qubit_count()

    parameter_count = ansatz_circuit.get_parameter_count()



    for i in range(parameter_count):
        ansatz_circuit.set_parameter(i,para[i])
        
    ansatz_circuit.update_quantum_state(state)

    return  hamiltonian.get_expectation_value(state)




def add_parametric_gate_from_observable(hamiltonian,circuit):

    nqubits = hamiltonian.get_qubit_count()
    
    for j in range(hamiltonian.get_term_count()):
        pauli = hamiltonian.get_term(j)

        # Get the subscript of each pauli symbol
        index_list = pauli.get_index_list()

        # Get pauli symbols (I,X,Y,Z -> 0,1,2,3)
        pauli_id_list = pauli.get_pauli_id_list()

        circuit.add_parametric_multi_Pauli_rotation_gate(index_list, pauli_id_list, 0.0)
        
 #   print(ansatz_circuit.get_parameter_count())
#    print(ansatz_circuit)
    return circuit

def hamiltonian_drivers_ansatz(hamiltonian,driver,max_depth):
    i=0

    nqubits = hamiltonian.get_qubit_count()
    ansatz_circuit =ParametricQuantumCircuit(nqubits)

    num_drivers = len(driver)

    for depth in range(max_depth):
        for j in range(hamiltonian.get_term_count()):
            pauli = hamiltonian.get_term(j)

            # Get the subscript of each pauli symbol
            index_list = pauli.get_index_list()

            # Get pauli symbols (I,X,Y,Z -> 0,1,2,3)
            pauli_id_list = pauli.get_pauli_id_list()

            ansatz_circuit.add_parametric_multi_Pauli_rotation_gate(index_list, pauli_id_list, 0.0)
            i+=1
        for k in range(num_drivers):

            for j in range(driver[k].get_term_count()):
                pauli = driver[k].get_term(j)

                # Get the subscript of each pauli symbol
                index_list = pauli.get_index_list()
                # Get pauli symbols (I,X,Y,Z -> 0,1,2,3)
                pauli_id_list = pauli.get_pauli_id_list()

                ansatz_circuit.add_parametric_multi_Pauli_rotation_gate(index_list, pauli_id_list, 0.0)
                i+=1    
 #   print(ansatz_circuit.get_parameter_count())
#    print(ansatz_circuit)
    return ansatz_circuit


def hamiltonian_ansatz(hamiltonian,driver,max_depth):
    i=0

    nqubits = hamiltonian.get_qubit_count()
    ansatz_circuit =ParametricQuantumCircuit(nqubits)

    

    for depth in range(max_depth):
        for j in range(hamiltonian.get_term_count()):
            pauli = hamiltonian.get_term(j)

            # Get the subscript of each pauli symbol
            index_list = pauli.get_index_list()

            # Get pauli symbols (I,X,Y,Z -> 0,1,2,3)
            pauli_id_list = pauli.get_pauli_id_list()

            ansatz_circuit.add_parametric_multi_Pauli_rotation_gate(index_list, pauli_id_list, 0.0)
            i+=1

        for j in range(driver.get_term_count()):
            pauli = driver.get_term(j)

            # Get the subscript of each pauli symbol
            index_list = pauli.get_index_list()
            # Get pauli symbols (I,X,Y,Z -> 0,1,2,3)
            pauli_id_list = pauli.get_pauli_id_list()

            ansatz_circuit.add_parametric_multi_Pauli_rotation_gate(index_list, pauli_id_list, 0.0)
            i+=1    
 #   print(ansatz_circuit.get_parameter_count())
#    print(ansatz_circuit)
    return ansatz_circuit

def show_distribution(state):
    nqubits = state.get_qubit_count()
    plt.bar([i for i in range(pow(2,nqubits))],pow(abs(state.get_vector()),2))
    plt.show()

def qasm_to_qulacs_fromfile(input_filepath,qulacs_circuit):

    with open(input_filepath, "r") as ifile:
        lines = ifile.readlines()
        

        for line in lines:
            s = re.search(r"qreg|cx|u3|u1", line)

            if s is None:
                continue

            elif s.group() == 'qreg':
                match = re.search(r"\d\d*", line)
                print(match)
                continue

            elif s.group() == 'cx':
                match = re.findall(r"\[\d\d*\]", line)  # int抽出
                c_qbit = int(match[0].strip('[]'))
                t_qbit = int(match[1].strip('[]'))
                qulacs_circuit.add_gate(CNOT(c_qbit,t_qbit))   

                continue

            elif s.group() == 'u3':
                m_r = re.findall(r"[-]?\d\.\d\d*", line)  # real抽出, 負符号考慮
                m_i = re.findall(r"\[\d\d*\]", line)  # int抽出

                # target_bit = m_i[0]
                # u3parameters = m_r
                qulacs_circuit.add_gate(U3(int(m_i[0].strip('[]')),float(m_r[0]),float(m_r[1]),float(m_r[2])))

                continue

            elif s.group() == 'u1':
                m_r = re.findall(r"[-]?\d\.\d\d*", line)  # real抽出
                m_i = re.findall(r"\[\d\d*\]", line)  # int抽出

                qulacs_circuit.add_gate(U1(int(m_i[0].strip('[]')),float(m_r[0])))

                continue



def define_X_field(operator):
    nqubits = operator.get_qubit_count()
    for k in range(nqubits):
        operator.add_operator(1.0,"X {0}".format(k)) 
    return operator


def define_Z_field(operator):
    nqubits = operator.get_qubit_count()
    for k in range(nqubits):
        operator.add_operator(1.0,"Z {0}".format(k)) 
    return operator


def define_Heisenberg_Hamiltonian(operator,ListOfInt):
    nqubits = operator.get_qubit_count()
    for k in range(len(ListOfInt)):
        operator.add_operator(1.0,"Z {0}".format(ListOfInt[k][0])+"Z {0}".format(ListOfInt[k][1]))
        operator.add_operator(1.0,"X {0}".format(ListOfInt[k][0])+"X {0}".format(ListOfInt[k][1])) 
        operator.add_operator(1.0,"Y {0}".format(ListOfInt[k][0])+"Y {0}".format(ListOfInt[k][1])) 
    return operator



def define_Ising_Hamiltonian(operator,ListOfInt):
    nqubits = operator.get_qubit_count()
    
    for k in range(len(ListOfInt)):
        operator.add_operator(ListOfInt[k][2],"Z {0}".format(ListOfInt[k][0])+"Z {0}".format(ListOfInt[k][1]))
    return operator

def parametrix_dense_exp_gate(index_list,dense_matrix,para):
    dense_exp_gate = DenseMatrix(index_list,scipy.linalg.expm(-1.0j*para*np.array(dense_matrix)))
    return dense_exp_gate


def show_observable(hamiltonian):
    for j in range(hamiltonian.get_term_count()):
        pauli=hamiltonian.get_term(j)

        # Get the subscript of each pauli symbol
        index_list = pauli.get_index_list()

        # Get pauli symbols (I,X,Y,Z -> 0,1,2,3)
        pauli_id_list = pauli.get_pauli_id_list()

        # Get pauli coefficient
        coef = pauli.get_coef()

        # Create a copy of pauli operator
        another_pauli = pauli.copy()

        s = ["I","X","Y","Z"]
        pauli_str = [s[i] for i in pauli_id_list]
        terms_str = [item[0]+str(item[1]) for item in zip(pauli_str,index_list)]
        full_str = str(coef) + " " + " ".join(terms_str)
        print(full_str)

