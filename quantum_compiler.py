import numpy as np
import networkx as nx
import qiskit
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.circuit.quantumregister import Qubit
from typing import Dict, Any, List, Tuple
import itertools
import dataclasses
import heapq

@dataclasses.dataclass
class GateWrapper:
    # Wrap Gates to make them hashable in graph struct
    gate: Any
    qubits: List[Qubit]
    extra_params: List[Any]
    label: str
    
    def __hash__(self):
        return hash((type(self.gate), tuple(self.qubits), self.label))

    def __str__(self):
        args = ','.join(str(q._index) for q in self.qubits)
        return f'{self.label}{{{self.gate.name}({args})}}'
    
    def __repr__(self):
        return str(self)

def build_program_dependency_graph(circuit):
    # Builds dependency graph where each node is a gate op and each
    # edge is a dependency between operations
    i = 0
    
    qubit_last_use = {}
    g = nx.DiGraph()
    g.add_node(-1)
    
    for instr in circuit:
        hashable_gate = GateWrapper(instr.operation, instr.qubits, instr.clbits, label=i)
        i += 1
        g.add_node(hashable_gate)
        
        for qubit in hashable_gate.qubits:
            if qubit in qubit_last_use:
                g.add_edge(qubit_last_use[qubit], hashable_gate)
            else:
                g.add_edge(-1, hashable_gate)
            qubit_last_use[qubit] = hashable_gate
            
    g.add_node(float('inf'))
    
    for qubit in qubit_last_use:
        g.add_edge(qubit_last_use[qubit], float('inf'))
            
    return g

def interaction_graph_from_circuit(circuit):
    # Build weighted interaction graph for circuit
    q = nx.Graph()

    for instr in circuit:
        for q in instr.qubits:
            g.add_node(q)

        for i in range(len(instr.qubits)):
            for j in range(i):
                q1 = instr.qubits[i]
                q2 = instr.qubits[j]
                if q1 != q2:
                    if (q1, q2) not in g.edges:
                        g.add_edge(q1, q2, weight=1)
                    else:
                        g.edges[q1, q2]['weight'] += 1

    return g

def lookup_xtalk_error(error_dict, edge1, edge2):
    # Lookup crosstalk error from pair of CNOT gates
    edge1 = tuple(sorted(edge1))
    edge2 = tuple(sorted(edge2))
    edges = tuple(sorted([edge1, edge2]))
    return error_dict.get(edges, 1.0)

def map_circuit(quantum_circuit, target_hardware, t1_times=None, xtalk_errors=None):
    # Map circuit qubits to hardware qubits to minimize interaction distance
    ig = interaction_graph_from_circuit(quantum_circuit)
    assert len(ig.nodes) <= len(target_hardware.nodes), "Not enough qubits in hardware"

    mapping = {}
    available_hw_qubits = set(target_hardware.nodes)
    hw_paths = dict(nx.all_pairs_shortest_path_length(target_hardware))

    # Prioritize qubits with longer coherence times
    t1_priority = {}
    if t1_times:
        t1_priority = {hw_q: t1 for hw_q, t1 in t1_times.items()}

    interaction_edges = sorted([(u, v, ig[u][v]['weight']) for u, v in ig.edges], key=lambda x: x[2], reverse=True)

    # Place first pair of most-interacting qubits on adj hardware nodes
    # Prioritize pairs with better T1 times
    if interaction_edges:
        q1, q2, _ = interaction_edges[0]

        best_hw_pair = None
        best_hw_score = -float('inf')

        for hw1 in target_hardware.nodes:
            for hw2 in target_hardware.neighbors(hw1):
                if t1_times:
                    score = t1_times.get(hw1, 0) + t1_times.get(hw2, 0)

                    if xtalk_errors:
                        avg_xtalk = 1.0
                        count = 0

                        for other_hw1, other_hw2 in target_hardware.edges:
                            if (hw1, hw2) != (other_hw1, other_hw2) and (hw1, hw2) != (other_hw2, other_hw1):
                                error = lookup_xtalk_error(xtalk_errors, (hw1, hw2), (other_hw1, other_hw2))
                                avg_xtalk *= error
                                count += 1

                        if count > 0:
                            score *= avg_xtalk

                    if score > best_hw_score:
                        best_hw_score = score
                        best_hw_pair = (hw1, hw2)
                else:
                    best_hw_pair = (hw1, hw2)
                    break

            if best_hw_pair and not t1_times:
                break

        if best_hw_pair:
            hw1, hw2 = best_hw_pair
            mapping[q1] = hw1
            mapping[q2] = hw2
            available_hw_qubits.remove(hw1)
            available_hw_qubits.remove(hw2)
        else:
            hw1 = next(iter(available_hw_qubits))
            available_hw_qubits.remove(hw1)
            hw2 = next(iter(available_hw_qubits))
            available_hw_qubits.remove(hw2)
            mapping[q1] = hw1
            mapping[q2] = hw2

    for q1, q2, weight in interaction_edges:
        if q1 in mapping and q2 in mapping:
            continue
        elif q1 in mapping and q2 not in mapping:
            best_hw = None
            best_score = -float('inf')

            # Find hw node maximizing weighted score by distance and T1
            for hw in available_hw_qubits:
                dist = hw_paths[mapping[q1]][hw]
                score = -dist
                if t1_times:
                    t1_factor = t1_times.get(hw, 0) / max(t1_times.values())
                    score += t1_factor * 2

                if xtalk_errors and (mapping[q1], hw) in target_hardware.edges:
                    avg_xtalk = 1.0
                    count = 0

                    for other_hw1, other_hw2 in target_hardware.edges:
                        if (mapping[q1], hw) != (other_hw1, other_hw2) and (mapping[q1], hw) != (other_hw2, other_hw1):
                            error = lookup_xtalk_error(xtalk_errors, (mapping[q1], hw), (other_hw1, other_hw2))
                            avg_xtalk *= error
                            count += 1

                    if count > 0:
                        # Scale crosstalk factor for meaningful impact
                        score += (avg_xtalk - 0.9) * 10
                if score > best_score:
                    best_score = score
                    best_hw = hw
            
            mapping[q2] = best_hw
            available_hw_qubits.remove(best_hw)
        elif q2 in mapping and q1 not in mapping:
            best_hw = None
            best_score = -float('inf')

            for hw in available_hw_qubits:
                dist = hw_paths[mapping[q2]][hw]
                score = -dist

                if t1_times:
                    t1_factor = t1_times.get(hw, 0) / max(t1_times.values())
                    score += t1_factor * 2

                if xtalk_errors and (mapping[q2], hw) in target_hardware.edges:
                    avg_xtalk = 1.0
                    count = 0

                    for other_hw1, other_hw2 in target_hardware.edges:
                        if (mapping[q2], hw) != (other_hw1, other_hw2) and (mapping[q2], hw) != (other_hw2, other_hw1):
                            error = lookup_xtalk_error(xtalk_errors, (mapping[q2], hw), (other_hw1, other_hw2))
                            avg_xtalk *= error
                            count += 1

                    if count > 0:
                        score += (avg_xtalk - 0.9) * 10

                if score > best_score:
                    best_score = score
                    best_hw = hw
            
            mapping[q1] = best_hw
            available_hw_qubits.remove(best_hw)
        
    unmapped_qubits = [q for q in ig.nodes if q not in mapping]

    if t1_times:
        sorted_hw_qubits = sorted(list(available_hw_qubits), key=lambda hw: t1_times.get(hw, 0), reverse=True)

        for q in unmapped_qubits:
            if sorted_hw_qubits:
                hw = sorted_hw_qubits.pop(0)
                mapping[q]
                available_hw_qubits.remove(hw)
    else:
        for q in unmapped_qubits:
            hw = available_hw_qubits.pop()
            mapping[q] = hw

    return mapping

def route_circuit(quantum_circuit, target_hardware, mapping, t1_times=None, x_talk_errors=None):
    # Route circuit by adding SWAP ops for non-adjacent qubits
    new_circuit = QuantumCircuit(len(target_hardware.nodes))
    current_mapping = mapping.copy()
    
    # Create reverse mapping
    rev_mapping = {v: k for k, v in current_mapping.items()}
    shortest_paths = dict(nx.all_pairs_shortest_path(target_hardware))
    
    for instr in quantum_circuit:
        if len(instr.qubits) == 1:
            q = instr.qubits[0]
            hw_q = current_mapping[q]
            new_circuit.append(instr.operation, [hw_q], instr.clbits)
            
        elif len(instr.qubits) == 2:
            q1, q2 = instr.qubits
            hw_q1, hw_q2 = current_mapping[q1], current_mapping[q2]
            
            # Check if qubits are already adjacent in hardware
            if hw_q2 in target_hardware.neighbors(hw_q1):
                new_circuit.append(instr.operation, [hw_q1, hw_q2], instr.clbits)
            else:
                path = shortest_paths[hw_q1][hw_q2]
                
                if t1_times:
                    all_paths = list(nx.all_simple_paths(target_hardware, hw_q1, hw_q2, cutoff=len(path) + 2))
                    
                    best_path = path
                    best_path_score = -float('inf')
                    
                    for candidate_path in all_paths:
                        path_score = sum(t1_times.get(hw_q, 0) for hw_q in candidate_path)
                        
                        # Penalize longer paths
                        path_score -= len(candidate_path) * 100000  # Large penalty to prioritize shorter paths
                        
                        if path_score > best_path_score:
                            best_path_score = path_score
                            best_path = candidate_path
                    
                    path = best_path
                
                swap_path = path[:-1]
                
                # Apply SWAPs along the path to move q1 next to q2
                for i in range(len(swap_path) - 1):
                    u, v = swap_path[i], swap_path[i+1]
                    new_circuit.swap(u, v)
                    
                    for logical_q, hw_q in list(current_mapping.items()):
                        if hw_q == u:
                            current_mapping[logical_q] = v
                        elif hw_q == v:
                            current_mapping[logical_q] = u
                    
                    rev_mapping = {v: k for k, v in current_mapping.items()}
                
                hw_q1, hw_q2 = current_mapping[q1], current_mapping[q2]
                
                new_circuit.append(instr.operation, [hw_q1, hw_q2], instr.clbits)
                
                # Apply the same SWAPs in reverse to restore mapping
                for i in range(len(swap_path) - 2, -1, -1):
                    u, v = swap_path[i+1], swap_path[i]
                    new_circuit.swap(u, v)
                    
                    for logical_q, hw_q in list(current_mapping.items()):
                        if hw_q == u:
                            current_mapping[logical_q] = v
                        elif hw_q == v:
                            current_mapping[logical_q] = u
                    
                    rev_mapping = {v: k for k, v in current_mapping.items()}
    
    return new_circuit

def schedule_circuit(routed_circuit, gate_times, T1_times, xtalk_errors):
    # Convert SWAP gates to CNOT gates
    
    # Build dependency graph

    # Get gate execution times, build map from qubit to gates using it

    def get_crosstalk_error(gate1, gate2):
        return

    def share_qubits(gate1, gate2):
        return

    def gates_overlap(gate1, gate2, start_times):
        return

    def calculate_t1_cost(gate, start_time, schedule):
        return
    
    def calculate_crosstalk_cost(gate, start_time, schedule):
        return

    # TODO: Implement remaining circuit scheduling logic

def compile_circuit(quantum_circuit, target_hardware, gate_times, t1_times=None, xtalk_errors=None):
    # Complete circuit compilation with mapping, routing, and scheduling
    mapping = map_circuit(quantum_circuit, target_hardware, t1_times, xtalk_errors)
    routed = route_circuit(quantum_circuit, target_hardware, mapping, t1_times, xtalk_errors)
    schedule = schedule_circuit(routed, gate_times, t1_times, xtalk_errors)

    return mapping, schedule
