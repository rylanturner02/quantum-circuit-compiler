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
    pm = qiskit.transpiler.PassManager([
        qiskit.transpiler.passes.BasisTranslator(
            qiskit.circuit.equivalence_library.SessionEquivalenceLibrary, 
            ['rz', 'rx', 'h', 'u3', 'cx']
        )
    ])
    routed_circuit = pm.run(routed_circuit)

    dep_graph = build_program_dependency_graph(routed_circuit)
    
    gates = [node for node in dep_graph.nodes() if node not in [-1, float('inf')]]
    
    gate_execution_times = {}
    for gate in gates:
        if isinstance(gate.gate, qiskit.circuit.library.standard_gates.x.CXGate):
            gate_execution_times[gate] = gate_times['cx']
        elif isinstance(gate.gate, qiskit.circuit.library.standard_gates.rz.RZGate):
            gate_execution_times[gate] = gate_times['rz']
        else:
            gate_execution_times[gate] = gate_times['1']
    
    gate_qubits = {gate: [q._index for q in gate.qubits] for gate in gates}
    
    cnot_gates = [gate for gate in gates if isinstance(gate.gate, qiskit.circuit.library.standard_gates.x.CXGate)]
    
    # Build map from qubit to gates that use it
    qubit_to_gates = {}
    for gate in gates:
        for q in gate_qubits[gate]:
            if q not in qubit_to_gates:
                qubit_to_gates[q] = []
            qubit_to_gates[q].append(gate)
    
    def get_crosstalk_error(gate1, gate2):
        if not (isinstance(gate1.gate, qiskit.circuit.library.standard_gates.x.CXGate) and 
                isinstance(gate2.gate, qiskit.circuit.library.standard_gates.x.CXGate)):
            return 1.0
            
        edge1 = tuple(sorted(gate_qubits[gate1]))
        edge2 = tuple(sorted(gate_qubits[gate2]))
        
        if len(set(edge1).intersection(set(edge2))) > 0:
            return 1.0

        try:
            return lookup_xtalk_error(xtalk_errors, edge1, edge2)
        except:
            return 1.0
    
    def share_qubits(gate1, gate2):
        qubits1 = set(gate_qubits[gate1])
        qubits2 = set(gate_qubits[gate2])
        return len(qubits1.intersection(qubits2)) > 0
    
    def gates_overlap(gate1, gate2, start_times):
        start1 = start_times[gate1]
        end1 = start1 + gate_execution_times[gate1]
        start2 = start_times[gate2]
        end2 = start2 + gate_execution_times[gate2]
        
        return (start1 < end2) and (start2 < end1)
    
    # Compute t1 degradation
    def calculate_t1_cost(gate, start_time, schedule):
        t1_cost = 0
        for q in gate_qubits[gate]:
            q_gates = [g for g in schedule.keys() if q in gate_qubits[g]]
            
            if q_gates:
                # Calculate activity time range of qubit before and after adding new gate
                earliest_op = min([schedule[g] for g in q_gates])
                latest_op = max([schedule[g] + gate_execution_times[g] for g in q_gates])
                current_time = latest_op - earliest_op
                
                potential_earliest = min(earliest_op, start_time)
                potential_latest = max(latest_op, start_time + gate_execution_times[gate])
                potential_time = potential_latest - potential_earliest
                
                t1_cost += (potential_time - current_time) / T1_times.get(q, 1e6)
        
        return t1_cost
    
    # Calculate crosstalk error for potential schedule
    def calculate_crosstalk_cost(gate, start_time, schedule):
        if not isinstance(gate.gate, qiskit.circuit.library.standard_gates.x.CXGate):
            return 0
            
        crosstalk_cost = 0
        gate_end = start_time + gate_execution_times[gate]
        
        for other_gate, other_start in schedule.items():
            if not isinstance(other_gate.gate, qiskit.circuit.library.standard_gates.x.CXGate):
                continue
                
            other_end = other_start + gate_execution_times[other_gate]
            
            if start_time < other_end and gate_end > other_start:
                xt_error = get_crosstalk_error(gate, other_gate)
                crosstalk_cost += (1 - xt_error) 
        
        return crosstalk_cost
    
    # Calculate ASAP schedule
    asap_schedule = {-1: 0}
    for gate in nx.topological_sort(dep_graph):
        if gate == -1:
            continue
        if gate == float('inf'):
            break
            
        pred_finish_times = [asap_schedule[pred] + gate_execution_times.get(pred, 0) for pred in dep_graph.predecessors(gate)]
        
        asap_schedule[gate] = max(pred_finish_times) if pred_finish_times else 0
    
    # Calculate ALAP schedule
    alap_schedule = {float('inf'): float('inf')}
    reversed_dep_graph = nx.reverse(dep_graph)
    
    critical_path_length = 0
    for gate in gates:
        gate_duration = gate_execution_times[gate]
        critical_path_length = max(critical_path_length, asap_schedule[gate] + gate_duration)
    
    for gate in nx.topological_sort(reversed_dep_graph):
        if gate == float('inf'):
            continue
        if gate == -1:
            break
        
        succ_start_times = [alap_schedule[succ] for succ in reversed_dep_graph.predecessors(gate)]
        alap_time = critical_path_length
        
        if succ_start_times:
            alap_time = min(succ_start_times) - gate_execution_times.get(gate, 0)
        
        alap_schedule[gate] = alap_time
    
    # Use list scheduling with priority by T1 times and crosstalk
    ready_gates = []
    scheduled_gates = set()
    schedule = {}
    current_time = 0
    qubit_busy_until = {q: 0 for q in qubit_to_gates.keys()}
    
    for gate in gates:
        preds = list(dep_graph.predecessors(gate))
        if len(preds) == 1 and preds[0] == -1:
            urgency = -alap_schedule.get(gate, 0)
            
            cx_priority = 1000 if isinstance(gate.gate, qiskit.circuit.library.standard_gates.x.CXGate) else 0
            
            t1_priority = -sum(1/T1_times.get(q, 1e6) for q in gate_qubits[gate])
            
            priority = (urgency, cx_priority, t1_priority)
            heapq.heappush(ready_gates, (priority, gate))
    
    while ready_gates or len(scheduled_gates) < len(gates):
        if ready_gates:
            _, gate = heapq.heappop(ready_gates)
            
            gate_qubits_list = gate_qubits[gate]
            earliest_qubit_time = max([qubit_busy_until.get(q, 0) for q in gate_qubits_list])
            
            # Find the earliest time when dependencies are satisfied
            earliest_dep_time = 0
            for pred in dep_graph.predecessors(gate):
                if pred != -1:
                    if pred in schedule:
                        pred_end_time = schedule[pred] + gate_execution_times[pred]
                        earliest_dep_time = max(earliest_dep_time, pred_end_time)
            
            earliest_start = max(earliest_qubit_time, earliest_dep_time)
            
            # For CNOT gates consider crosstalk
            if isinstance(gate.gate, qiskit.circuit.library.standard_gates.x.CXGate):
                best_start_time = earliest_start
                best_cost = float('inf')
                
                for trial_time in range(int(earliest_start), int(earliest_start + 500), 20):
                    t1_cost = calculate_t1_cost(gate, trial_time, schedule)
                    crosstalk_cost = calculate_crosstalk_cost(gate, trial_time, schedule)
                    
                    total_cost = t1_cost + crosstalk_cost * 10
                    
                    # If time is better, update
                    if total_cost < best_cost:
                        best_cost = total_cost
                        best_start_time = trial_time
                        
                        if total_cost < 0.01:
                            break
                
                earliest_start = best_start_time
            
            schedule[gate] = earliest_start
            scheduled_gates.add(gate)
            
            gate_end_time = earliest_start + gate_execution_times[gate]
            for q in gate_qubits_list:
                qubit_busy_until[q] = gate_end_time
            
            current_time = max(current_time, gate_end_time)
            
            # Check if new gates are ready for scheduling
            for succ in dep_graph.successors(gate):
                if succ == float('inf'):  # Skip end node
                    continue
                    
                all_deps_scheduled = True
                for pred in dep_graph.predecessors(succ):
                    if pred != -1 and pred not in scheduled_gates:
                        all_deps_scheduled = False
                        break
                
                if all_deps_scheduled and succ not in scheduled_gates:
                    # Compute priority for gate
                    urgency = -alap_schedule.get(succ, 0)
                    cx_priority = 1000 if isinstance(succ.gate, qiskit.circuit.library.standard_gates.x.CXGate) else 0
                    t1_priority = -sum(1/T1_times.get(q, 1e6) for q in gate_qubits[succ])
                    priority = (urgency, cx_priority, t1_priority)
                    
                    heapq.heappush(ready_gates, (priority, succ))
        else:
            min_gate_time = float('inf')
            for gate in gates:
                if gate not in scheduled_gates:
                    # Check if any of its dependencies are scheduled
                    for pred in dep_graph.predecessors(gate):
                        if pred != -1 and pred in schedule:
                            pred_end_time = schedule[pred] + gate_execution_times[pred]
                            min_gate_time = min(min_gate_time, pred_end_time)
            
            if min_gate_time < float('inf'):
                current_time = min_gate_time
            
            # Check if any gates are ready
            for gate in gates:
                if gate not in scheduled_gates:
                    all_deps_scheduled = True
                    for pred in dep_graph.predecessors(gate):
                        if pred != -1 and pred not in scheduled_gates:
                            all_deps_scheduled = False
                            break
                    
                    if all_deps_scheduled:
                        urgency = -alap_schedule.get(gate, 0)
                        cx_priority = 1000 if isinstance(gate.gate, qiskit.circuit.library.standard_gates.x.CXGate) else 0
                        t1_priority = -sum(1/T1_times.get(q, 1e6) for q in gate_qubits[gate])
                        priority = (urgency, cx_priority, t1_priority)
                        
                        heapq.heappush(ready_gates, (priority, gate))
    
    # Local search to further improve the schedule
    for _ in range(3):
        for gate in cnot_gates:
            current_start = schedule[gate]
            best_start = current_start
            best_cost = float('inf')
            
            # Calculate current cost
            t1_cost = calculate_t1_cost(gate, current_start, {g: t for g, t in schedule.items() if g != gate})
            crosstalk_cost = calculate_crosstalk_cost(gate, current_start, {g: t for g, t in schedule.items() if g != gate})
            current_cost = t1_cost + crosstalk_cost * 10
            
            earliest_dep_time = 0
            for pred in dep_graph.predecessors(gate):
                if pred != -1:
                    pred_end_time = schedule[pred] + gate_execution_times[pred]
                    earliest_dep_time = max(earliest_dep_time, pred_end_time)
            
            latest_start_time = float('inf')
            for succ in dep_graph.successors(gate):
                if succ != float('inf'):
                    if succ in schedule:
                        latest_start_time = min(latest_start_time, schedule[succ])
            
            if latest_start_time == float('inf'):
                latest_start_time = current_start + 1000
            
            max_search = min(latest_start_time - gate_execution_times[gate], current_start + 200)
            
            # Try different time slots within this range
            for trial_time in range(int(earliest_dep_time), int(max_search), 10):
                qubits_available = True
                for q in gate_qubits[gate]:
                    for other_gate in schedule:
                        if other_gate != gate and q in gate_qubits[other_gate]:
                            other_start = schedule[other_gate]
                            other_end = other_start + gate_execution_times[other_gate]
                            
                            if trial_time < other_end and trial_time + gate_execution_times[gate] > other_start:
                                qubits_available = False
                                break
                    
                    if not qubits_available:
                        break
                
                if not qubits_available:
                    continue
                
                # Calculate cost for time slot
                t1_cost = calculate_t1_cost(gate, trial_time, {g: t for g, t in schedule.items() if g != gate})
                crosstalk_cost = calculate_crosstalk_cost(gate, trial_time, {g: t for g, t in schedule.items() if g != gate})
                total_cost = t1_cost + crosstalk_cost * 10
                
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_start = trial_time
            
            if best_cost < current_cost:
                schedule[gate] = best_start
    
    return schedule

def compile_circuit(quantum_circuit, target_hardware, gate_times, t1_times=None, xtalk_errors=None):
    # Complete circuit compilation with mapping, routing, and scheduling
    mapping = map_circuit(quantum_circuit, target_hardware, t1_times, xtalk_errors)
    routed = route_circuit(quantum_circuit, target_hardware, mapping, t1_times, xtalk_errors)
    schedule = schedule_circuit(routed, gate_times, t1_times, xtalk_errors)

    return mapping, schedule
