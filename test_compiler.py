import networkx as nx
from qiskit import QuantumCircuit
from quantum_compiler import compile_circuit

circuit = QuantumCircuit(4)
circuit.h(0)
circuit.cx(0, 1)
circuit.cx(1, 2)
circuit.cx(2, 3)

hardware = nx.grid_2d_graph(2, 2)
hardware = nx.relabel_nodes(hardware, {n: i for i, n in enumerate(hardware.nodes)})

gate_times = {
    '1': 10.0,
    'rz': 0.0,
    'cx': 100.0
}

# T1 times for each hardware qubit (ns)
t1_times = {
    0: 50000,
    1: 75000,
    2: 60000,
    3: 80000
}

# Crosstalk errors between edge pairs
xtalk_errors = {
    tuple(sorted([tuple(sorted([0, 1])), tuple(sorted([2, 3]))])): 0.95,  # 5% error
}

print("Compiling circuit...")
mapping, schedule = compile_circuit(circuit, hardware, gate_times, t1_times, xtalk_errors)

print("\nQubit Mapping:")
for logical, physical in mapping.items():
    print(f"Logical qubit {logical._index} → Physical qubit {physical}")

print("\nSchedule (first 5 gates):")
schedule_items = list(schedule.items())
for gate, time in schedule_items[:5]:
    print(f"Gate {gate} starts at t={time}")

print(f"\nTotal gates scheduled: {len(schedule)}")