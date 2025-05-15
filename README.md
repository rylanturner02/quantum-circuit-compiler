# quantum-circuit-compiler
A comprehensive quantum circuit compiler with noise mitigation.

## Setup and run testing
- Create a virtual environment: `python3 -m venv qc-env` followed by `source qc-env/bin/activate` (Mac) or `qc-env/bin/activate` (Windows)
- Install Python packages: `pip install qiskit networkx numpy matplotlib`
- In root directory of this project (`quantum-circuit-compiler`), run the test compiler: `python3 test_compiler.py`

## To-do
- Add visualization to test script (hardware topology, Gantt chart, original and compiled circuits).
- Add support for loading QASM files, exporting to QASM files.
- Implement [pulse-level scheduling](https://arxiv.org/abs/2206.05144) over gate-level for improved accuracy.
