"""Create local-only evidence for the isolated Python 3.11 Qiskit toolchain."""
from __future__ import annotations

import ast
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import math
import platform
import subprocess
import sys
from pathlib import Path

from qiskit import QuantumCircuit, transpile
from qiskit.providers import BackendV2
from qiskit_ibm_runtime import fake_provider


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "Results" / "evidence"
LOCK = ROOT / "requirements-qiskit-lock.txt"
SOURCE = ROOT / "QCNN" / "circuits.py"
ALLOWED = {"rz", "sx", "x", "ecr", "measure", "barrier", "reset"}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _candidate_fake_backends() -> list[BackendV2]:
    candidates = []
    for name in dir(fake_provider):
        candidate = getattr(fake_provider, name)
        if not inspect.isclass(candidate) or candidate is BackendV2:
            continue
        try:
            if not issubclass(candidate, BackendV2):
                continue
            backend = candidate()
        except Exception:
            continue
        operations = set(backend.target.operation_names)
        if backend.num_qubits >= 6 and {"rz", "sx", "x", "ecr", "measure"} <= operations:
            candidates.append((backend.num_qubits, name, backend))
    return [backend for _, _, backend in sorted(candidates, key=lambda item: (item[0], item[1]))]


def _target_validation(circuit: QuantumCircuit, backend: BackendV2) -> dict:
    invalid = []
    checked = 0
    for instruction in circuit.data:
        name = instruction.operation.name
        if name == "barrier":
            continue
        checked += 1
        qargs = tuple(circuit.find_bit(qubit).index for qubit in instruction.qubits)
        if not backend.target.instruction_supported(operation_name=name, qargs=qargs):
            invalid.append({"operation": name, "qargs": list(qargs)})
    return {"uses_actual_target": True, "checked_instructions": checked, "invalid": invalid}


def _select_fake_backend(circuit: QuantumCircuit) -> tuple[BackendV2, QuantumCircuit, dict]:
    failures = []
    for backend in _candidate_fake_backends():
        try:
            transpiled = transpile(
                circuit, backend=backend, seed_transpiler=42, optimization_level=1
            )
        except Exception as exc:
            failures.append(f"{type(backend).__name__}: {type(exc).__name__}")
            continue
        validation = _target_validation(transpiled, backend)
        if not validation["invalid"]:
            return backend, transpiled, validation
        failures.append(f"{type(backend).__name__}: invalid target qargs")
    raise RuntimeError(
        "no local fake BackendV2 accepted the deterministic fixture against its actual target: "
        + ", ".join(failures)
    )


def _import_statuses() -> dict:
    statuses = {}
    for module_name in ("qiskit", "qiskit_aer", "qiskit_ibm_runtime"):
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            statuses[module_name] = {
                "status": "fail",
                "exception": f"{type(exc).__name__}: {exc}",
            }
        else:
            statuses[module_name] = {"status": "pass", "exception": None}
    return statuses


def _training_environment() -> dict:
    probe = subprocess.run(
        [
            "py",
            "-3.9",
            "-c",
            "import json,platform,sys; print(json.dumps({'executable': sys.executable, 'version': platform.python_version()}))",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if probe.returncode == 0:
        observed = json.loads(probe.stdout)
        return {
            "required_python_version": "3.9.13",
            "lock_path": "requirements-lock.txt",
            "observed": True,
            "observed_version": observed["version"],
            "executable": observed["executable"],
            "probe_exception": None,
        }
    return {
        "required_python_version": "3.9.13",
        "lock_path": "requirements-lock.txt",
        "observed": False,
        "observed_version": None,
        "executable": None,
        "probe_exception": probe.stderr.strip() or f"py launcher exited {probe.returncode}",
    }


def _canonical_signature() -> tuple[str, str]:
    source_text = SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source_text)
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "build_circuit"
    )
    signature = {
        "name": function.name,
        "arguments": [argument.arg for argument in function.args.args],
        "defaults": [ast.unparse(default) for default in function.args.defaults],
    }
    function_text = ast.get_source_segment(source_text, function)
    return _sha256(json.dumps(signature, sort_keys=True).encode()), _sha256(function_text.encode())


def _fixture() -> tuple[QuantumCircuit, dict]:
    n_qubits = 4
    parameters = {
        "input_amplitudes": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "classifier": [round((index + 1) * math.pi / 64.0, 12) for index in range(32)],
    }
    circuit = QuantumCircuit(n_qubits, 1, name="canonical_build_circuit_classifier_fixture")
    circuit.initialize(parameters["input_amplitudes"], range(n_qubits))
    classifier = parameters["classifier"]
    active = list(range(n_qubits))
    for index, qubit in enumerate(active):
        circuit.rx(classifier[index * 2 % 32], qubit)
        circuit.ry(classifier[(index * 2 + 1) % 32], qubit)
        circuit.rz(classifier[(index * 2 + 8) % 32], qubit)
    for index in range(n_qubits - 1):
        circuit.cx(active[index], active[index + 1])
    circuit.cx(active[-1], active[0])
    for index, qubit in enumerate(active):
        circuit.rx(classifier[(index * 2 + 16) % 32], qubit)
        circuit.ry(classifier[(index * 2 + 17) % 32], qubit)
    circuit.cx(active[0], active[1])
    circuit.rz(classifier[31], active[0])
    circuit.measure(active[0], 0)
    return circuit, parameters


def _metrics(circuit: QuantumCircuit) -> dict:
    return {
        "depth": circuit.depth(),
        "width": circuit.width(),
        "size": circuit.size(),
        "operation_counts": dict(sorted(circuit.count_ops().items())),
    }


def main() -> None:
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError("Task 6 smoke must run under Python 3.11")
    if ".venv-qiskit" not in str(Path(sys.executable).resolve()):
        raise RuntimeError("Task 6 smoke must run from .venv-qiskit")

    import_statuses = _import_statuses()
    failed_imports = [name for name, result in import_statuses.items() if result["status"] != "pass"]
    if failed_imports:
        raise RuntimeError(f"required imports failed: {failed_imports}; {import_statuses}")

    circuit, parameters = _fixture()
    backend, transpiled, target_validation = _select_fake_backend(circuit)
    target_operations = sorted(backend.target.operation_names)
    decomposed = circuit.decompose(reps=10)
    target_edges = list(backend.coupling_map.get_edges()) if backend.coupling_map else []
    transpiled_counts = dict(sorted(transpiled.count_ops().items()))
    unsupported = sorted(set(transpiled_counts) - ALLOWED)
    if unsupported:
        raise RuntimeError(f"unsupported transpiled operations: {unsupported}")

    lock_hash = _sha256(LOCK.read_bytes())
    signature_hash, function_hash = _canonical_signature()
    dirty_paths = _git("status", "--short").splitlines()
    backend_data = {
        "class": type(backend).__name__,
        "name": backend.name,
        "num_qubits": backend.num_qubits,
        "backend_v2": isinstance(backend, BackendV2),
        "target_operations": target_operations,
        "coupling_edges": len(target_edges),
    }
    environment = {
        "schema_version": 1,
        "status": "pass",
        "hardware_environment": {
            "python_version": platform.python_version(),
            "executable": str(Path(sys.executable).resolve()),
            "environment": ".venv-qiskit",
        },
        "training_environment": _training_environment(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "platform": platform.platform(),
        },
        "installed_versions": {
            package: importlib.metadata.version(package)
            for package in ("qiskit", "qiskit-aer", "qiskit-ibm-runtime")
        },
        "imports": import_statuses,
        "lock": {"path": "requirements-qiskit-lock.txt", "sha256": lock_hash},
        "git": {
            "head": _git("rev-parse", "HEAD"),
            "dirty": bool(dirty_paths),
            "dirty_paths": dirty_paths,
            "provenance": "generated from the current checkout; dirty_paths records uncommitted Task 6 changes",
        },
        "fake_backend": {"available": True, **backend_data},
        "network_accessed": False,
        "credentials_accessed": False,
    }

    payload = {
        "fixture": parameters,
        "qasm": circuit.qasm() if hasattr(circuit, "qasm") else str(circuit),
        "transpiled": str(transpiled),
    }
    layout = getattr(transpiled, "layout", None)
    transpilation = {
        "schema_version": 1,
        "status": "pass",
        "canonical_source": {
            "path": "QCNN/circuits.py",
            "function": "build_circuit",
            "sha256": _sha256(SOURCE.read_bytes()),
            "signature_sha256": signature_hash,
            "function_source_sha256": function_hash,
        },
        "fixture": {
            "identity": "deterministic n=4 classifier-tail structural fixture",
            "n_qubits": 4,
            "parameters": parameters,
            "source_mapping": "Gate order and parameter indexing mirror build_circuit's classifier tail.",
        },
        "readout": {
            "wire": 0,
            "classical_bit": 0,
            "measurement_semantics": "single Z-basis measurement of canonical readout wire 0",
        },
        "backend": backend_data,
        "transpilation": {"seed_transpiler": 42, "optimization_level": 1},
        "layout": {
            "policy": "transpiler-selected layout against the actual directed fake BackendV2 target; no initial_layout supplied",
            "initial": str(getattr(layout, "initial_layout", None)),
            "final": str(getattr(layout, "final_layout", None)),
        },
        "operations": {
            "target_basis": target_operations,
            "transpiled_counts": transpiled_counts,
            "unsupported": unsupported,
            "target_validation": target_validation,
        },
        "metrics": {
            "logical": _metrics(circuit),
            "decomposed": _metrics(decomposed),
            "transpiled": _metrics(transpiled),
        },
        "payload_sha256": _sha256(json.dumps(payload, sort_keys=True).encode()),
        "equivalence": {
            "result": "limited",
            "method": "mechanical source/signature hashing plus classifier-tail gate-order mapping",
            "limitations": "The isolated environment intentionally excludes PennyLane and pennylane-qiskit, so this smoke does not convert or prove full build_circuit unitary equivalence. It does validate every final non-barrier instruction and physical qarg against the selected fake BackendV2 target.",
        },
        "network_accessed": False,
        "credentials_accessed": False,
    }

    EVIDENCE.mkdir(parents=True, exist_ok=True)
    (EVIDENCE / "qiskit_environment.json").write_text(
        json.dumps(environment, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (EVIDENCE / "qiskit_transpile_smoke.json").write_text(
        json.dumps(transpilation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"PASS Python {platform.python_version()}, qiskit {environment['installed_versions']['qiskit']}, "
        f"backend {type(backend).__name__}, lock {lock_hash}"
    )


if __name__ == "__main__":
    main()
