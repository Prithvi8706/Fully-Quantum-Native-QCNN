"""Bounded, local-only resource and noise evidence for the Q1 study.

This module is intentionally executable with the isolated ``.venv-qiskit``
environment.  The training environment contains the canonical PennyLane model;
the Qiskit environment deliberately does not.  The Qiskit circuits are therefore
never written by hand: the ``export`` subcommand, run in the training
environment, records the exact operation list emitted by
``QCNN/circuits.py::build_circuit`` (plus the PennyLane readouts of every noise
validation input) in ``Results/evidence/q1_canonical_schedules.json``, and the
Qiskit subcommands translate that list gate by gate.  The noise command checks
that the translated circuits reproduce the PennyLane readouts before any noise
is added.  Nothing here contacts a provider, reads a token, or submits a job.

    python -m experiments.q1_local_evidence export
    .venv-qiskit\\Scripts\\python -m experiments.q1_local_evidence resources
    .venv-qiskit\\Scripts\\python -m experiments.q1_local_evidence pooling
    .venv-qiskit\\Scripts\\python -m experiments.q1_local_evidence noise \
        --tasks mnist:3,5 fashion_mnist:0,6 --seeds 0 1 2 --samples 25
    .venv-qiskit\\Scripts\\python -m experiments.q1_local_evidence rehearse

All emitted artifacts state their local target, layout policy, lock hash,
source hash, and limitations.  Local Aer noise is evidence about the declared
noise model; it is not device validation and must not be written as QPU data.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_ROOT = ROOT / "Results" / "evidence"
LOCK_PATH = ROOT / "requirements-qiskit-lock.txt"
CANONICAL_SOURCE = ROOT / "QCNN" / "circuits.py"
SCHEDULE_PATH = EVIDENCE_ROOT / "q1_canonical_schedules.json"
# Image size whose amplitude encoding fills an n-qubit register (Table II).
SCHEDULE_IMAGE_SIZE = {4: 4, 6: 8, 8: 16, 10: 28}
# PennyLane operation name -> Qiskit gate name for the canonical gate set.
PENNYLANE_TO_QISKIT = {"RX": "rx", "RY": "ry", "RZ": "rz", "CNOT": "cx", "CRY": "cry", "CRZ": "crz"}
NOISE_BASIS = ("rz", "sx", "x", "cx")

DEFAULT_BACKEND = "FakePeekskill"
DEFAULT_SEED_TRANSPILER = 42
DEFAULT_OPTIMIZATION_LEVEL = 1
DEFAULT_SHOTS = 128
RESOURCE_QUBITS = (4, 6, 8, 10)
POOLING_ARMS = ("none", "measurement", "unitary", "coherent", "su4")
NOISE_LEVELS = (0.0, 0.005, 0.01, 0.02)
TARGET_BASIS = ("rz", "sx", "x", "ecr", "measure", "barrier", "reset")
DATASET_DIRECTORIES = {
    "mnist": "MNIST",
    "fashion_mnist": "FashionMNIST",
    "kmnist": "KMNIST",
}
IDX_FILENAMES = {
    "mnist": {
        ("train", "images"): "train-images.idx3-ubyte",
        ("train", "labels"): "train-labels.idx1-ubyte",
        ("test", "images"): "t10k-images.idx3-ubyte",
        ("test", "labels"): "t10k-labels.idx1-ubyte",
    },
    "fashion_mnist": {
        ("train", "images"): "train-images-idx3-ubyte.gz",
        ("train", "labels"): "train-labels-idx1-ubyte.gz",
        ("test", "images"): "t10k-images-idx3-ubyte.gz",
        ("test", "labels"): "t10k-labels-idx1-ubyte.gz",
    },
    "kmnist": {
        ("train", "images"): "train-images-idx3-ubyte.gz",
        ("train", "labels"): "train-labels-idx1-ubyte.gz",
        ("test", "images"): "t10k-images-idx3-ubyte.gz",
        ("test", "labels"): "t10k-labels-idx1-ubyte.gz",
    },
}

SCHEMAS = {
    "resources": {"name": "fqcnn_q1_resources", "version": 1},
    "pooling": {"name": "fqcnn_q1_pooling_practicality", "version": 1},
    "noise": {"name": "fqcnn_q1_noise_validation", "version": 1},
    "rehearsal": {"name": "fqcnn_q1_fake_backend_rehearsal", "version": 1},
    "schedules": {"name": "fqcnn_q1_canonical_schedules", "version": 1},
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text_sha256(path: Path) -> str:
    """SHA-256 of a text file with LF line endings, independent of the checkout."""
    return sha256_bytes(Path(path).read_bytes().replace(b"\r\n", b"\n"))


def _json_hash(payload: Any) -> str:
    return sha256_bytes(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    )


def _git_snapshot() -> Dict[str, Any]:
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT), check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--short"], cwd=str(ROOT), check=True,
            capture_output=True, text=True,
        ).stdout.splitlines()
        # Record cleanliness without embedding workstation-specific filenames.
        return {"head": head, "dirty": bool(dirty), "dirty_entries": len(dirty)}
    except (OSError, subprocess.CalledProcessError):
        return {"head": None, "dirty": True, "dirty_entries": None, "error": "git unavailable"}


def _require_qiskit() -> Dict[str, Any]:
    """Import the isolated-toolchain pieces lazily.

    Importing this module remains safe in the Python 3.9 PennyLane training
    environment, where the Qiskit Runtime package may be incompatible.  The
    command then fails closed with an actionable instruction instead of
    silently switching environments.
    """
    try:
        from qiskit import QuantumCircuit, transpile
        from qiskit.quantum_info import Pauli, Statevector
    except Exception as exc:
        raise RuntimeError(
            "local evidence requires the isolated .venv-qiskit environment "
            f"(Qiskit import failed: {type(exc).__name__}: {exc})"
        ) from exc
    return {
        "QuantumCircuit": QuantumCircuit,
        "Statevector": Statevector,
        "Pauli": Pauli,
        "transpile": transpile,
    }


def _require_fake_provider():
    try:
        from qiskit_ibm_runtime import fake_provider
    except Exception as exc:
        raise RuntimeError(
            "local evidence requires qiskit-ibm-runtime in .venv-qiskit "
            f"(fake backend import failed: {type(exc).__name__}: {exc})"
        ) from exc
    return fake_provider


def _select_backend(name: str = DEFAULT_BACKEND):
    """Return the named local BackendV2 fake; never query a provider."""
    fake_provider = _require_fake_provider()
    try:
        backend_type = getattr(fake_provider, name)
    except AttributeError as exc:
        raise RuntimeError(f"local fake backend {name!r} is unavailable") from exc
    try:
        backend = backend_type()
    except Exception as exc:
        raise RuntimeError(f"could not instantiate local fake backend {name!r}") from exc
    if not hasattr(backend, "target") or not hasattr(backend, "num_qubits"):
        raise RuntimeError(f"{name!r} is not a BackendV2-compatible fake backend")
    return backend


def _backend_snapshot(backend) -> Dict[str, Any]:
    operations = sorted(str(name) for name in getattr(backend.target, "operation_names", []))
    coupling = getattr(backend, "coupling_map", None)
    try:
        edges = len(coupling.get_edges()) if coupling is not None else 0
    except Exception:
        edges = None
    return {
        "class": type(backend).__name__,
        "name": str(getattr(backend, "name", type(backend).__name__)),
        "num_qubits": int(backend.num_qubits),
        "backend_v2": True,
        "target_operations": operations,
        "coupling_edges": edges,
        "source": "local qiskit_ibm_runtime.fake_provider; no provider query",
    }


def _metrics(circuit) -> Dict[str, Any]:
    return {
        "depth": int(circuit.depth()),
        "width": int(circuit.width()),
        "size": int(circuit.size()),
        "operation_counts": {
            str(name): int(count) for name, count in sorted(circuit.count_ops().items())
        },
    }


def _target_validation(circuit, backend) -> Dict[str, Any]:
    """Validate final operations against the selected fake target when possible."""
    invalid: List[Dict[str, Any]] = []
    checked = 0
    actual_target = hasattr(backend, "target") and hasattr(
        backend.target, "instruction_supported")
    for item in circuit.data:
        if hasattr(item, "operation"):
            operation = item.operation
            qargs = item.qubits
        else:
            operation, qargs, _ = item
        name = str(operation.name)
        if name == "barrier":
            continue
        checked += 1
        indices = [int(circuit.find_bit(qubit).index) for qubit in qargs]
        if actual_target:
            try:
                supported = bool(
                    backend.target.instruction_supported(operation_name=name, qargs=tuple(indices))
                )
            except Exception:
                supported = False
            if not supported:
                invalid.append({"operation": name, "qargs": indices})
    return {
        "uses_actual_target": bool(actual_target),
        "checked_instructions": int(checked),
        "invalid": invalid,
    }


def _transpile_stage(circuit, backend, seed_transpiler: int, optimization_level: int) -> Dict[str, Any]:
    qiskit = _require_qiskit()
    try:
        transpiled = qiskit["transpile"](
            circuit,
            backend=backend,
            seed_transpiler=int(seed_transpiler),
            optimization_level=int(optimization_level),
        )
    except Exception as exc:
        return {
            "status": "unsupported",
            "error": f"{type(exc).__name__}: {exc}",
            "metrics": None,
            "target_validation": None,
            "unsupported_operations": [],
        }
    target_validation = _target_validation(transpiled, backend)
    counts = _metrics(transpiled)["operation_counts"]
    target_ops = set(str(v) for v in getattr(backend.target, "operation_names", []))
    unsupported = sorted(set(counts) - target_ops - {"barrier"})
    if target_validation["invalid"]:
        unsupported = sorted(set(unsupported) | {
            str(item["operation"]) for item in target_validation["invalid"]
        })
    return {
        "status": "pass" if not unsupported else "unsupported_target_operations",
        "error": None,
        "metrics": _metrics(transpiled),
        "target_validation": target_validation,
        "unsupported_operations": unsupported,
        "circuit": transpiled,
    }


def _decompose(circuit):
    try:
        return circuit.decompose(reps=10)
    except Exception:
        return circuit.decompose()


def _weights_from_path(path: Optional[Path]) -> Tuple[Optional[Dict[str, np.ndarray]], Dict[str, Any]]:
    if path is None or not Path(path).is_file():
        return None, {
            "status": "fixture",
            "path": None if path is None else str(path),
            "sha256": None,
            "claim_scope": "deterministic circuit probe; no trained checkpoint available",
        }
    try:
        with np.load(path, allow_pickle=False) as archive:
            weights = {key: np.asarray(archive[key], dtype=float) for key in archive.files}
    except Exception as exc:
        raise ValueError(f"invalid checkpoint archive {path}: {exc}") from exc
    try:
        relative_path = str(Path(path).resolve().relative_to(ROOT.resolve()))
    except ValueError:
        relative_path = str(Path(path).resolve())
    return weights, {
        "status": "loaded",
        "path": relative_path,
        "sha256": sha256_file(Path(path)),
        "claim_scope": "validation-selected checkpoint when available",
    }


def _validate_checkpoint_identity(
    path: Optional[Path],
    dataset: str,
    classes: Sequence[int],
    seed: int,
    split_id: str,
) -> Path:
    """Require a completed proposed run bound to the validation manifest.

    Noise validation is a scientific release artifact, not a circuit-only probe.
    A missing checkpoint must therefore fail closed instead of silently switching
    to deterministic fixture weights, and a checkpoint from a different split or
    model geometry must not be evaluated against this manifest.
    """
    if path is None or not Path(path).is_file():
        raise ValueError(
            f"noise validation requires a complete proposed checkpoint for "
            f"{dataset} {tuple(classes)} seed {int(seed)}"
        )
    checkpoint = Path(path)
    status_path = checkpoint.parent / "status.json"
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"checkpoint provenance is missing or malformed: {status_path}"
        ) from exc
    if not isinstance(status, dict) or status.get("state") != "complete":
        raise ValueError(f"checkpoint run is not complete: {status_path}")
    if type(status.get("seed")) is not int or status["seed"] != int(seed):
        raise ValueError(f"checkpoint seed does not match requested seed: {status_path}")
    if status.get("split_id") != split_id:
        raise ValueError(f"checkpoint split does not match validation manifest: {status_path}")
    config = status.get("config")
    if not isinstance(config, dict):
        raise ValueError(f"checkpoint has no run configuration: {status_path}")
    expected = {
        "evidence_role": "scientific",
        "image_size": 28,
        "n_qubits": 10,
        "encoding_type": "amplitude",
        "pooling_mode": "unitary",
    }
    mismatches = {
        key: (config.get(key), value)
        for key, value in expected.items()
        if config.get(key) != value
    }
    ablation = config.get("ablation")
    if not isinstance(ablation, dict) or ablation.get("name") != "proposed":
        mismatches["ablation.name"] = (
            ablation.get("name") if isinstance(ablation, dict) else None,
            "proposed",
        )
    if mismatches:
        raise ValueError(f"checkpoint configuration does not match proposed Q1 model: {status_path}")
    return checkpoint


def _find_checkpoint(dataset: str, classes: Sequence[int], seed: int, weights_root: Path) -> Optional[Path]:
    task = f"{dataset}_{int(classes[0])}v{int(classes[1])}"
    candidates = (
        Path(weights_root) / task / "proposed" / f"seed_{int(seed)}" / "weights.npz",
        Path(weights_root) / "runs" / task / "proposed" / f"seed_{int(seed)}" / "weights.npz",
        ROOT / "Results" / "q1_comparison" / "runs" / task / "proposed" /
        f"seed_{int(seed)}" / "weights.npz",
        ROOT / "Results" / "runs" / task / "proposed" / f"seed_{int(seed)}" / "weights.npz",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _canonical_model(n_qubits: int, pooling_mode: str = "unitary", seed: int = 0):
    """Instantiate the canonical PennyLane model (training environment only)."""
    from QCNN.config.Qconfig import QuantumNativeConfig
    from QCNN.models.QCNNModel import PureQuantumNativeCNN

    n_qubits = int(n_qubits)
    if n_qubits not in SCHEDULE_IMAGE_SIZE:
        raise ValueError(f"no canonical image size for n={n_qubits}; choose from {sorted(SCHEDULE_IMAGE_SIZE)}")
    cfg = QuantumNativeConfig.from_image_size(SCHEDULE_IMAGE_SIZE[n_qubits], "amplitude")
    if cfg.n_qubits != n_qubits:
        raise ValueError(f"image size {SCHEDULE_IMAGE_SIZE[n_qubits]} configures {cfg.n_qubits} qubits, not {n_qubits}")
    cfg.pooling_mode = pooling_mode
    cfg.seed = int(seed)
    return PureQuantumNativeCNN(cfg)


def _serialise_operations(x: np.ndarray, model) -> List[Dict[str, Any]]:
    """Record the model-body operations emitted by ``circuits.build_circuit``.

    The amplitude-encoding operation is omitted: the Qiskit side prepares the
    input state itself so that one schedule serves every input.
    """
    import pennylane as qml
    from QCNN import circuits

    tape = qml.tape.make_qscript(circuits.build_circuit)(x, model.quantum_params, model.config)
    operations = list(tape.operations)
    if not operations or type(operations[0]).__name__ != "AmplitudeEmbedding":
        raise ValueError("canonical circuit must begin with amplitude encoding")
    records: List[Dict[str, Any]] = []
    for op in operations[1:]:
        kind = type(op).__name__
        wires = [int(w) for w in op.wires]
        if kind == "MidMeasureMP":
            records.append({"gate": "measure", "wires": wires, "id": str(op.id)})
        elif kind == "Conditional":
            base = op.base
            if base.name not in PENNYLANE_TO_QISKIT:
                raise ValueError(f"unsupported conditional operation {base.name}")
            records.append({
                "gate": PENNYLANE_TO_QISKIT[base.name],
                "wires": [int(w) for w in base.wires],
                "params": [float(p) for p in base.parameters],
                "condition": [str(m.id) for m in op.meas_val.measurements],
            })
        elif kind in PENNYLANE_TO_QISKIT:
            records.append({
                "gate": PENNYLANE_TO_QISKIT[kind],
                "wires": wires,
                "params": [float(p) for p in op.parameters],
            })
        else:
            # General multi-qubit unitaries (the SU(4) pooling arm).
            matrix = np.asarray(qml.matrix(op, wire_order=op.wires), dtype=complex)
            records.append({
                "gate": "unitary",
                "wires": wires,
                "matrix_real": matrix.real.tolist(),
                "matrix_imag": matrix.imag.tolist(),
            })
    return records


def build_canonical_schedules(
    tasks: Sequence[str] = ("mnist:3,5", "fashion_mnist:0,6"),
    seeds: Sequence[int] = (0, 1, 2),
    *,
    samples: int = 25,
    data_root: Path = ROOT / "datasets",
    weights_root: Path = ROOT / "Results" / "q1_comparison" / "runs",
    output: Optional[Path] = SCHEDULE_PATH,
) -> Dict[str, Any]:
    """Export canonical operation lists (training environment, PennyLane)."""
    import pennylane as qml

    resources: Dict[str, Any] = {}
    combos = [(n, "unitary") for n in RESOURCE_QUBITS] + [(4, mode) for mode in POOLING_ARMS if mode != "unitary"]
    for n_qubits, mode in combos:
        model = _canonical_model(n_qubits, mode, seed=0)
        x = np.arange(1, 2 ** n_qubits + 1, dtype=float)
        resources[f"{n_qubits}:{mode}"] = {
            "n_qubits": n_qubits,
            "image_size": SCHEDULE_IMAGE_SIZE[n_qubits],
            "pooling_mode": mode,
            "parameters": "model initialisation with seed 0 (resource counting only)",
            "operations": _serialise_operations(x, model),
        }

    noise_records: List[Dict[str, Any]] = []
    for dataset, classes in (_parse_task(value) for value in tasks):
        for seed in (int(value) for value in seeds):
            images, labels, sample_ids, split_id, _ = _validation_subset(
                dataset, classes, seed, int(samples), Path(data_root), require_persisted=True)
            checkpoint_path = _validate_checkpoint_identity(
                _find_checkpoint(dataset, classes, seed, Path(weights_root)),
                dataset, classes, seed, split_id)
            weights, checkpoint = _weights_from_path(checkpoint_path)
            model = _canonical_model(10, "unitary", seed=seed)
            model.quantum_params = {key: np.asarray(value, dtype=float) for key, value in weights.items()}
            # Same global min-max scaling (pixel range 0..255) and padding as training.
            features = np.asarray(images, dtype=float).reshape(len(images), -1) / 255.0
            padded = model._preprocess_input(features)
            readouts = np.asarray(model.raw_expectations(padded, already_preprocessed=True), dtype=float)
            predictions = np.where(readouts > 0.0, 1, -1)
            noise_records.append({
                "dataset": dataset,
                "classes": list(classes),
                "seed": seed,
                "split_id": split_id,
                "sample_ids": list(sample_ids),
                "checkpoint": checkpoint,
                "operations": _serialise_operations(padded[0], model),
                "pennylane_readouts": [float(value) for value in readouts],
                "pennylane_accuracy": float(np.mean(predictions == np.asarray(labels))),
            })

    payload = {
        "schema": SCHEMAS["schedules"],
        "status": "pass",
        "canonical_source": {
            "path": "QCNN/circuits.py",
            "function": "build_circuit",
            "sha256": _text_sha256(CANONICAL_SOURCE),
        },
        "pennylane": qml.version(),
        "wire_convention": "wire index == Qiskit qubit index; amplitudes are bit-reversed for Qiskit",
        "resources": resources,
        "noise": noise_records,
    }
    payload["payload_sha256"] = _json_hash(payload)
    if output is not None:
        write_artifact(payload, output)
    return payload


def _load_schedules(path: Path = SCHEDULE_PATH) -> Dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is missing; run `python -m experiments.q1_local_evidence export` "
            "in the training environment first")
    schedules = json.loads(path.read_text(encoding="utf-8"))
    if schedules.get("schema") != SCHEMAS["schedules"]:
        raise ValueError(f"{path} is not a canonical schedule artifact")
    if schedules["canonical_source"]["sha256"] != _text_sha256(CANONICAL_SOURCE):
        raise ValueError(f"{path} was exported from a different QCNN/circuits.py; re-run export")
    return schedules


def _build_state_preparation(n_qubits: int, amplitudes: Optional[np.ndarray] = None):
    qiskit = _require_qiskit()
    circuit = qiskit["QuantumCircuit"](int(n_qubits), name="state_preparation")
    if amplitudes is None:
        amplitudes = np.arange(1, 2 ** int(n_qubits) + 1, dtype=float)
        amplitudes /= np.linalg.norm(amplitudes)
    amplitudes = np.asarray(amplitudes, dtype=complex).reshape(-1)
    expected = 2 ** int(n_qubits)
    if amplitudes.size != expected:
        raise ValueError(f"state-preparation vector has {amplitudes.size} values; expected {expected}")
    norm = float(np.linalg.norm(amplitudes))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("state-preparation vector must have a finite, non-zero norm")
    circuit.initialize(amplitudes / norm, list(range(int(n_qubits))))
    return circuit


def _apply_operation(circuit, record: Mapping[str, Any]) -> None:
    gate, wires = record["gate"], [int(w) for w in record["wires"]]
    params = [float(p) for p in record.get("params", [])]
    if gate in ("rx", "ry", "rz"):
        getattr(circuit, gate)(params[0], wires[0])
    elif gate == "cx":
        circuit.cx(wires[0], wires[1])
    elif gate in ("cry", "crz"):
        getattr(circuit, gate)(params[0], wires[0], wires[1])
    elif gate == "unitary":
        matrix = np.asarray(record["matrix_real"]) + 1j * np.asarray(record["matrix_imag"])
        # PennyLane orders the first wire as the most-significant bit; Qiskit
        # treats the first listed qubit as the least-significant one.
        circuit.unitary(matrix, list(reversed(wires)))
    else:
        raise ValueError(f"unsupported schedule gate {gate!r}")


def _build_model_body(
    n_qubits: int,
    pooling_mode: str = "unitary",
    *,
    operations: Optional[Sequence[Mapping[str, Any]]] = None,
):
    """Translate the canonical ``circuits.build_circuit`` operations to Qiskit.

    Without ``operations`` the exported resource schedule for ``(n, mode)`` is
    used.  Mid-circuit measurements get their own classical bits after bit 0,
    which is reserved for the terminal readout; consecutive conditional gates
    on the same measurement share one ``if_test`` block.
    """
    qiskit = _require_qiskit()
    if pooling_mode not in POOLING_ARMS:
        raise ValueError(f"unknown pooling mode {pooling_mode!r}; choose from {POOLING_ARMS}")
    if operations is None:
        key = f"{int(n_qubits)}:{pooling_mode}"
        resources = _load_schedules()["resources"]
        if key not in resources:
            raise ValueError(f"no exported schedule for {key}")
        operations = resources[key]["operations"]
    n_measure = sum(1 for record in operations if record["gate"] == "measure")
    circuit = qiskit["QuantumCircuit"](
        int(n_qubits), max(int(n_qubits), 1 + n_measure), name=f"model_{pooling_mode}")
    clbit_of: Dict[str, int] = {}
    index = 0
    while index < len(operations):
        record = operations[index]
        if record["gate"] == "measure":
            clbit_of[record["id"]] = 1 + len(clbit_of)
            circuit.measure(int(record["wires"][0]), clbit_of[record["id"]])
            index += 1
        elif "condition" in record:
            (condition,) = record["condition"]
            block = []
            while index < len(operations) and operations[index].get("condition") == [condition]:
                block.append(operations[index])
                index += 1
            with circuit.if_test((clbit_of[condition], 1)):
                for item in block:
                    _apply_operation(circuit, item)
        else:
            _apply_operation(circuit, record)
            index += 1
    return circuit


def _build_readout(n_qubits: int):
    qiskit = _require_qiskit()
    circuit = qiskit["QuantumCircuit"](int(n_qubits), int(n_qubits), name="readout")
    circuit.measure(0, 0)
    return circuit


def _build_full_circuit(
    n_qubits: int,
    pooling_mode: str = "unitary",
    *,
    amplitudes: Optional[np.ndarray] = None,
    operations: Optional[Sequence[Mapping[str, Any]]] = None,
    measure: bool = True,
):
    qiskit = _require_qiskit()
    state = _build_state_preparation(n_qubits, amplitudes)
    body = _build_model_body(n_qubits, pooling_mode, operations=operations)
    circuit = qiskit["QuantumCircuit"](int(n_qubits), body.num_clbits, name="fqcnn_local_probe")
    circuit.compose(state, qubits=list(range(int(n_qubits))), inplace=True)
    circuit.compose(
        body,
        qubits=list(range(int(n_qubits))),
        clbits=list(range(body.num_clbits)),
        inplace=True,
    )
    if measure:
        circuit.measure(0, 0)
    return circuit


def _component_record(circuit, backend, seed_transpiler: int, optimization_level: int) -> Dict[str, Any]:
    decomposed = _decompose(circuit)
    transpiled = _transpile_stage(circuit, backend, seed_transpiler, optimization_level)
    result: Dict[str, Any] = {
        "logical": _metrics(circuit),
        "decomposed": _metrics(decomposed),
        "transpiled": transpiled["metrics"],
        "transpilation_status": transpiled["status"],
        "transpilation_error": transpiled["error"],
        "unsupported_operations": transpiled["unsupported_operations"],
        "target_validation": transpiled["target_validation"],
    }
    return result


def _resource_record(
    n_qubits: int,
    pooling_mode: str,
    backend,
    *,
    seed_transpiler: int = DEFAULT_SEED_TRANSPILER,
    optimization_level: int = DEFAULT_OPTIMIZATION_LEVEL,
) -> Dict[str, Any]:
    state = _build_state_preparation(n_qubits)
    body = _build_model_body(n_qubits, pooling_mode)
    readout = _build_readout(n_qubits)
    full = _build_full_circuit(n_qubits, pooling_mode)
    components = {
        "state_preparation": _component_record(state, backend, seed_transpiler, optimization_level),
        "model_body": _component_record(body, backend, seed_transpiler, optimization_level),
        "readout": _component_record(readout, backend, seed_transpiler, optimization_level),
        "total": _component_record(full, backend, seed_transpiler, optimization_level),
    }
    statuses = [value["transpilation_status"] for value in components.values()]
    payload = {
        "qubits": int(n_qubits),
        "pooling_mode": pooling_mode,
        "backend": _backend_snapshot(backend),
        "transpilation": {
            "seed_transpiler": int(seed_transpiler),
            "optimization_level": int(optimization_level),
            "layout_policy": "transpiler-selected layout against the declared fake BackendV2 target; no initial_layout",
        },
        "components": components,
        "status": "pass" if all(status == "pass" for status in statuses) else "partial",
        "limitations": (
            "Counts are local logical/decomposed/transpiled estimates. They are not a hardware run; "
            "measurement-style dynamic control may be unsupported by the fake target."
        ),
    }
    payload["payload_sha256"] = _json_hash(payload)
    return payload


def build_resources(
    qubits: Sequence[int] = RESOURCE_QUBITS,
    *,
    backend_name: str = DEFAULT_BACKEND,
    seed_transpiler: int = DEFAULT_SEED_TRANSPILER,
    optimization_level: int = DEFAULT_OPTIMIZATION_LEVEL,
    output: Optional[Path] = None,
) -> Dict[str, Any]:
    requested = tuple(int(value) for value in qubits)
    if not requested or any(value < 2 for value in requested):
        raise ValueError("qubits must be a non-empty sequence of values >= 2")
    if len(set(requested)) != len(requested):
        raise ValueError("qubits must be unique")
    backend = _select_backend(backend_name)
    if max(requested) > int(backend.num_qubits):
        raise ValueError(f"requested n={max(requested)} exceeds fake backend width {backend.num_qubits}")
    records = []
    for n_qubits in requested:
        records.append(_resource_record(
            n_qubits, "unitary", backend,
            seed_transpiler=seed_transpiler,
            optimization_level=optimization_level,
        ))
    payload = {
        "schema": SCHEMAS["resources"],
        "status": "pass" if all(item["status"] == "pass" for item in records) else "partial",
        "protocol": {
            "qubits": list(requested),
            "arm": "frozen unitary pooling",
            "components": ["state_preparation", "model_body", "readout", "total"],
            "backend": _backend_snapshot(backend),
            "seed_transpiler": int(seed_transpiler),
            "optimization_level": int(optimization_level),
            "layout_policy": "transpiler-selected layout; no initial_layout supplied",
        },
        "canonical_source": {
            "path": "QCNN/circuits.py",
            "function": "build_circuit",
            "sha256": _text_sha256(CANONICAL_SOURCE),
        },
        "records": records,
        "environment": {
            "python": sys.version.split()[0],
            # Keep the artifact portable and avoid leaking the workstation path.
            "executable": Path(sys.executable).name,
            "qiskit_lock": {
                "path": "requirements-qiskit-lock.txt",
                "sha256": _text_sha256(LOCK_PATH) if LOCK_PATH.exists() else None,
            },
            "git": _git_snapshot(),
            "network_accessed": False,
            "credentials_accessed": False,
        },
        "limitations": (
            "This artifact reports local transpilation/resource estimates only. "
            "State preparation and model-body counts are intentionally separated; "
            "no QPU execution is implied."
        ),
    }
    payload["payload_sha256"] = _json_hash(payload)
    if output is not None:
        write_artifact(payload, output)
    return payload


def build_pooling_practicality(
    n_qubits: int = 4,
    *,
    backend_name: str = DEFAULT_BACKEND,
    seed_transpiler: int = DEFAULT_SEED_TRANSPILER,
    optimization_level: int = DEFAULT_OPTIMIZATION_LEVEL,
    output: Optional[Path] = None,
) -> Dict[str, Any]:
    n_qubits = int(n_qubits)
    backend = _select_backend(backend_name)
    if n_qubits > int(backend.num_qubits):
        raise ValueError(f"requested n={n_qubits} exceeds fake backend width {backend.num_qubits}")
    arms: Dict[str, Any] = {}
    for mode in POOLING_ARMS:
        record = _resource_record(
            n_qubits, mode, backend,
            seed_transpiler=seed_transpiler,
            optimization_level=optimization_level,
        )
        arms[f"pool_{mode}"] = {
            "mode": mode,
            "logical": record["components"]["model_body"]["logical"],
            "decomposed": record["components"]["model_body"]["decomposed"],
            "transpiled": record["components"]["model_body"]["transpiled"],
            "transpilation_status": record["components"]["model_body"]["transpilation_status"],
            "unsupported_operations": record["components"]["model_body"]["unsupported_operations"],
            "target_validation": record["components"]["model_body"]["target_validation"],
            "limitations": record["limitations"],
        }
    measurement = arms["pool_measurement"]
    dynamic_unsupported = bool(
        measurement["unsupported_operations"] or
        measurement["transpilation_status"] != "pass"
    )
    payload = {
        "schema": SCHEMAS["pooling"],
        "status": "pass",
        "protocol": {
            "n_qubits": n_qubits,
            "arms": list(arms),
            "backend": _backend_snapshot(backend),
            "seed_transpiler": int(seed_transpiler),
            "optimization_level": int(optimization_level),
            "same_backend_settings": True,
        },
        "arms": arms,
        "dynamic_behavior": {
            "measurement_arm_target_unsupported": dynamic_unsupported,
            "measurement_arm_statement": (
                "The measurement-style arm is reported as dynamic control and is not treated as a "
                "fully unitary hardware path; unsupported target operations remain visible."
            ),
        },
        "canonical_source": {
            "path": "QCNN/circuits.py",
            "function": "build_circuit",
            "sha256": _text_sha256(CANONICAL_SOURCE),
        },
        "environment": {
            "qiskit_lock": {
                "path": "requirements-qiskit-lock.txt",
                "sha256": _text_sha256(LOCK_PATH) if LOCK_PATH.exists() else None,
            },
            "git": _git_snapshot(),
            "network_accessed": False,
            "credentials_accessed": False,
        },
        "limitations": (
            "Fake-backend transpilation is a local practicality comparison, not device execution. "
            "SU(4) is an expressivity/resource ceiling arm; it is not promoted to the headline model."
        ),
    }
    payload["payload_sha256"] = _json_hash(payload)
    if output is not None:
        write_artifact(payload, output)
    return payload


def _parse_task(value: str) -> Tuple[str, Tuple[int, int]]:
    try:
        dataset, pair = str(value).split(":", 1)
        low, high = pair.split(",", 1)
        classes = (int(low), int(high))
    except (ValueError, TypeError) as exc:
        raise ValueError(f"task must have the form dataset:low,high, got {value!r}") from exc
    if dataset not in DATASET_DIRECTORIES:
        raise ValueError(f"unknown dataset {dataset!r}; choose from {sorted(DATASET_DIRECTORIES)}")
    if classes[0] == classes[1] or min(classes) < 0 or max(classes) > 9:
        raise ValueError(f"task classes must be two distinct labels in 0..9, got {classes!r}")
    return dataset, classes


def _idx_open(path: Path):
    return gzip.open(path, "rb") if path.suffix == ".gz" else path.open("rb")


def _load_idx_train(dataset: str, data_root: Path = ROOT / "datasets") -> Tuple[np.ndarray, np.ndarray, List[str]]:
    if dataset not in IDX_FILENAMES:
        raise ValueError(f"unknown dataset {dataset!r}")
    directory = Path(data_root) / DATASET_DIRECTORIES[dataset]
    image_path = directory / IDX_FILENAMES[dataset][("train", "images")]
    label_path = directory / IDX_FILENAMES[dataset][("train", "labels")]
    if not image_path.is_file() or not label_path.is_file():
        raise FileNotFoundError(
            f"missing local source files for {dataset}: {image_path}, {label_path}; "
            "run the checksummed dataset registry fetch first"
        )
    with _idx_open(label_path) as handle:
        header = handle.read(8)
        magic, count = struct.unpack(">II", header)
        if magic != 2049:
            raise ValueError(f"invalid IDX label magic {magic} in {label_path}")
        labels = np.frombuffer(handle.read(), dtype=np.uint8)
    with _idx_open(image_path) as handle:
        header = handle.read(16)
        magic, image_count, rows, columns = struct.unpack(">IIII", header)
        if magic != 2051 or (rows, columns) != (28, 28):
            raise ValueError(f"invalid IDX image header in {image_path}")
        images = np.frombuffer(handle.read(), dtype=np.uint8)
    if count != image_count or labels.size != count or images.size != count * 28 * 28:
        raise ValueError(f"IDX source count mismatch for {dataset}")
    images = images.reshape(count, 28 * 28)
    sample_ids = [f"{dataset}:train:{index:05d}" for index in range(count)]
    return images, labels, sample_ids


def _canonical_validation_manifest(
    dataset: str, classes: Sequence[int], seed: int
) -> Optional[Tuple[Path, Dict[str, Any]]]:
    """Return a persisted Q1 manifest when one already exists.

    The local Qiskit environment intentionally does not import the training
    split service.  Reading its JSON manifest is sufficient to preserve exact
    sample identity, and prevents this adapter from silently inventing a new
    validation split when a canonical comparison split is available.
    """
    low, high = sorted(int(value) for value in classes)
    task = f"{dataset}_{low}v{high}"
    roots = (
        ROOT / "Results" / "q1_comparison" / "manifests",
        ROOT / "Results" / "smoke" / "q1_datasets" / "manifests",
    )
    candidates: List[Path] = []
    for root in roots:
        if root.exists():
            candidates.extend(sorted(root.glob(f"{task}_n*_seed{int(seed)}.json")))
    for candidate in candidates:
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
            if (
                payload.get("class_mapping") == {str(low): -1, str(high): 1}
                and isinstance(payload.get("sample_ids"), list)
                and isinstance(payload.get("val_idx"), list)
                and isinstance(payload.get("id"), str)
            ):
                return candidate, payload
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
    return None


def _validation_subset(
    dataset: str,
    classes: Sequence[int],
    seed: int,
    samples: int,
    data_root: Path,
    *,
    require_persisted: bool = False,
) -> Tuple[np.ndarray, np.ndarray, List[str], str, Dict[str, Any]]:
    images, labels, sample_ids = _load_idx_train(dataset, data_root)
    classes = tuple(sorted(int(value) for value in classes))
    persisted = _canonical_validation_manifest(dataset, classes, seed)
    if persisted is None and require_persisted:
        raise ValueError(
            f"noise validation requires a canonical Q1 manifest for "
            f"{dataset} {classes} seed {int(seed)}"
        )
    if persisted is not None:
        manifest_path, manifest = persisted
        val_positions = [int(value) for value in manifest["val_idx"]]
        if samples > len(val_positions):
            raise ValueError(
                f"requested {samples} validation examples; only {len(val_positions)} available"
            )
        selected_positions = val_positions[:int(samples)]
        selected_ids = [str(value) for value in manifest["sample_ids"]]
        source_indices: List[int] = []
        for value in selected_ids:
            try:
                source_indices.append(int(value.rsplit(":", 1)[1]))
            except (IndexError, ValueError) as exc:
                raise ValueError(f"manifest contains non-source sample ID {value!r}") from exc
        selected = np.asarray([source_indices[position] for position in selected_positions], dtype=int)
        mapped_labels = np.where(labels[selected] == classes[1], 1, -1).astype(np.int8)
        return (
            images[selected],
            mapped_labels,
            [selected_ids[position] for position in selected_positions],
            str(manifest["id"]),
            {
                "method": "persisted canonical Q1 manifest validation partition",
                "manifest_path": manifest_path.resolve().relative_to(ROOT.resolve()).as_posix(),
                "fractions": manifest.get("fractions", [0.60, 0.15, 0.25]),
                "source_positions": [int(position) for position in selected_positions],
            },
        )
    keep = np.flatnonzero(np.isin(labels, np.asarray(classes, dtype=np.uint8)))
    if keep.size < 4:
        raise ValueError(f"not enough source samples for {dataset} {classes}")
    # The project split is deterministic and stratified.  This local selector
    # mirrors its fractions while remaining independent of sklearn in the
    # isolated Qiskit environment.  It records its own hash so it cannot be
    # confused with a historical split.
    rng = np.random.default_rng(int(seed))
    val_indices: List[int] = []
    train_indices: List[int] = []
    test_indices: List[int] = []
    for label in classes:
        group = keep[labels[keep] == label].copy()
        group = group[rng.permutation(len(group))]
        n_train = int(round(0.60 * len(group)))
        n_val = int(round(0.15 * len(group)))
        train_indices.extend(int(v) for v in group[:n_train])
        val_indices.extend(int(v) for v in group[n_train:n_train + n_val])
        test_indices.extend(int(v) for v in group[n_train + n_val:])
    val_indices = sorted(val_indices)
    if samples > len(val_indices):
        raise ValueError(f"requested {samples} validation examples; only {len(val_indices)} available")
    # Selection is seeded, but sorted source order makes the persisted IDs easy
    # to audit and keeps repeated runs byte-stable.
    selected = np.asarray(val_indices[:int(samples)], dtype=int)
    mapped_labels = np.where(labels[selected] == classes[1], 1, -1).astype(np.int8)
    selected_ids = [sample_ids[int(index)] for index in selected]
    split_payload = {
        "dataset": dataset,
        "classes": list(classes),
        "seed": int(seed),
        "fractions": [0.60, 0.15, 0.25],
        "train_source_indices": sorted(train_indices),
        "val_source_indices": val_indices,
        "test_source_indices": sorted(test_indices),
    }
    split_id = _json_hash(split_payload)
    return images[selected], mapped_labels, selected_ids, split_id, {
        "method": "deterministic per-class 60/15/25 source split; validation selected before noise sweep",
        "fractions": [0.60, 0.15, 0.25],
        "source_indices": [int(index) for index in selected],
    }


def _amplitudes_from_pixels(images: np.ndarray, n_qubits: int = 10) -> np.ndarray:
    target = 2 ** int(n_qubits)
    features = np.asarray(images, dtype=float) / 255.0
    if features.ndim == 1:
        features = features.reshape(1, -1)
    if features.shape[1] > target:
        features = features[:, :target]
    if features.shape[1] < target:
        features = np.pad(features, ((0, 0), (0, target - features.shape[1])))
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.where(norms <= 1e-15, 1.0, norms)
    features = features / norms
    # PennyLane's AmplitudeEmbedding treats the supplied vector as ordered by
    # wires [0, ..., n-1], whereas Qiskit stores basis amplitudes with qubit 0
    # as the least-significant bit.  Reverse the bit indices before calling
    # QuantumCircuit.initialize so the Qiskit reconstruction is genuinely the
    # canonical wire-order circuit rather than merely a self-consistent probe.
    n = int(n_qubits)
    order = [int(f"{index:0{n}b}"[::-1], 2) for index in range(target)]
    return features[:, order]


def _exact_statevector_expectation(circuit) -> float:
    qiskit = _require_qiskit()
    state = qiskit["Statevector"].from_instruction(circuit)
    probabilities = np.asarray(state.probabilities(qargs=[0]), dtype=float)
    return float(probabilities[0] - probabilities[1])


def _aer_statevector_expectation(circuit) -> float:
    try:
        from qiskit_aer import AerSimulator
    except Exception as exc:
        raise RuntimeError("qiskit-aer is required for the zero-noise agreement check") from exc
    qiskit = _require_qiskit()
    probe = circuit.copy()
    probe.save_expectation_value(qiskit["Pauli"]("Z"), [0], label="z_readout")
    simulator = AerSimulator(method="statevector")
    result = simulator.run(probe).result()
    value = result.data(0)["z_readout"]
    return float(np.real(value))


def _noise_model(kind: str, level: float):
    try:
        from qiskit_aer.noise import NoiseModel, depolarizing_error, phase_damping_error
    except Exception as exc:
        raise RuntimeError("qiskit-aer noise helpers are required for local noise evidence") from exc
    level = float(level)
    if not 0.0 <= level <= 1.0:
        raise ValueError("noise level must be between 0 and 1")
    model = NoiseModel()
    if level == 0.0:
        return model
    if kind == "depolarizing":
        one = depolarizing_error(level, 1)
        two = depolarizing_error(min(1.0, 2.0 * level), 2)
    elif kind == "dephasing":
        one = phase_damping_error(level)
        two = phase_damping_error(min(1.0, 2.0 * level)).tensor(phase_damping_error(min(1.0, 2.0 * level)))
    else:
        raise ValueError("noise model must be 'depolarizing' or 'dephasing'")
    for gate in NOISE_BASIS[:-1]:
        model.add_all_qubit_quantum_error(one, gate)
    model.add_all_qubit_quantum_error(two, "cx")
    return model


def _shot_only_expected_accuracy(readouts: Sequence[float], labels: Sequence[int], shots: int) -> float:
    """Exact expected accuracy from finite-shot sampling alone (no gate noise).

    With readout r, each shot returns 1 with probability (1 - r) / 2; the
    estimate 1 - 2k/shots is classified +1 only when it is strictly positive,
    matching ``_noisy_accuracy``.
    """
    shots = int(shots)
    expected = []
    for readout, label in zip(readouts, labels):
        p_one = min(max((1.0 - float(readout)) / 2.0, 0.0), 1.0)
        p_positive = sum(
            math.comb(shots, k) * p_one ** k * (1.0 - p_one) ** (shots - k)
            for k in range(shots) if 2 * k < shots
        )
        expected.append(p_positive if int(label) == 1 else 1.0 - p_positive)
    return float(np.mean(expected))


def _noisy_accuracy(
    circuits: Sequence[Any],
    labels: Sequence[int],
    *,
    kind: str,
    level: float,
    shots: int,
) -> Dict[str, Any]:
    try:
        from qiskit_aer import AerSimulator
    except Exception as exc:
        raise RuntimeError("qiskit-aer is required for local noise evidence") from exc
    qiskit = _require_qiskit()
    model = _noise_model(kind, level)
    measured = []
    for circuit in circuits:
        candidate = circuit.copy()
        if not any(
            str(item.operation.name if hasattr(item, "operation") else item[0].name) == "measure"
            for item in candidate.data
        ):
            candidate.measure(0, 0)
        measured.append(candidate)
    simulator = AerSimulator(noise_model=model, method="automatic")
    # Compile every model gate to the basis the noise model is attached to, so
    # RX/RY/CRY/CRZ are noisy too; state preparation stays an ideal initialize.
    compiled = qiskit["transpile"](
        measured, basis_gates=list(NOISE_BASIS) + ["initialize", "measure"],
        optimization_level=0, seed_transpiler=42)
    result = simulator.run(compiled, shots=int(shots), seed_simulator=991).result()
    raw = []
    for index in range(len(compiled)):
        counts = result.get_counts(index)
        # The one-bit readout is classical bit 0; strings may be padded when a
        # circuit carries the model's unused classical bits.
        total = max(1, sum(counts.values()))
        ones = sum(value for key, value in counts.items() if str(key).replace(" ", "")[-1:] == "1")
        raw.append(float(1.0 - 2.0 * ones / total))
    labels = np.asarray(labels, dtype=int)
    predictions = np.where(np.asarray(raw) > 0.0, 1, -1)
    correct = predictions == labels
    accuracy = float(correct.mean()) if len(correct) else None
    stderr = math.sqrt(max(accuracy * (1.0 - accuracy), 0.0) / len(correct)) if correct.size else None
    return {
        "noise_model": kind,
        "level": float(level),
        "shots": int(shots),
        "accuracy": accuracy,
        "accuracy_standard_error": float(stderr) if stderr is not None else None,
        "mean_readout": float(np.mean(raw)) if raw else None,
        "raw_readouts": [float(value) for value in raw],
        "predictions": [int(value) for value in predictions],
    }


def build_noise_validation(
    tasks: Sequence[str],
    seeds: Sequence[int],
    *,
    samples: int = 25,
    shots: int = DEFAULT_SHOTS,
    levels: Sequence[float] = NOISE_LEVELS,
    data_root: Path = ROOT / "datasets",
    weights_root: Path = ROOT / "Results" / "q1_comparison" / "runs",
    output: Optional[Path] = None,
) -> Dict[str, Any]:
    parsed_tasks = [_parse_task(value) for value in tasks]
    seeds = tuple(int(value) for value in seeds)
    if not parsed_tasks or not seeds:
        raise ValueError("noise validation requires at least one task and seed")
    if samples < 1 or shots < 1:
        raise ValueError("samples and shots must be positive")
    levels = tuple(float(value) for value in levels)
    if not levels or levels[0] != 0.0:
        raise ValueError("noise ladder must begin with declared zero-noise level 0.0")
    schedules = {
        (item["dataset"], tuple(item["classes"]), int(item["seed"])): item
        for item in _load_schedules()["noise"]
    }
    records: List[Dict[str, Any]] = []
    for dataset, classes in parsed_tasks:
        for seed in seeds:
            images, labels, sample_ids, split_id, split_metadata = _validation_subset(
                dataset, classes, seed, int(samples), Path(data_root),
                require_persisted=True,
            )
            checkpoint_path = _find_checkpoint(dataset, classes, seed, Path(weights_root))
            checkpoint_path = _validate_checkpoint_identity(
                checkpoint_path, dataset, classes, seed, split_id
            )
            _, checkpoint = _weights_from_path(checkpoint_path)
            schedule = schedules.get((dataset, tuple(classes), int(seed)))
            if schedule is None:
                raise ValueError(f"no exported schedule for {dataset} {tuple(classes)} seed {seed}")
            if (schedule["sample_ids"] != list(sample_ids) or schedule["split_id"] != split_id
                    or schedule["checkpoint"]["sha256"] != checkpoint["sha256"]):
                raise ValueError(f"exported schedule does not match {dataset} {tuple(classes)} seed {seed}")
            operations = schedule["operations"]
            amplitudes = _amplitudes_from_pixels(images, 10)
            exact_values: List[float] = []
            aer_values: List[float] = []
            noisy_circuits = []
            for vector in amplitudes:
                base = _build_full_circuit(
                    10, "unitary", amplitudes=vector, operations=operations, measure=False)
                exact_values.append(_exact_statevector_expectation(base))
                aer_values.append(_aer_statevector_expectation(base))
                noisy_circuits.append(_build_full_circuit(
                    10, "unitary", amplitudes=vector, operations=operations, measure=True))
            exact_values_arr = np.asarray(exact_values, dtype=float)
            aer_values_arr = np.asarray(aer_values, dtype=float)
            pennylane_arr = np.asarray(schedule["pennylane_readouts"], dtype=float)
            agreement = {
                "method": (
                    "Qiskit Statevector.from_instruction of the translated canonical circuit versus "
                    "the PennyLane readouts of the trained model, and versus AerSimulator statevector"
                ),
                "max_abs_difference_vs_pennylane": float(np.max(np.abs(exact_values_arr - pennylane_arr))),
                "max_abs_difference": float(np.max(np.abs(exact_values_arr - aer_values_arr))),
                "mean_abs_difference": float(np.mean(np.abs(exact_values_arr - aer_values_arr))),
                "tolerance": 1e-8,
                "passes": bool(
                    np.max(np.abs(exact_values_arr - pennylane_arr)) <= 1e-8
                    and np.max(np.abs(exact_values_arr - aer_values_arr)) <= 1e-8
                ),
                "pennylane_accuracy": float(schedule["pennylane_accuracy"]),
                "shot_only_expected_accuracy": _shot_only_expected_accuracy(
                    exact_values_arr, labels, int(shots)),
                "pennylane_readouts": [float(value) for value in pennylane_arr],
                "statevector_readouts": [float(value) for value in exact_values_arr],
                "aer_readouts": [float(value) for value in aer_values_arr],
            }
            ladder = []
            for kind in ("depolarizing", "dephasing"):
                for level in levels:
                    if level == 0.0:
                        predictions = np.where(exact_values_arr > 0.0, 1, -1)
                        correct = predictions == labels
                        result = {
                            "noise_model": kind,
                            "level": 0.0,
                            "shots": None,
                            "accuracy": float(correct.mean()),
                            "accuracy_standard_error": float(math.sqrt(max(float(correct.mean()) * (1.0 - float(correct.mean())), 0.0) / len(correct))),
                            "mean_readout": float(np.mean(exact_values_arr)),
                            "raw_readouts": [float(value) for value in exact_values_arr],
                            "predictions": [int(value) for value in predictions],
                            "evaluation": "exact statevector; shared zero-noise reference",
                        }
                    else:
                        result = _noisy_accuracy(
                            noisy_circuits, labels, kind=kind, level=level, shots=int(shots)
                        )
                        result["evaluation"] = "local Aer noise model; not hardware data"
                    ladder.append(result)
            records.append({
                "dataset": dataset,
                "classes": list(classes),
                "seed": int(seed),
                "split_id": split_id,
                "sample_ids": sample_ids,
                "n_validation": len(sample_ids),
                "split": split_metadata,
                "checkpoint": checkpoint,
                "zero_noise_agreement": agreement,
                "noise_ladder": ladder,
            })
    payload = {
        "schema": SCHEMAS["noise"],
        "status": "pass" if all(item["zero_noise_agreement"]["passes"] for item in records) else "partial",
        "protocol": {
            "tasks": [f"{dataset}:{classes[0]},{classes[1]}" for dataset, classes in parsed_tasks],
            "seeds": list(seeds),
            "samples_per_validation_subset": int(samples),
            "shots": int(shots),
            "noise_models": ["depolarizing", "dephasing"],
            "levels": list(levels),
            "retraining": False,
            "test_data_used": False,
        },
        "records": records,
        "environment": {
            "python": sys.version.split()[0],
            "executable": Path(sys.executable).name,
            "qiskit_lock": {
                "path": "requirements-qiskit-lock.txt",
                "sha256": _text_sha256(LOCK_PATH) if LOCK_PATH.exists() else None,
            },
            "git": _git_snapshot(),
            "network_accessed": False,
            "credentials_accessed": False,
        },
        "limitations": (
            "Noise values are local Aer simulations of declared depolarizing/dephasing channels. "
            "They are not calibration-accurate device noise, fake-backend execution, or QPU data. "
            "Records using a deterministic fixture rather than a loaded checkpoint are explicitly labelled."
        ),
    }
    payload["payload_sha256"] = _json_hash(payload)
    if output is not None:
        write_artifact(payload, output)
    return payload


def build_rehearsal(
    n_qubits: int = 6,
    *,
    backend_name: str = DEFAULT_BACKEND,
    shots: int = 1024,
    seed_transpiler: int = DEFAULT_SEED_TRANSPILER,
    optimization_level: int = DEFAULT_OPTIMIZATION_LEVEL,
    output: Optional[Path] = None,
) -> Dict[str, Any]:
    """Prepare/hash a local payload for later QPU review, without submission."""
    backend = _select_backend(backend_name)
    circuit = _build_full_circuit(int(n_qubits), "unitary", measure=True)
    transpiled = _transpile_stage(circuit, backend, seed_transpiler, optimization_level)
    if transpiled["metrics"] is None:
        raise RuntimeError(f"fake-backend rehearsal transpilation failed: {transpiled['error']}")
    payload_spec = {
        "backend": _backend_snapshot(backend),
        "n_qubits": int(n_qubits),
        "shots": int(shots),
        "seed_transpiler": int(seed_transpiler),
        "optimization_level": int(optimization_level),
        "layout_policy": "transpiler-selected layout; no initial_layout supplied",
        "sample_ids": [f"rehearsal:synthetic:{index:05d}" for index in range(1)],
        "circuit": {
            "logical": _metrics(circuit),
            "transpiled": transpiled["metrics"],
            "target_validation": transpiled["target_validation"],
            "unsupported_operations": transpiled["unsupported_operations"],
        },
        "canonical_source": {
            "path": "QCNN/circuits.py",
            "function": "build_circuit",
            "sha256": _text_sha256(CANONICAL_SOURCE),
        },
        "qiskit_lock": {
            "path": "requirements-qiskit-lock.txt",
            "sha256": _text_sha256(LOCK_PATH) if LOCK_PATH.exists() else None,
        },
    }
    payload = {
        "schema": SCHEMAS["rehearsal"],
        "status": "pass" if not transpiled["unsupported_operations"] else "partial",
        "submission": {
            "authenticated": False,
            "submitted": False,
            "network_accessed": False,
            "credentials_accessed": False,
            "provider": None,
            "statement": "Payload was prepared and hashed locally; no service was contacted.",
        },
        "payload": payload_spec,
        "payload_sha256": _json_hash(payload_spec),
        "environment": {"python": sys.version.split()[0], "executable": Path(sys.executable).name, "git": _git_snapshot()},
        "limitations": (
            "This is a fake-backend rehearsal only. It does not reserve a backend, estimate a live queue, "
            "authenticate, submit, or establish hardware performance."
        ),
    }
    if output is not None:
        write_artifact(payload, output)
    return payload


def write_artifact(payload: Mapping[str, Any], output: Path) -> None:
    output = Path(output)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export", help="export canonical PennyLane operation lists (training env)")
    export.add_argument("--tasks", nargs="+", default=["mnist:3,5", "fashion_mnist:0,6"])
    export.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    export.add_argument("--samples", type=int, default=25)
    export.add_argument("--data-root", default=str(ROOT / "datasets"))
    export.add_argument("--weights-root", default=str(ROOT / "Results" / "q1_comparison" / "runs"))
    export.add_argument("--output", default=str(SCHEDULE_PATH.relative_to(ROOT)))

    resources = sub.add_parser("resources", help="count logical/decomposed/transpiled resource stages")
    resources.add_argument("--qubits", nargs="+", type=int, default=list(RESOURCE_QUBITS))
    resources.add_argument("--backend", default=DEFAULT_BACKEND)
    resources.add_argument("--seed-transpiler", type=int, default=DEFAULT_SEED_TRANSPILER)
    resources.add_argument("--optimization-level", type=int, default=DEFAULT_OPTIMIZATION_LEVEL)
    resources.add_argument("--output", default="Results/evidence/q1_resources.json")

    pooling = sub.add_parser("pooling", help="compare pooling arms on one local fake backend")
    pooling.add_argument("--qubits", type=int, default=4)
    pooling.add_argument("--backend", default=DEFAULT_BACKEND)
    pooling.add_argument("--seed-transpiler", type=int, default=DEFAULT_SEED_TRANSPILER)
    pooling.add_argument("--optimization-level", type=int, default=DEFAULT_OPTIMIZATION_LEVEL)
    pooling.add_argument("--output", default="Results/evidence/q1_pooling_practicality.json")

    noise = sub.add_parser("noise", help="run bounded local Aer validation/noise ladders")
    noise.add_argument("--tasks", nargs="+", default=["mnist:3,5", "fashion_mnist:0,6"])
    noise.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    noise.add_argument("--samples", type=int, default=25)
    noise.add_argument("--shots", type=int, default=DEFAULT_SHOTS)
    noise.add_argument("--levels", nargs="+", type=float, default=list(NOISE_LEVELS))
    noise.add_argument("--data-root", default=str(ROOT / "datasets"))
    noise.add_argument("--weights-root", default=str(ROOT / "Results" / "q1_comparison" / "runs"))
    noise.add_argument("--output", default="Results/evidence/q1_noise_validation.json")

    rehearsal = sub.add_parser("rehearse", help="prepare/hash one fake-backend hardware payload")
    rehearsal.add_argument("--qubits", type=int, default=6)
    rehearsal.add_argument("--shots", type=int, default=1024)
    rehearsal.add_argument("--backend", default=DEFAULT_BACKEND)
    rehearsal.add_argument("--seed-transpiler", type=int, default=DEFAULT_SEED_TRANSPILER)
    rehearsal.add_argument("--optimization-level", type=int, default=DEFAULT_OPTIMIZATION_LEVEL)
    rehearsal.add_argument("--output", default="Results/evidence/q1_fake_backend_rehearsal.json")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "export":
        payload = build_canonical_schedules(
            args.tasks,
            args.seeds,
            samples=args.samples,
            data_root=Path(args.data_root),
            weights_root=Path(args.weights_root),
            output=Path(args.output),
        )
    elif args.command == "resources":
        payload = build_resources(
            args.qubits,
            backend_name=args.backend,
            seed_transpiler=args.seed_transpiler,
            optimization_level=args.optimization_level,
            output=Path(args.output),
        )
    elif args.command == "pooling":
        payload = build_pooling_practicality(
            args.qubits,
            backend_name=args.backend,
            seed_transpiler=args.seed_transpiler,
            optimization_level=args.optimization_level,
            output=Path(args.output),
        )
    elif args.command == "noise":
        payload = build_noise_validation(
            args.tasks,
            args.seeds,
            samples=args.samples,
            shots=args.shots,
            levels=args.levels,
            data_root=Path(args.data_root),
            weights_root=Path(args.weights_root),
            output=Path(args.output),
        )
    else:
        payload = build_rehearsal(
            args.qubits,
            backend_name=args.backend,
            shots=args.shots,
            seed_transpiler=args.seed_transpiler,
            optimization_level=args.optimization_level,
            output=Path(args.output),
        )
    print(json.dumps({"status": payload["status"], "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
