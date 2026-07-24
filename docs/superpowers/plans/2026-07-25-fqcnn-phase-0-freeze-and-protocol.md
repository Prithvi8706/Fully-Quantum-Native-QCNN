# FQCNN Phase 0 — Freeze Guards and Protocol Integrity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the frozen FQCNN model's identity, its leakage-free evaluation protocol, its single circuit source, and its reproducibility inputs all testable, so that every later phase of the Q1 upgrade builds on evidence that cannot silently drift.

**Architecture:** A new `QCNN/freeze.py` pins the protected function `(x, θ) ↦ ⟨Z_readout⟩` through a marker-vector circuit signature, 20 committed expectation fixtures, a unitarity audit, and a gradient-based effective-parameter audit. A new `QCNN/utils/splits.py` centralises seeded stratified 60/15/25 splitting behind persisted manifests, and `QCNN/utils/run_artifacts.py` gives every run an isolated output directory plus a guard that permits exactly one test evaluation. `QCNN/circuits.py` then becomes the one topology consumed by the model, the noise simulator, and later hardware/resource tooling — with the expectation fixtures as the proof the consolidation preserved behaviour.

**Tech Stack:** Python 3.9.13, PennyLane 0.38.0, PennyLane-Lightning 0.38.0, NumPy 1.26.4, scikit-learn 1.6.1, matplotlib 3.9.4, pytest (to be installed), Git Bash / PowerShell on Windows 11.

## Global Constraints

These bind every task. They are copied from `UPGRADE_PLAN.md` §A and the roadmap's global constraints.

- **A1 — Architecture freeze.** The headline FQCNN, exactly as executed by `QCNNModel._pure_quantum_forward` today, must not change. No gate added, removed, reordered, or reparameterised. Where paper and code disagree, the paper moves.
- **A2 — Freeze = observational equivalence.** The protected object is `(x, θ) ↦ ⟨Z_readout⟩`. Expectation regression tolerance is **1e-10**. The only permitted circuit edit is removal of a gate proven inert, which requires a written proof in the commit message and a passing regression at **1e-12**. Phase 0 does not exercise that exception.
- **A3 — Unitarity invariant.** No mid-circuit measurement, reset, classical feed-forward, or non-unitary channel in the main pipeline. The 10-qubit state stays pure from state preparation to the single terminal measurement.
- **A4 — Variants are arms, never replacements.** No task here promotes a variant to the headline.
- **A5 — No `[ARCH — opt-in]` work.** Phase 0 contains none and must not introduce any.
- **A6 — Honesty invariant.** Never report a number produced under test-set leakage; never call a zero-gradient slot "trainable" without disclosure.
- **Headline identity:** `QuantumNativeConfig.from_image_size(28, 'amplitude')` with `seed = 42` → 10 qubits, 1024 features, 4 conv layers, `lightning.qubit`, `shots = None`, **269 allocated parameter slots**, archived weights at `Results/Weights/quantum_model_params.npz`.
- **Split protocol:** stratified, seeded **60/15/25** train/validation/test. Validation alone drives checkpointing, LR plateau, early stopping, and tuning. Test is evaluated exactly once per run, after the model is fixed.
- **Environment:** do not upgrade PennyLane, NumPy, or scikit-learn. Installing `pytest` is the only permitted dependency addition in this phase.
- **No retraining in Phase 0.** The clean-protocol retrain is Phase 1 work and must not be started here.

---

## File Structure

**New files**

| Path | Responsibility |
|---|---|
| `pytest.ini` | Test discovery + markers, rooted at the repo. |
| `QCNN/freeze.py` | Headline identity constants, model/params loading, circuit signature, unitarity audit, effective-parameter audit. |
| `QCNN/circuits.py` | Canonical frozen circuit builder + evaluation hooks. |
| `QCNN/utils/seeding.py` | One seed policy for numpy, random, PennyLane. |
| `QCNN/utils/splits.py` | Seeded stratified 60/15/25 split manifests: create, save, load, verify, apply. |
| `QCNN/utils/run_artifacts.py` | Per-run isolated output directories, status lifecycle, per-example predictions, single-test-evaluation guard. |
| `scripts/regen_freeze_fixtures.py` | Regenerates every committed freeze fixture from the archived model. |
| `tests/conftest.py` | Shared headline fixtures. |
| `tests/fixtures/headline_signature.json` | Committed circuit-topology reference. |
| `tests/fixtures/headline_expectations.npz` | Committed 20-input `⟨Z⟩` reference. |
| `tests/fixtures/effective_params.json` | Committed effective-parameter audit. |
| `tests/test_headline_identity.py` | The archived headline model is reconstructible. |
| `tests/test_freeze_architecture.py` | Signature + unitarity guards. |
| `tests/test_freeze_expectations.py` | 20-input expectation regression at 1e-10. |
| `tests/test_effective_params.py` | Gradient audit matches the committed artifact. |
| `tests/test_splits.py` | Determinism, stratification, disjointness. |
| `tests/test_run_artifacts.py` | Isolation, status lifecycle, single-test-evaluation guard. |
| `tests/test_protocol.py` | No test-set leakage in the trainer or runners. |
| `tests/test_circuit_source.py` | Shared builder equals the frozen oracle; no duplicate topology. |
| `docs/paper_code_reconciliation.md` | Equation-by-equation paper↔code dispositions (M0.7). |

**Modified files**

| Path | Change |
|---|---|
| `.gitignore` | Stop ignoring `Results/metadata.json` (it defines headline identity). |
| `QCNN/models/QCNNModel.py` | Delegate `_pure_quantum_forward` to `QCNN/circuits.py`; delete the unused MSE duplicate `quantum_loss_function`. |
| `QCNN/training/Qtrainer.py` | Train/validation only; no test data reaches model selection; weights write to a caller-supplied path. |
| `main.py` | Consume split manifests; exactly one guarded test evaluation; use `QCNN/utils/seeding.py`. |
| `experiments/run_experiments.py` | Consume split manifests; pass validation to the trainer; one guarded test evaluation; use shared seeding. |
| `noise_sim.py` | Delete the hand-copied topology; call `QCNN/circuits.py` with noise hooks; use shared seeding. |
| `requirements.txt`, `requirements-lock.txt` | Truthful pinned environment (F9). |
| `setup_env.ps1`, `setup_env.bat` | Match the lock. |
| `reproduce.sh` | Run the freeze/protocol gate first; fix the interpreter default. |

**Deleted files** (`UPGRADE_PLAN.md` 0.6 permits exactly these non-circuit dead artifacts)

- `Results/Weights/quantum_model_params(old).npz`
- `output.txt`
- `QCNNModel.quantum_loss_function` (method, not a file)

---

## Task 1: Test harness and headline identity

Establishes that the archived headline model can be reconstructed from committed artifacts, and gives every later task a place to put tests.

**Files:**
- Create: `pytest.ini`
- Create: `QCNN/freeze.py`
- Create: `tests/conftest.py`
- Create: `tests/test_headline_identity.py`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `QCNN.freeze.HEADLINE_IMAGE_SIZE: int = 28`, `HEADLINE_ENCODING: str = 'amplitude'`, `HEADLINE_SEED: int = 42`, `HEADLINE_N_QUBITS: int = 10`, `HEADLINE_N_PARAM_SLOTS: int = 269`, `HEADLINE_WEIGHTS: str`, `FIXTURE_DIR: str`, `N_REGRESSION_INPUTS: int = 20`, `REGRESSION_INPUT_SEED: int = 20260725`, `REGRESSION_TOL: float = 1e-10`
  - `QCNN.freeze.build_headline_model() -> PureQuantumNativeCNN`
  - `QCNN.freeze.load_archived_params(model) -> pnp.ndarray` (shape `(269,)`)
  - `QCNN.freeze.fixed_regression_inputs() -> np.ndarray` (shape `(20, 1024)`)
  - `QCNN.freeze.slot_ranges(model) -> dict[str, list[int]]` (name → `[start, stop)`)
  - pytest fixtures `headline_model`, `archived_params`, `regression_inputs`

- [ ] **Step 1: Install pytest into the pinned environment**

`pytest` is a test-only dependency and does not touch PennyLane, NumPy, or scikit-learn versions.

Run: `python -m pip install "pytest==7.4.4"`
Expected: `Successfully installed ... pytest-7.4.4` (or "already satisfied").

Then confirm the scientific stack is unchanged:

Run: `python -c "import pennylane, numpy, sklearn; print(pennylane.__version__, numpy.__version__, sklearn.__version__)"`
Expected: `0.38.0 1.26.4 1.6.1`

- [ ] **Step 2: Write `pytest.ini`**

```ini
[pytest]
testpaths = tests
python_files = test_*.py
markers =
    slow: circuit-simulation tests that take more than ~10 seconds
addopts = -ra
```

- [ ] **Step 3: Write the failing identity test**

Create `tests/test_headline_identity.py`:

```python
"""The archived headline model must be reconstructible from committed artifacts.

Everything downstream in the Q1 upgrade is defined relative to this exact model
(UPGRADE_PLAN.md M0.1), so its identity is asserted before anything else.
"""
import json
import os

import numpy as np
import pytest

from QCNN import freeze


def test_headline_config_matches_archived_metadata():
    with open(os.path.join("Results", "metadata.json")) as fh:
        archived = json.load(fh)["config"]

    model = freeze.build_headline_model()
    cfg = model.config

    assert cfg.image_size == archived["image_size"] == freeze.HEADLINE_IMAGE_SIZE
    assert cfg.n_qubits == archived["n_qubits"] == freeze.HEADLINE_N_QUBITS
    assert cfg.n_features == archived["n_features"]
    assert cfg.encoding_type == archived["encoding_type"] == freeze.HEADLINE_ENCODING
    assert cfg.n_conv_layers == archived["n_conv_layers"]
    assert cfg.device == archived["device"]
    assert cfg.shots is archived["shots"] is None


def test_headline_allocates_269_parameter_slots():
    model = freeze.build_headline_model()
    flat = model._flatten_params(model.quantum_params)
    assert flat.shape == (freeze.HEADLINE_N_PARAM_SLOTS,)


def test_archived_weights_load_into_the_headline_layout():
    model = freeze.build_headline_model()
    params = freeze.load_archived_params(model)
    assert params.shape == (freeze.HEADLINE_N_PARAM_SLOTS,)
    assert np.all(np.isfinite(np.array(params)))


def test_slot_ranges_tile_the_parameter_vector():
    model = freeze.build_headline_model()
    ranges = freeze.slot_ranges(model)
    assert list(ranges) == [
        "quantum_conv_kernel_0", "quantum_conv_kernel_1",
        "quantum_conv_kernel_2", "quantum_conv_kernel_3",
        "quantum_pooling_0", "quantum_pooling_1", "quantum_pooling_2",
        "quantum_classifier",
    ]
    cursor = 0
    for start, stop in ranges.values():
        assert start == cursor
        cursor = stop
    assert cursor == freeze.HEADLINE_N_PARAM_SLOTS


def test_regression_inputs_are_fixed_and_reproducible():
    first = freeze.fixed_regression_inputs()
    second = freeze.fixed_regression_inputs()
    assert first.shape == (freeze.N_REGRESSION_INPUTS, 2 ** freeze.HEADLINE_N_QUBITS)
    np.testing.assert_array_equal(first, second)
```

- [ ] **Step 4: Run the test to verify it fails**

Run: `python -m pytest tests/test_headline_identity.py -v`
Expected: collection error — `ModuleNotFoundError: No module named 'QCNN.freeze'`

- [ ] **Step 5: Implement `QCNN/freeze.py` (identity portion)**

Create `QCNN/freeze.py`:

```python
"""Freeze machinery for the FQCNN headline model (UPGRADE_PLAN.md §A2).

The protected object is the function ``(x, theta) -> <Z_readout>``, not the
incidental layout of the source. Everything here exists to pin that function
down so structural or semantic drift becomes a test failure instead of a silent
change in the paper's numbers.
"""
import os

import numpy as np
import pennylane.numpy as pnp

from QCNN.config.Qconfig import QuantumNativeConfig
from QCNN.models.QCNNModel import PureQuantumNativeCNN

# Identity of the archived headline run, mirrored from Results/metadata.json.
HEADLINE_IMAGE_SIZE = 28
HEADLINE_ENCODING = 'amplitude'
HEADLINE_SEED = 42
HEADLINE_N_QUBITS = 10
HEADLINE_N_PARAM_SLOTS = 269
HEADLINE_WEIGHTS = os.path.join('Results', 'Weights', 'quantum_model_params.npz')

FIXTURE_DIR = os.path.join('tests', 'fixtures')

# Regression inputs are drawn once, from a fixed seed, and never re-drawn.
N_REGRESSION_INPUTS = 20
REGRESSION_INPUT_SEED = 20260725
REGRESSION_TOL = 1e-10


def build_headline_model() -> PureQuantumNativeCNN:
    """The frozen headline model with freshly seeded (untrained) parameters."""
    cfg = QuantumNativeConfig.from_image_size(HEADLINE_IMAGE_SIZE, HEADLINE_ENCODING)
    cfg.seed = HEADLINE_SEED
    return PureQuantumNativeCNN(cfg)


def load_archived_params(model: PureQuantumNativeCNN) -> pnp.ndarray:
    """Archived trained weights, flattened in the model's own slot order."""
    data = np.load(HEADLINE_WEIGHTS)
    ordered = [pnp.array(data[key], requires_grad=True) for key in model.quantum_params]
    return pnp.concatenate([p.flatten() for p in ordered])


def fixed_regression_inputs() -> np.ndarray:
    """The committed regression inputs: 20 fixed vectors in the encoded length."""
    rng = np.random.default_rng(REGRESSION_INPUT_SEED)
    return rng.random((N_REGRESSION_INPUTS, 2 ** HEADLINE_N_QUBITS))


def slot_ranges(model: PureQuantumNativeCNN) -> dict:
    """Named ``[start, stop)`` ranges of each parameter group in the flat vector."""
    ranges = {}
    cursor = 0
    for name, block in model.quantum_params.items():
        size = int(np.prod(block.shape))
        ranges[name] = [cursor, cursor + size]
        cursor += size
    return ranges
```

- [ ] **Step 6: Write `tests/conftest.py`**

The headline model takes a few seconds to build and the archived weights are read-only, so both are session-scoped.

```python
"""Shared fixtures for the freeze and protocol test suites."""
import pytest

from QCNN import freeze


@pytest.fixture(scope="session")
def headline_model():
    """The frozen headline model. Read-only: never mutate its parameters."""
    return freeze.build_headline_model()


@pytest.fixture(scope="session")
def archived_params(headline_model):
    return freeze.load_archived_params(headline_model)


@pytest.fixture(scope="session")
def regression_inputs():
    return freeze.fixed_regression_inputs()
```

- [ ] **Step 7: Run the test to verify it passes**

Run: `python -m pytest tests/test_headline_identity.py -v`
Expected: `5 passed`

If `test_headline_config_matches_archived_metadata` errors with `FileNotFoundError` on `Results/metadata.json`, the file exists locally but is currently git-ignored — Step 8 fixes the tracking, not the file.

- [ ] **Step 8: Track the headline metadata**

`Results/metadata.json` defines which archived run the freeze protects, so it must be reviewable. Edit `.gitignore` and delete this line:

```
Results/metadata.json
```

Then force-add the file (it was ignored until now):

Run: `git add -f Results/metadata.json`
Expected: no output.

- [ ] **Step 9: Commit**

```bash
git add pytest.ini QCNN/freeze.py tests/conftest.py tests/test_headline_identity.py .gitignore Results/metadata.json
git commit -m "Test: pin headline model identity behind pytest gate"
```

---

## Task 2: Circuit signature guard

Serialises the frozen topology so any accidental structural edit or parameter-slot rewiring fails a test.

**Files:**
- Modify: `QCNN/freeze.py`
- Create: `scripts/regen_freeze_fixtures.py`
- Create: `tests/fixtures/headline_signature.json`
- Create: `tests/test_freeze_architecture.py`

**Interfaces:**
- Consumes: `freeze.build_headline_model`, `freeze.fixed_regression_inputs`, `freeze.FIXTURE_DIR`.
- Produces:
  - `QCNN.freeze.headline_tape(model, x, flat_params) -> QuantumScript`
  - `QCNN.freeze.circuit_signature(model) -> dict` with keys `operations` (list of `{name, wires, params}`) and `measurements` (list of str)
  - `QCNN.freeze.signature_hash(signature) -> str` (sha256 hex)
  - `QCNN.freeze.SIGNATURE_FIXTURE: str` path constant

The parameter descriptors are the key idea: the circuit is constructed on a **marker vector** whose slot `i` holds the value `i + 1`. Because every rotation in this model consumes a slot value directly, with no arithmetic, a gate reading slot `i` is recorded as `"slot{i}"` and a hard-coded angle as `"const{value}"`. Structural edits *and* slot rewiring both change the serialisation.

- [ ] **Step 1: Write the failing signature test**

Create `tests/test_freeze_architecture.py`:

```python
"""Structural guards for the frozen circuit (UPGRADE_PLAN.md §A2, §A3)."""
import json

import pytest

from QCNN import freeze


def test_circuit_signature_matches_committed_reference(headline_model):
    with open(freeze.SIGNATURE_FIXTURE) as fh:
        reference = json.load(fh)

    signature = freeze.circuit_signature(headline_model)

    assert freeze.signature_hash(signature) == reference["hash"], (
        "The frozen circuit's topology changed. Under UPGRADE_PLAN.md A1 this is "
        "only permitted for a gate proven inert; otherwise revert the change."
    )
    assert signature == reference["signature"]


def test_signature_records_the_known_frozen_topology(headline_model):
    signature = freeze.circuit_signature(headline_model)
    names = [op["name"] for op in signature["operations"]]

    # One amplitude state preparation, one terminal expectation value.
    assert names.count("AmplitudeEmbedding") == 1
    assert names[0] == "AmplitudeEmbedding"
    assert signature["measurements"] == ["expval(Z(0))"]

    # F1/F2: one effective convolution stage and eight pooling pairs at n=10
    # (5 + 2 + 1), with one wire retired unpaired at the 5-active stage.
    assert names.count("CRY") == names.count("CRZ") == 8


def test_only_hard_coded_angle_is_the_inert_discard_rotation(headline_model):
    signature = freeze.circuit_signature(headline_model)
    constants = {
        descriptor
        for op in signature["operations"]
        for descriptor in op["params"]
        if descriptor.startswith("const")
    }
    # F4: the RY(0.02) on already-discarded wires is the sole magic constant.
    # Removing it is governed by the A2 inert-gate exception and is NOT Phase 0 work.
    assert constants == {"const0.02"}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_freeze_architecture.py -v`
Expected: FAIL with `AttributeError: module 'QCNN.freeze' has no attribute 'SIGNATURE_FIXTURE'`

- [ ] **Step 3: Implement the signature functions**

Append to `QCNN/freeze.py` (and add `import hashlib`, `import json`, `import pennylane as qml` to the imports at the top):

```python
SIGNATURE_FIXTURE = os.path.join(FIXTURE_DIR, 'headline_signature.json')

# State-preparation operations carry data, not trainable slots; their parameters
# are described by length so the signature stays a topology statement.
_STATE_PREP_OPS = ('AmplitudeEmbedding', 'StatePrep', 'MottonenStatePreparation')


def headline_tape(model, x, flat_params):
    """Construct and return the QNode's tape for one (input, parameter) pair."""
    model.quantum_circuit.construct((x, flat_params), {})
    return model.quantum_circuit.tape


def _marker_vector(n_slots: int) -> pnp.ndarray:
    """Parameter vector whose slot ``i`` carries the unique tag ``i + 1``."""
    return pnp.array(np.arange(1, n_slots + 1, dtype=float), requires_grad=True)


def _describe_param(value, lookup) -> str:
    slot = lookup.get(round(float(value), 9))
    if slot is not None:
        return 'slot{}'.format(slot)
    return 'const{:.12g}'.format(float(value))


def circuit_signature(model) -> dict:
    """Serialise the frozen topology: gate name, wires, and parameter slot."""
    n_slots = len(model._flatten_params(model.quantum_params))
    marker = _marker_vector(n_slots)
    lookup = {round(float(i + 1), 9): i for i in range(n_slots)}
    tape = headline_tape(model, fixed_regression_inputs()[0], marker)

    operations = []
    for op in tape.operations:
        if op.name in _STATE_PREP_OPS:
            params = ['data{}'.format(int(np.shape(op.data[0])[0]))]
        else:
            params = [_describe_param(p, lookup) for p in op.data]
        operations.append({
            'name': op.name,
            'wires': [int(w) for w in op.wires],
            'params': params,
        })

    return {
        'operations': operations,
        'measurements': [str(m) for m in tape.measurements],
    }


def signature_hash(signature: dict) -> str:
    blob = json.dumps(signature, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(blob.encode('utf-8')).hexdigest()
```

- [ ] **Step 4: Write the fixture generator**

Create `scripts/regen_freeze_fixtures.py`. Task 4 and Task 5 extend it; write the whole file now so those tasks only add calls.

```python
"""Regenerate the committed freeze fixtures from the archived headline model.

Run this ONLY when a change to the frozen circuit has been approved under
UPGRADE_PLAN.md A2. Regenerating fixtures to make a failing guard pass defeats
the entire purpose of the freeze.

Usage:  python scripts/regen_freeze_fixtures.py
"""
import json
import os

import numpy as np

from QCNN import freeze


def _write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print('wrote {}'.format(path))


def regen_signature(model):
    signature = freeze.circuit_signature(model)
    _write_json(freeze.SIGNATURE_FIXTURE, {
        'hash': freeze.signature_hash(signature),
        'signature': signature,
    })


def main():
    model = freeze.build_headline_model()
    regen_signature(model)


if __name__ == '__main__':
    main()
```

- [ ] **Step 5: Generate the signature fixture**

Run: `python scripts/regen_freeze_fixtures.py`
Expected: `wrote tests/fixtures/headline_signature.json`

- [ ] **Step 6: Run the tests to verify they pass**

Run: `python -m pytest tests/test_freeze_architecture.py -v`
Expected: `3 passed`

- [ ] **Step 7: Prove the guard actually catches drift**

Temporarily add a gate to `QCNN/models/QCNNModel.py` immediately before the `return qml.expval(...)` line in `_pure_quantum_forward`:

```python
        qml.RZ(0.0, wires=readout)
```

Run: `python -m pytest tests/test_freeze_architecture.py::test_circuit_signature_matches_committed_reference -v`
Expected: FAIL with the "frozen circuit's topology changed" message.

Now **revert that edit** (`git checkout QCNN/models/QCNNModel.py`) and re-run:

Run: `python -m pytest tests/test_freeze_architecture.py -v`
Expected: `3 passed`

- [ ] **Step 8: Commit**

```bash
git add QCNN/freeze.py scripts/regen_freeze_fixtures.py tests/fixtures/headline_signature.json tests/test_freeze_architecture.py
git commit -m "Test: guard the frozen circuit topology with a signature fixture"
```

---

## Task 3: Unitarity audit

Turns §A3 from a written promise into an executable check, with a positive control so it cannot pass vacuously.

**Files:**
- Modify: `QCNN/freeze.py`
- Modify: `tests/test_freeze_architecture.py`

**Interfaces:**
- Consumes: `freeze.headline_tape`, `freeze.build_headline_model`, `freeze.load_archived_params`.
- Produces: `QCNN.freeze.unitarity_violations(tape) -> list[str]` — empty list means the tape satisfies §A3.

- [ ] **Step 1: Write the failing unitarity tests**

Append to `tests/test_freeze_architecture.py`:

```python
def test_main_path_contains_no_non_unitary_operation(headline_model, archived_params, regression_inputs):
    tape = freeze.headline_tape(headline_model, regression_inputs[0], archived_params)
    assert freeze.unitarity_violations(tape) == []


def test_unitarity_audit_detects_measurement_pooling():
    """Positive control: the audit must flag the measurement-pooling arm.

    'measurement' is the labelled ablation arm from UPGRADE_PLAN.md A3, never
    part of the headline model. If this test stops failing the audit, the audit
    has stopped working.
    """
    from QCNN.config.Qconfig import QuantumNativeConfig
    from QCNN.models.QCNNModel import PureQuantumNativeCNN

    cfg = QuantumNativeConfig.from_image_size(
        freeze.HEADLINE_IMAGE_SIZE, freeze.HEADLINE_ENCODING)
    cfg.seed = freeze.HEADLINE_SEED
    cfg.pooling_mode = 'measurement'
    cfg.device = 'default.qubit'  # lightning.qubit rejects mid-circuit measurement
    arm = PureQuantumNativeCNN(cfg)

    params = arm._flatten_params(arm.quantum_params)
    tape = freeze.headline_tape(arm, freeze.fixed_regression_inputs()[0], params)

    violations = freeze.unitarity_violations(tape)
    assert len(violations) == 16, violations  # 8 mid-circuit measurements + 8 conditionals


def test_headline_has_exactly_one_terminal_measurement(headline_model, archived_params, regression_inputs):
    tape = freeze.headline_tape(headline_model, regression_inputs[0], archived_params)
    assert len(tape.measurements) == 1
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_freeze_architecture.py -v -k "unitarity or terminal"`
Expected: FAIL with `AttributeError: module 'QCNN.freeze' has no attribute 'unitarity_violations'`

- [ ] **Step 3: Implement the audit**

Append to `QCNN/freeze.py`:

```python
def unitarity_violations(tape) -> list:
    """Operations on the main path that break the A3 unitarity invariant.

    An empty result means the tape is pure state preparation plus unitaries up
    to a single terminal measurement: no mid-circuit measurement, no classical
    feed-forward, no non-unitary channel.
    """
    violations = []
    for op in tape.operations:
        if isinstance(op, qml.measurements.MidMeasureMP):
            violations.append('{}: mid-circuit measurement'.format(op.name))
        elif isinstance(op, qml.ops.op_math.Conditional):
            violations.append('{}: classical feed-forward'.format(op.name))
        elif isinstance(op, qml.operation.Channel):
            violations.append('{}: non-unitary channel'.format(op.name))
    return violations
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_freeze_architecture.py -v`
Expected: `6 passed`

- [ ] **Step 5: Commit**

```bash
git add QCNN/freeze.py tests/test_freeze_architecture.py
git commit -m "Test: audit main-path unitarity with a measurement-pooling control"
```

---

## Task 4: Expectation regression against archived weights

The semantic guard. This is what proves the Task 10 circuit consolidation is behaviour-preserving, so it must exist first.

**Files:**
- Modify: `QCNN/freeze.py`
- Modify: `scripts/regen_freeze_fixtures.py`
- Create: `tests/fixtures/headline_expectations.npz`
- Create: `tests/test_freeze_expectations.py`

**Interfaces:**
- Consumes: `freeze.fixed_regression_inputs`, `freeze.load_archived_params`, `freeze.REGRESSION_TOL`.
- Produces:
  - `QCNN.freeze.headline_expectations(model, flat_params, inputs=None) -> np.ndarray` (shape `(20,)`)
  - `QCNN.freeze.EXPECTATION_FIXTURE: str`

Each forward pass costs roughly 1.3 s, so 20 inputs take about 30 s. The test is marked `slow`.

- [ ] **Step 1: Write the failing regression test**

Create `tests/test_freeze_expectations.py`:

```python
"""Semantic guard: the frozen function (x, theta) -> <Z> must not drift.

This is the test that makes refactoring safe. Any change to how the circuit is
built, dispatched, or executed must leave these 20 numbers unchanged to 1e-10
(UPGRADE_PLAN.md A2).
"""
import numpy as np
import pytest

from QCNN import freeze


@pytest.mark.slow
def test_expectations_match_archived_reference(headline_model, archived_params, regression_inputs):
    reference = np.load(freeze.EXPECTATION_FIXTURE)

    np.testing.assert_array_equal(
        reference["inputs"], regression_inputs,
        err_msg="The committed regression inputs changed; the fixture is invalid.",
    )

    actual = freeze.headline_expectations(headline_model, archived_params, regression_inputs)

    np.testing.assert_allclose(
        actual, reference["expectations"], atol=freeze.REGRESSION_TOL, rtol=0.0,
        err_msg=(
            "The frozen model's outputs drifted beyond 1e-10. Under UPGRADE_PLAN.md "
            "A2 the refactor that caused this is not behaviour-preserving."
        ),
    )


@pytest.mark.slow
def test_expectations_are_deterministic(headline_model, archived_params, regression_inputs):
    first = freeze.headline_expectations(headline_model, archived_params, regression_inputs[:3])
    second = freeze.headline_expectations(headline_model, archived_params, regression_inputs[:3])
    np.testing.assert_array_equal(first, second)


def test_expectations_are_valid_pauli_z_values(headline_model, archived_params):
    reference = np.load(freeze.EXPECTATION_FIXTURE)
    assert reference["expectations"].shape == (freeze.N_REGRESSION_INPUTS,)
    assert np.all(np.abs(reference["expectations"]) <= 1.0 + 1e-12)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_freeze_expectations.py -v`
Expected: FAIL with `AttributeError: module 'QCNN.freeze' has no attribute 'EXPECTATION_FIXTURE'`

- [ ] **Step 3: Implement the expectation helper**

Append to `QCNN/freeze.py`:

```python
EXPECTATION_FIXTURE = os.path.join(FIXTURE_DIR, 'headline_expectations.npz')


def headline_expectations(model, flat_params, inputs=None) -> np.ndarray:
    """``<Z_readout>`` for each fixed regression input under the given parameters."""
    if inputs is None:
        inputs = fixed_regression_inputs()
    return np.array([
        float(model.quantum_circuit(np.asarray(x), flat_params)) for x in inputs
    ])
```

- [ ] **Step 4: Extend the fixture generator**

In `scripts/regen_freeze_fixtures.py`, add this function above `main`:

```python
def regen_expectations(model):
    params = freeze.load_archived_params(model)
    inputs = freeze.fixed_regression_inputs()
    expectations = freeze.headline_expectations(model, params, inputs)
    os.makedirs(os.path.dirname(freeze.EXPECTATION_FIXTURE), exist_ok=True)
    np.savez(freeze.EXPECTATION_FIXTURE, inputs=inputs, expectations=expectations)
    print('wrote {}'.format(freeze.EXPECTATION_FIXTURE))
```

and change `main` to:

```python
def main():
    model = freeze.build_headline_model()
    regen_signature(model)
    regen_expectations(model)
```

- [ ] **Step 5: Generate the expectation fixture**

Run: `python scripts/regen_freeze_fixtures.py`
Expected (after roughly 30 s):
```
wrote tests/fixtures/headline_signature.json
wrote tests/fixtures/headline_expectations.npz
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `python -m pytest tests/test_freeze_expectations.py -v`
Expected: `3 passed` in roughly 60 s.

- [ ] **Step 7: Prove the guard catches semantic drift**

Temporarily change the inert discard rotation in `QCNN/layers/QPool.py:80` from `qml.RY(0.02, wires=discard)` to `qml.RY(0.03, wires=discard)`.

Run: `python -m pytest tests/test_freeze_expectations.py::test_expectations_match_archived_reference -v`
Expected: FAIL with the "outputs drifted beyond 1e-10" message.

> Note for the paper record: this failure is itself evidence that `RY(0.02)` is **not** inert as currently applied — the discarded wire still influences the readout through the CRY/CRZ pairing of *later* pooling stages. Record this observation in Task 11's reconciliation table; do not act on it here.

**Revert the edit** (`git checkout QCNN/layers/QPool.py`) and re-run:

Run: `python -m pytest tests/test_freeze_expectations.py -v`
Expected: `3 passed`

- [ ] **Step 8: Commit**

```bash
git add QCNN/freeze.py scripts/regen_freeze_fixtures.py tests/fixtures/headline_expectations.npz tests/test_freeze_expectations.py
git commit -m "Test: pin frozen-model expectations to archived weights at 1e-10"
```

---

## Task 5: Effective-parameter audit

Converts F1 from a hidden defect into a disclosed, regression-guarded property, and produces the single authoritative count that later parameter-matched baselines and resource tables consume.

**Files:**
- Modify: `QCNN/freeze.py`
- Modify: `scripts/regen_freeze_fixtures.py`
- Create: `tests/fixtures/effective_params.json`
- Create: `tests/test_effective_params.py`

**Interfaces:**
- Consumes: `freeze.slot_ranges`, `freeze.fixed_regression_inputs`, `freeze.load_archived_params`, `freeze.circuit_signature`.
- Produces:
  - `QCNN.freeze.effective_parameter_audit(model, flat_params, inputs) -> dict` with keys `n_allocated`, `n_syntactically_used`, `syntactically_used_slots`, `n_effective`, `effective_slots`, `zero_gradient_slots`, `per_group`, `max_abs_gradient`
  - `QCNN.freeze.EFFECTIVE_PARAMS_FIXTURE: str`

The audit distinguishes three populations the manuscript must report separately: **allocated** slots (269), slots the circuit **syntactically reads**, and slots with a **nonzero gradient** on at least one fixed input.

- [ ] **Step 1: Write the failing audit test**

Create `tests/test_effective_params.py`:

```python
"""Gradient audit of the frozen parameter vector (UPGRADE_PLAN.md 0.2, F1).

269 slots are allocated. Far fewer influence the output. The manuscript must
report both numbers, so both are pinned here.
"""
import json

import numpy as np
import pytest

from QCNN import freeze


@pytest.fixture(scope="module")
def committed_audit():
    with open(freeze.EFFECTIVE_PARAMS_FIXTURE) as fh:
        return json.load(fh)


def test_allocated_slot_count_is_disclosed(committed_audit):
    assert committed_audit["n_allocated"] == freeze.HEADLINE_N_PARAM_SLOTS


def test_effective_count_is_far_below_allocated(committed_audit):
    """F1: the majority of allocated slots receive no gradient."""
    assert committed_audit["n_effective"] < committed_audit["n_allocated"]
    assert committed_audit["n_effective"] == len(committed_audit["effective_slots"])
    assert (
        committed_audit["n_effective"] + len(committed_audit["zero_gradient_slots"])
        == committed_audit["n_allocated"]
    )


def test_effective_slots_are_a_subset_of_syntactically_used_slots(committed_audit):
    used = set(committed_audit["syntactically_used_slots"])
    effective = set(committed_audit["effective_slots"])
    assert effective <= used


def test_per_group_counts_sum_to_the_totals(committed_audit):
    per_group = committed_audit["per_group"]
    assert sum(g["n_allocated"] for g in per_group.values()) == committed_audit["n_allocated"]
    assert sum(g["n_effective"] for g in per_group.values()) == committed_audit["n_effective"]


@pytest.mark.slow
def test_audit_reproduces_the_committed_artifact(headline_model, archived_params, regression_inputs, committed_audit):
    fresh = freeze.effective_parameter_audit(headline_model, archived_params, regression_inputs)

    assert fresh["n_allocated"] == committed_audit["n_allocated"]
    assert fresh["n_effective"] == committed_audit["n_effective"]
    assert fresh["effective_slots"] == committed_audit["effective_slots"]
    assert fresh["syntactically_used_slots"] == committed_audit["syntactically_used_slots"]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_effective_params.py -v`
Expected: FAIL with `AttributeError: module 'QCNN.freeze' has no attribute 'EFFECTIVE_PARAMS_FIXTURE'`

- [ ] **Step 3: Implement the audit**

Append to `QCNN/freeze.py`:

```python
EFFECTIVE_PARAMS_FIXTURE = os.path.join(FIXTURE_DIR, 'effective_params.json')


def syntactically_used_slots(model) -> list:
    """Slots that appear as a gate argument anywhere in the frozen tape."""
    used = set()
    for op in circuit_signature(model)['operations']:
        for descriptor in op['params']:
            if descriptor.startswith('slot'):
                used.add(int(descriptor[len('slot'):]))
    return sorted(used)


def effective_parameter_audit(model, flat_params, inputs) -> dict:
    """Which parameter slots actually influence ``<Z_readout>``.

    A slot is *effective* when its gradient is nonzero for at least one of the
    fixed regression inputs. Slots that are identically zero across every input
    are allocated but dead (UPGRADE_PLAN.md F1).
    """
    jacobian = np.array([
        np.asarray(qml.jacobian(lambda p: model.quantum_circuit(np.asarray(x), p))(flat_params))
        for x in inputs
    ])
    max_abs = np.max(np.abs(jacobian), axis=0)

    effective = [int(i) for i in np.flatnonzero(max_abs > 0.0)]
    dead = [int(i) for i in np.flatnonzero(max_abs == 0.0)]
    used = syntactically_used_slots(model)

    per_group = {}
    for name, (start, stop) in slot_ranges(model).items():
        group = [i for i in effective if start <= i < stop]
        per_group[name] = {
            'range': [start, stop],
            'n_allocated': stop - start,
            'n_effective': len(group),
        }

    return {
        'n_allocated': int(len(max_abs)),
        'n_syntactically_used': len(used),
        'syntactically_used_slots': used,
        'n_effective': len(effective),
        'effective_slots': effective,
        'zero_gradient_slots': dead,
        'per_group': per_group,
        'max_abs_gradient': [float(v) for v in max_abs],
        'n_inputs': int(len(inputs)),
        'input_seed': REGRESSION_INPUT_SEED,
    }
```

- [ ] **Step 4: Extend the fixture generator**

In `scripts/regen_freeze_fixtures.py`, add above `main`:

```python
def regen_effective_params(model):
    params = freeze.load_archived_params(model)
    inputs = freeze.fixed_regression_inputs()
    audit = freeze.effective_parameter_audit(model, params, inputs)
    _write_json(freeze.EFFECTIVE_PARAMS_FIXTURE, audit)
    print('  allocated={} syntactically used={} effective={}'.format(
        audit['n_allocated'], audit['n_syntactically_used'], audit['n_effective']))
```

and extend `main`:

```python
def main():
    model = freeze.build_headline_model()
    regen_signature(model)
    regen_expectations(model)
    regen_effective_params(model)
```

- [ ] **Step 5: Generate the audit fixture**

Run: `python scripts/regen_freeze_fixtures.py`
Expected (roughly 60 s total): the three `wrote ...` lines plus a summary line of the form
`  allocated=269 syntactically used=<N> effective=<M>` with `M < 269`.

**Record the printed numbers.** They are the authoritative counts. `UPGRADE_PLAN.md` estimated ~76 effective and the design note observed 74 under a different probe; the committed artifact supersedes both, and Task 11 records the measured value with its tolerance.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `python -m pytest tests/test_effective_params.py -v`
Expected: `5 passed`

- [ ] **Step 7: Commit**

```bash
git add QCNN/freeze.py scripts/regen_freeze_fixtures.py tests/fixtures/effective_params.json tests/test_effective_params.py
git commit -m "Test: commit the effective-parameter gradient audit"
```

---

## Task 6: Split service with persisted manifests

The centre of the leakage fix (F3). Everything downstream — trainer, runners, baselines, noise, hardware — consumes manifests instead of re-splitting.

**Files:**
- Create: `QCNN/utils/splits.py`
- Create: `tests/test_splits.py`
- Create: `Results/manifests/.gitkeep`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `QCNN.utils.splits.SPLIT_FRACTIONS: tuple = (0.60, 0.15, 0.25)`
  - `make_split_manifest(labels, seed, dataset_id, class_mapping, sample_ids=None) -> dict`
  - `save_manifest(manifest, path) -> None`
  - `load_manifest(path) -> dict`
  - `verify_manifest(manifest) -> None` (raises `ValueError`)
  - `apply_manifest(manifest, X, y) -> tuple` of six arrays in the order `(X_train, y_train, X_val, y_val, X_test, y_test)`
  - `manifest_id(manifest) -> str` (sha256 hex over the split indices)

  Manifest keys: `dataset_id`, `class_mapping`, `seed`, `fractions`, `n_total`, `sample_ids`, `train_idx`, `val_idx`, `test_idx`, `id`.

- [ ] **Step 1: Write the failing split tests**

Create `tests/test_splits.py`:

```python
"""Split protocol tests (UPGRADE_PLAN.md 0.3, F3).

Model selection must never see the test set, so the split has to be
deterministic, stratified, disjoint, and recorded.
"""
import os

import numpy as np
import pytest

from QCNN.utils import splits


@pytest.fixture
def labels():
    # Deliberately imbalanced so stratification is observable.
    return np.array([1] * 120 + [-1] * 80)


def test_fractions_are_the_mandated_60_15_25():
    assert splits.SPLIT_FRACTIONS == (0.60, 0.15, 0.25)


def test_split_sizes_follow_the_fractions(labels):
    manifest = splits.make_split_manifest(labels, seed=0, dataset_id="unit", class_mapping={"1": 1, "-1": -1})
    assert len(manifest["train_idx"]) == 120
    assert len(manifest["val_idx"]) == 30
    assert len(manifest["test_idx"]) == 50


def test_splits_are_disjoint_and_exhaustive(labels):
    manifest = splits.make_split_manifest(labels, seed=0, dataset_id="unit", class_mapping={})
    train, val, test = (set(manifest[k]) for k in ("train_idx", "val_idx", "test_idx"))
    assert train & val == set()
    assert train & test == set()
    assert val & test == set()
    assert train | val | test == set(range(len(labels)))


def test_splits_are_stratified(labels):
    manifest = splits.make_split_manifest(labels, seed=0, dataset_id="unit", class_mapping={})
    overall = np.mean(labels == 1)
    for key in ("train_idx", "val_idx", "test_idx"):
        part = labels[np.array(manifest[key])]
        assert abs(np.mean(part == 1) - overall) < 0.05


def test_same_seed_reproduces_the_same_split(labels):
    a = splits.make_split_manifest(labels, seed=7, dataset_id="unit", class_mapping={})
    b = splits.make_split_manifest(labels, seed=7, dataset_id="unit", class_mapping={})
    assert a["train_idx"] == b["train_idx"]
    assert a["val_idx"] == b["val_idx"]
    assert a["test_idx"] == b["test_idx"]
    assert a["id"] == b["id"]


def test_different_seeds_give_different_splits(labels):
    a = splits.make_split_manifest(labels, seed=1, dataset_id="unit", class_mapping={})
    b = splits.make_split_manifest(labels, seed=2, dataset_id="unit", class_mapping={})
    assert a["id"] != b["id"]


def test_manifest_round_trips_through_disk(labels, tmp_path):
    manifest = splits.make_split_manifest(labels, seed=3, dataset_id="unit", class_mapping={"a": 1})
    path = os.path.join(str(tmp_path), "split.json")
    splits.save_manifest(manifest, path)
    assert splits.load_manifest(path) == manifest


def test_apply_manifest_returns_matching_features_and_labels(labels):
    X = np.arange(len(labels) * 4, dtype=float).reshape(len(labels), 4)
    manifest = splits.make_split_manifest(labels, seed=0, dataset_id="unit", class_mapping={})
    X_tr, y_tr, X_va, y_va, X_te, y_te = splits.apply_manifest(manifest, X, labels)

    assert len(X_tr) == len(y_tr) == 120
    assert len(X_va) == len(y_va) == 30
    assert len(X_te) == len(y_te) == 50
    # Row identity is preserved: column 0 equals 4 * original index.
    np.testing.assert_array_equal(X_tr[:, 0], 4.0 * np.array(manifest["train_idx"]))


def test_verify_manifest_rejects_an_overlapping_split(labels):
    manifest = splits.make_split_manifest(labels, seed=0, dataset_id="unit", class_mapping={})
    manifest["val_idx"] = manifest["val_idx"] + [manifest["train_idx"][0]]
    with pytest.raises(ValueError, match="disjoint"):
        splits.verify_manifest(manifest)


def test_apply_manifest_rejects_a_size_mismatch(labels):
    manifest = splits.make_split_manifest(labels, seed=0, dataset_id="unit", class_mapping={})
    with pytest.raises(ValueError, match="n_total"):
        splits.apply_manifest(manifest, np.zeros((10, 4)), labels[:10])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_splits.py -v`
Expected: collection error — `ModuleNotFoundError: No module named 'QCNN.utils.splits'`

- [ ] **Step 3: Implement the split service**

Create `QCNN/utils/splits.py`:

```python
"""Deterministic stratified train/validation/test splits with persisted manifests.

UPGRADE_PLAN.md 0.3 / F3: model selection and early stopping previously read the
test set. The fix is protocol, not architecture: one seeded stratified 60/15/25
split, recorded as a manifest that every model, baseline, and analysis consumes.

A run never reconstructs a split by replaying an assumed RNG sequence; it loads
the recorded indices.
"""
import hashlib
import json
import os

import numpy as np
from sklearn.model_selection import train_test_split

# train / validation / test. Fixed by the upgrade plan; not a tunable.
SPLIT_FRACTIONS = (0.60, 0.15, 0.25)


def manifest_id(manifest: dict) -> str:
    """Stable identity of a split, over its indices and provenance."""
    payload = {
        'dataset_id': manifest['dataset_id'],
        'seed': manifest['seed'],
        'fractions': list(manifest['fractions']),
        'train_idx': manifest['train_idx'],
        'val_idx': manifest['val_idx'],
        'test_idx': manifest['test_idx'],
    }
    blob = json.dumps(payload, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(blob.encode('utf-8')).hexdigest()


def make_split_manifest(labels, seed, dataset_id, class_mapping, sample_ids=None) -> dict:
    """Build a seeded stratified 60/15/25 manifest over ``labels``.

    ``sample_ids`` records where each row came from in the source dataset so the
    selection is auditable; it defaults to positional identity.
    """
    labels = np.asarray(labels)
    n_total = len(labels)
    if sample_ids is None:
        sample_ids = list(range(n_total))
    if len(sample_ids) != n_total:
        raise ValueError('sample_ids has {} entries for {} labels'.format(len(sample_ids), n_total))

    train_frac, val_frac, test_frac = SPLIT_FRACTIONS
    positions = np.arange(n_total)

    train_idx, holdout_idx = train_test_split(
        positions,
        train_size=train_frac,
        random_state=seed,
        stratify=labels,
    )
    # Within the 40% holdout, validation is 15/40 and test is 25/40.
    val_idx, test_idx = train_test_split(
        holdout_idx,
        train_size=val_frac / (val_frac + test_frac),
        random_state=seed,
        stratify=labels[holdout_idx],
    )

    manifest = {
        'dataset_id': dataset_id,
        'class_mapping': class_mapping,
        'seed': int(seed),
        'fractions': list(SPLIT_FRACTIONS),
        'n_total': int(n_total),
        'sample_ids': [int(s) for s in sample_ids],
        'train_idx': sorted(int(i) for i in train_idx),
        'val_idx': sorted(int(i) for i in val_idx),
        'test_idx': sorted(int(i) for i in test_idx),
    }
    manifest['id'] = manifest_id(manifest)
    verify_manifest(manifest)
    return manifest


def verify_manifest(manifest: dict) -> None:
    """Raise ``ValueError`` unless the split is disjoint and exhaustive."""
    train = set(manifest['train_idx'])
    val = set(manifest['val_idx'])
    test = set(manifest['test_idx'])

    if len(train) + len(val) + len(test) != len(train | val | test):
        raise ValueError('split partitions are not disjoint')
    if train | val | test != set(range(manifest['n_total'])):
        raise ValueError('split partitions do not cover all {} samples'.format(manifest['n_total']))


def save_manifest(manifest: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)


def load_manifest(path: str) -> dict:
    with open(path) as fh:
        manifest = json.load(fh)
    verify_manifest(manifest)
    return manifest


def apply_manifest(manifest: dict, X, y):
    """Slice ``(X, y)`` into ``(X_train, y_train, X_val, y_val, X_test, y_test)``."""
    X = np.asarray(X)
    y = np.asarray(y)
    if len(X) != manifest['n_total'] or len(y) != manifest['n_total']:
        raise ValueError(
            'manifest n_total={} but received {} samples'.format(manifest['n_total'], len(X)))

    train = np.array(manifest['train_idx'])
    val = np.array(manifest['val_idx'])
    test = np.array(manifest['test_idx'])
    return X[train], y[train], X[val], y[val], X[test], y[test]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_splits.py -v`
Expected: `10 passed`

- [ ] **Step 5: Create the manifest directory**

Manifests are immutable evidence and belong in version control.

Run: `python -c "import os; os.makedirs('Results/manifests', exist_ok=True); open('Results/manifests/.gitkeep','w').close()"`
Expected: no output.

- [ ] **Step 6: Commit**

```bash
git add QCNN/utils/splits.py tests/test_splits.py Results/manifests/.gitkeep
git commit -m "Feat: add seeded stratified 60/15/25 split manifests"
```

---

## Task 7: Run artifacts and the single-test-evaluation guard

Gives every run isolated outputs and makes "test is evaluated exactly once" a mechanism rather than a convention.

**Files:**
- Create: `QCNN/utils/run_artifacts.py`
- Create: `tests/test_run_artifacts.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `QCNN.utils.run_artifacts.RUN_ROOT: str = os.path.join('Results', 'runs')`
  - `run_dir(dataset_id, config_name, seed, root=RUN_ROOT) -> str`
  - `start_run(directory, config, split_id, seed, environment) -> None` (writes `status.json` with `state='running'`)
  - `complete_run(directory, metrics) -> None` (`state='complete'`)
  - `fail_run(directory, error) -> None` (`state='failed'`)
  - `is_complete(directory) -> bool`
  - `save_predictions(directory, sample_ids, y_true, raw_outputs) -> str`
  - `weights_path(directory) -> str`
  - `class TestEvaluationGuard` with `.count: int` and `.evaluate(fn, *args, **kwargs)`

- [ ] **Step 1: Write the failing artifact tests**

Create `tests/test_run_artifacts.py`:

```python
"""Run isolation and provenance (UPGRADE_PLAN.md M0.5).

Two runs must not be able to overwrite each other's outputs or the archived
headline weights, partial output must never look complete, and the test set must
be evaluated exactly once per run.
"""
import json
import os

import numpy as np
import pytest

from QCNN.utils import run_artifacts


def test_runs_get_distinct_directories(tmp_path):
    root = str(tmp_path)
    a = run_artifacts.run_dir("0v1", "headline", 0, root=root)
    b = run_artifacts.run_dir("0v1", "headline", 1, root=root)
    c = run_artifacts.run_dir("3v5", "headline", 0, root=root)
    assert len({a, b, c}) == 3


def test_run_weights_never_target_the_archived_headline_file(tmp_path):
    directory = run_artifacts.run_dir("0v1", "headline", 0, root=str(tmp_path))
    archived = os.path.normpath(os.path.join("Results", "Weights", "quantum_model_params.npz"))
    assert os.path.normpath(run_artifacts.weights_path(directory)) != archived


def test_status_lifecycle_pending_running_complete(tmp_path):
    directory = run_artifacts.run_dir("0v1", "headline", 0, root=str(tmp_path))

    assert run_artifacts.is_complete(directory) is False

    run_artifacts.start_run(directory, config={"n_qubits": 10}, split_id="abc",
                            seed=0, environment={"python": "3.9.13"})
    assert run_artifacts.is_complete(directory) is False

    run_artifacts.complete_run(directory, metrics={"accuracy": 0.9})
    assert run_artifacts.is_complete(directory) is True

    with open(os.path.join(directory, "status.json")) as fh:
        status = json.load(fh)
    assert status["state"] == "complete"
    assert status["split_id"] == "abc"


def test_failed_run_is_not_complete(tmp_path):
    directory = run_artifacts.run_dir("0v1", "headline", 0, root=str(tmp_path))
    run_artifacts.start_run(directory, config={}, split_id="abc", seed=0, environment={})
    run_artifacts.fail_run(directory, error="simulator crashed")

    assert run_artifacts.is_complete(directory) is False
    with open(os.path.join(directory, "status.json")) as fh:
        assert json.load(fh)["state"] == "failed"


def test_per_example_predictions_are_persisted_for_paired_statistics(tmp_path):
    directory = run_artifacts.run_dir("0v1", "headline", 0, root=str(tmp_path))
    run_artifacts.start_run(directory, config={}, split_id="abc", seed=0, environment={})

    path = run_artifacts.save_predictions(
        directory,
        sample_ids=[11, 22, 33],
        y_true=np.array([1, -1, 1]),
        raw_outputs=np.array([0.4, -0.7, 0.1]),
    )
    stored = np.load(path)
    np.testing.assert_array_equal(stored["sample_ids"], [11, 22, 33])
    np.testing.assert_array_equal(stored["y_true"], [1, -1, 1])
    np.testing.assert_allclose(stored["raw_outputs"], [0.4, -0.7, 0.1])


def test_test_evaluation_guard_allows_exactly_one_evaluation():
    guard = run_artifacts.TestEvaluationGuard()
    assert guard.evaluate(lambda: "metrics") == "metrics"
    assert guard.count == 1

    with pytest.raises(RuntimeError, match="exactly once"):
        guard.evaluate(lambda: "metrics again")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_run_artifacts.py -v`
Expected: collection error — `ModuleNotFoundError: No module named 'QCNN.utils.run_artifacts'`

- [ ] **Step 3: Implement the run artifact store**

Create `QCNN/utils/run_artifacts.py`:

```python
"""Isolated, traceable outputs for one (dataset, config, seed) run.

UPGRADE_PLAN.md M0.5: results must be attributable and parallel-safe. Each run
owns a directory; the archived headline weights are read-only input and are
never a worker's output destination.
"""
import json
import os

import numpy as np

RUN_ROOT = os.path.join('Results', 'runs')

_STATUS = 'status.json'
_PREDICTIONS = 'predictions.npz'
_WEIGHTS = 'weights.npz'


def run_dir(dataset_id: str, config_name: str, seed: int, root: str = RUN_ROOT) -> str:
    """Directory owned exclusively by this run. Created if absent."""
    directory = os.path.join(root, str(dataset_id), str(config_name), 'seed_{}'.format(seed))
    os.makedirs(directory, exist_ok=True)
    return directory


def weights_path(directory: str) -> str:
    return os.path.join(directory, _WEIGHTS)


def _write_status(directory: str, payload: dict) -> None:
    with open(os.path.join(directory, _STATUS), 'w') as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)


def _read_status(directory: str) -> dict:
    path = os.path.join(directory, _STATUS)
    if not os.path.exists(path):
        return {}
    with open(path) as fh:
        return json.load(fh)


def start_run(directory: str, config: dict, split_id: str, seed: int, environment: dict) -> None:
    _write_status(directory, {
        'state': 'running',
        'config': config,
        'split_id': split_id,
        'seed': int(seed),
        'environment': environment,
    })


def complete_run(directory: str, metrics: dict) -> None:
    status = _read_status(directory)
    status['state'] = 'complete'
    status['metrics'] = metrics
    _write_status(directory, status)


def fail_run(directory: str, error: str) -> None:
    status = _read_status(directory)
    status['state'] = 'failed'
    status['error'] = str(error)
    _write_status(directory, status)


def is_complete(directory: str) -> bool:
    """True only for a run that finished. Partial output never counts."""
    return _read_status(directory).get('state') == 'complete'


def save_predictions(directory: str, sample_ids, y_true, raw_outputs) -> str:
    """Persist per-example results so paired tests (McNemar) are possible later."""
    path = os.path.join(directory, _PREDICTIONS)
    np.savez(
        path,
        sample_ids=np.asarray(sample_ids),
        y_true=np.asarray(y_true),
        raw_outputs=np.asarray(raw_outputs),
    )
    return path


class TestEvaluationGuard:
    """Permits exactly one test-set evaluation per run (UPGRADE_PLAN.md 0.3).

    Wrap the single final evaluation in ``guard.evaluate(...)``. A second call
    raises, which is what turns "test is evaluated once" from a convention into
    an enforced property.
    """

    def __init__(self):
        self.count = 0

    def evaluate(self, fn, *args, **kwargs):
        if self.count:
            raise RuntimeError(
                'the test set may be evaluated exactly once per run '
                '(UPGRADE_PLAN.md 0.3); this is evaluation number {}'.format(self.count + 1))
        self.count += 1
        return fn(*args, **kwargs)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_run_artifacts.py -v`
Expected: `6 passed`

- [ ] **Step 5: Commit**

```bash
git add QCNN/utils/run_artifacts.py tests/test_run_artifacts.py
git commit -m "Feat: isolate run outputs and guard single test evaluation"
```

---

## Task 8: Validation-only trainer

Removes the leakage at its source (F3). The trainer stops seeing the test set entirely.

**Files:**
- Modify: `QCNN/training/Qtrainer.py`
- Create: `tests/test_protocol.py`

**Interfaces:**
- Consumes: `run_artifacts.weights_path` (callers supply the path).
- Produces: new trainer signature —
  ```python
  QuantumNativeTrainer.train_pure_quantum_cnn(
      model, X_train, y_train, X_val, y_val,
      log_filepath='quantum_training_log.txt',
      summary_filepath='training_summary.txt',
      validate_data=True,
      weights_path=None,
  ) -> PureQuantumNativeCNN
  ```
  `weights_path=None` means **do not write weights** — the archived headline file is never overwritten by default.

- [ ] **Step 1: Write the failing protocol tests**

Create `tests/test_protocol.py`:

```python
"""No test-set information may reach model selection (UPGRADE_PLAN.md 0.3, F3).

The trainer previously drove checkpointing, LR plateau, and early stopping from
test accuracy, which is what made the archived 98.86% unusable at a Q1 venue.
"""
import inspect
import re

import numpy as np
import pytest

from QCNN.training.Qtrainer import QuantumNativeTrainer


def _trainer_source():
    return inspect.getsource(QuantumNativeTrainer)


def test_trainer_signature_takes_validation_not_test():
    params = list(inspect.signature(
        QuantumNativeTrainer.train_pure_quantum_cnn).parameters)
    assert "X_val" in params and "y_val" in params
    assert "X_test" not in params and "y_test" not in params


def test_trainer_source_never_mentions_the_test_set():
    """The grep check mandated by UPGRADE_PLAN.md 0.3."""
    offenders = re.findall(r"\b(?:X_test|y_test|test_accuracy|test_preds)\b", _trainer_source())
    assert offenders == [], "test-set identifiers still present in the trainer: {}".format(offenders)


def test_trainer_selects_on_validation_accuracy():
    source = _trainer_source()
    assert "val_accuracy" in source
    assert "best_val_accuracy" in source


def test_trainer_does_not_write_the_archived_headline_weights():
    source = _trainer_source()
    assert "quantum_model_params.npz" not in source, (
        "the trainer must write to a caller-supplied per-run path, never the "
        "archived headline weights (UPGRADE_PLAN.md M0.5)"
    )


def test_weight_saving_is_conditional_on_a_caller_supplied_path():
    """With weights_path=None the trainer must not write any .npz file."""
    params = inspect.signature(QuantumNativeTrainer.train_pure_quantum_cnn).parameters
    assert params["weights_path"].default is None

    source = inspect.getsource(QuantumNativeTrainer.train_pure_quantum_cnn)
    assert "if weights_path" in source
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_protocol.py -v`
Expected: 5 failures, starting with `assert 'X_val' in params`.

- [ ] **Step 3: Rewrite the trainer's control flow**

In `QCNN/training/Qtrainer.py`, replace the method signature (currently lines 35–40):

```python
    def train_pure_quantum_cnn(self, model: PureQuantumNativeCNN,
                               X_train: np.ndarray, y_train: np.ndarray,
                               X_val: np.ndarray, y_val: np.ndarray,
                               log_filepath='quantum_training_log.txt',
                               summary_filepath='training_summary.txt',
                               validate_data: bool = True,
                               weights_path: str = None) -> PureQuantumNativeCNN:
        """Train on X_train, select on X_val. The test set is never passed here.

        UPGRADE_PLAN.md 0.3 / F3: checkpointing, LR plateau, and early stopping
        read validation only; the caller evaluates test exactly once afterwards.

        ``weights_path=None`` writes no weights, so the archived headline file
        cannot be clobbered by a training run.
        """
```

Then apply these substitutions throughout the method body:

| Current | Replacement |
|---|---|
| `self._validate_dataset(X_test, y_test, model)` | `self._validate_dataset(X_val, y_val, model)` |
| `X_test = cached['X_test']` / `X_test=X_test` in the patch cache block | `X_val = cached['X_val']` / `X_val=X_val` |
| `X_combined = np.concatenate([X_train, X_test], axis=0)` | `X_combined = np.concatenate([X_train, X_val], axis=0)` |
| `X_test = X_processed_all[n_train:]` | `X_val = X_processed_all[n_train:]` |
| `model._quantum_preprocessed_test = X_test` | `model._quantum_preprocessed_val = X_val` |
| `best_accuracy = 0` | `best_val_accuracy = 0` |
| `test_preds = model.quantum_predict_batch(X_test)` | `val_preds = model.quantum_predict_batch(X_val)` |
| `test_accuracy = np.mean(test_preds == y_test)` | `val_accuracy = np.mean(val_preds == y_val)` |
| the four `tp/tn/fp/fn` lines over `test_preds`/`y_test` | same lines over `val_preds`/`y_val` |
| `model.training_history['accuracy'].append(float(test_accuracy))` | `model.training_history['accuracy'].append(float(val_accuracy))` |
| `if test_accuracy > best_accuracy:` | `if val_accuracy > best_val_accuracy:` |
| the `[STABILITY] Test accuracy improved` message | `[STABILITY] Validation accuracy improved: {best_val_accuracy:.2%} -> {val_accuracy:.2%} (via EMA). Saving best params.` |
| `best_accuracy = test_accuracy` | `best_val_accuracy = val_accuracy` |
| `Test Acc: {test_accuracy:.1%}` in `epoch_summary_text` | `Val Acc: {val_accuracy:.1%}` |
| `final_msg = f"\nFinal Best Test Accuracy: {best_accuracy:.2%}\n"` | `final_msg = f"\nFinal Best Validation Accuracy: {best_val_accuracy:.2%}\n"` |
| `print(f"\nBest Quantum Test Accuracy: {best_accuracy:.3f}")` | `print(f"\nBest Quantum Validation Accuracy: {best_val_accuracy:.3f}")` |

Finally replace the weight-saving block (currently lines 226–230):

```python
        if weights_path:
            os.makedirs(os.path.dirname(weights_path) or '.', exist_ok=True)
            self.save_params(model.quantum_params, weights_path)
            print(f"Saved trained quantum model parameters to '{weights_path}'")
```

Also update the patch-encoding cache key in the same method, which currently hashes `X_test[:100]`:

```python
            data_signature = np.ascontiguousarray(X_train[:100]).tobytes() + \
                             np.ascontiguousarray(X_val[:100]).tobytes()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_protocol.py -v`
Expected: `5 passed`

- [ ] **Step 5: Verify the freeze guards still hold**

The trainer change must not have touched the circuit.

Run: `python -m pytest tests/test_freeze_architecture.py tests/test_freeze_expectations.py -v`
Expected: `6 passed`

- [ ] **Step 6: Commit**

```bash
git add QCNN/training/Qtrainer.py tests/test_protocol.py
git commit -m "Fix: select models on validation only, never on the test set"
```

---

## Task 9: Route `main.py` and the experiment runner through manifests

Makes the split service and the evaluation guard the only path to a result.

**Files:**
- Create: `QCNN/utils/seeding.py`
- Modify: `main.py`
- Modify: `experiments/run_experiments.py`
- Modify: `tests/test_protocol.py`

**Interfaces:**
- Consumes: `splits.make_split_manifest`, `splits.save_manifest`, `splits.apply_manifest`, `run_artifacts.TestEvaluationGuard`, `run_artifacts.run_dir`, `run_artifacts.weights_path`, `run_artifacts.save_predictions`, the new trainer signature.
- Produces: `QCNN.utils.seeding.seed_everything(seed: int) -> None`.

- [ ] **Step 1: Write the failing runner-protocol tests**

Append to `tests/test_protocol.py`:

```python
def _read(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def test_main_uses_split_manifests_not_ad_hoc_train_test_split():
    source = _read("main.py")
    assert "train_test_split(" not in source, (
        "main.py must consume a split manifest (UPGRADE_PLAN.md 0.3)")
    assert "make_split_manifest" in source
    assert "apply_manifest" in source


def test_runner_uses_split_manifests_not_ad_hoc_train_test_split():
    source = _read("experiments/run_experiments.py")
    assert "train_test_split(" not in source
    assert "apply_manifest" in source


def test_main_evaluates_the_test_set_through_the_guard():
    source = _read("main.py")
    assert "TestEvaluationGuard" in source
    assert source.count("guard.evaluate(") == 1


def test_runner_evaluates_the_test_set_through_the_guard():
    source = _read("experiments/run_experiments.py")
    assert "TestEvaluationGuard" in source
    assert source.count("guard.evaluate(") == 1


def test_seed_policy_lives_in_one_module():
    for path in ("main.py", "experiments/run_experiments.py", "noise_sim.py"):
        source = _read(path)
        assert "def _seed_everything" not in source, (
            "{} still defines its own seed policy; import "
            "QCNN.utils.seeding.seed_everything instead".format(path))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_protocol.py -v -k "manifest or guard or seed_policy"`
Expected: 5 failures.

- [ ] **Step 3: Create the shared seed policy**

Create `QCNN/utils/seeding.py`:

```python
"""One seed policy for the whole pipeline (UPGRADE_PLAN.md 0.6).

main.py, the experiment runner, and the noise simulator each carried their own
copy of this function. Divergence between them is a reproducibility bug, so
there is now exactly one.
"""
import random

import numpy as np


def seed_everything(seed: int) -> None:
    """Seed numpy, the stdlib RNG, and PennyLane."""
    np.random.seed(seed)
    random.seed(seed)
    try:
        import pennylane as qml
        qml.set_seed(seed)
    except Exception:
        # PennyLane exposes set_seed only in some builds; numpy seeding still applies.
        pass
```

- [ ] **Step 4: Rewire `main.py`**

Replace the import of `train_test_split` (line 9) with:

```python
from QCNN.utils import splits as split_service
from QCNN.utils.run_artifacts import TestEvaluationGuard
from QCNN.utils.seeding import seed_everything
```

Delete the local `_seed_everything` definition (lines 65–75) and replace every call to `_seed_everything(seed)` with `seed_everything(seed)`.

Replace the split block (lines 232–240):

```python
    # Split data through a recorded manifest: 60/15/25, stratified, seeded.
    # Validation drives model selection; test is touched exactly once, below.
    manifest = split_service.make_split_manifest(
        y_quantum,
        seed=config.seed,
        dataset_id=f"{dataset_type}_{config.classes[0]}v{config.classes[1]}_n{len(y_quantum)}",
        class_mapping={str(config.classes[0]): 1, str(config.classes[1]): -1},
    )
    manifest_path = os.path.join(
        "Results", "manifests", f"{manifest['dataset_id']}_seed{config.seed}.json")
    split_service.save_manifest(manifest, manifest_path)
    print(f" Split manifest: {manifest_path} (id {manifest['id'][:12]})")

    X_train, y_train, X_val, y_val, X_test, y_test = split_service.apply_manifest(
        manifest, X_quantum, y_quantum)

    # Limit training samples if train_sample_size parameter set
    if train_sample_size is not None and train_sample_size < len(X_train):
        X_train = X_train[:train_sample_size]
        y_train = y_train[:train_sample_size]

    print(f" Training samples used: {len(X_train)}")
    print(f" Validation samples: {len(X_val)}")
    print(f" Test samples: {len(X_test)}")
    print(f" Class distribution (training): {dict(zip(*np.unique(y_train, return_counts=True)))}")
```

Replace the trainer call (lines 259–261):

```python
    trained_model = trainer.train_pure_quantum_cnn(
        quantum_model, X_train, y_train, X_val, y_val,
        log_filepath=log_file, summary_filepath=summary_log_file,
        weights_path=os.path.join("Results", "Weights", f"run_seed{config.seed}.npz"),
    )
```

Replace the evaluation block (lines 268–287) — the `_quantum_preprocessed_test` fallback dies with the leakage:

```python
    # The single permitted test evaluation for this run (UPGRADE_PLAN.md 0.3).
    guard = TestEvaluationGuard()
    y_eval = y_test
    raw_outputs = guard.evaluate(
        predict_raw_outputs, trained_model, X_test, already_preprocessed=False)
    metrics = compute_classification_metrics(y_eval, raw_outputs)
```

- [ ] **Step 5: Rewire `experiments/run_experiments.py`**

Replace the `train_test_split` import (line 37) with:

```python
from QCNN.utils import splits as split_service
from QCNN.utils.run_artifacts import TestEvaluationGuard
from QCNN.utils.seeding import seed_everything
```

Delete the local `_seed_everything` (lines 70–77) and replace its call in `prepare_split` with `seed_everything(cfg.seed)`.

Replace the split construction in `prepare_split` (lines 124–129):

```python
    manifest = split_service.make_split_manifest(
        y,
        seed=cfg.seed,
        dataset_id="idx_{}v{}_n{}".format(classes[0], classes[1], len(y)),
        class_mapping={str(classes[0]): 1, str(classes[1]): -1},
    )
    split_service.save_manifest(manifest, os.path.join(
        "Results", "manifests", "{}_seed{}.json".format(manifest["dataset_id"], cfg.seed)))

    cfg.split_id = manifest["id"]
    X_train, y_train, X_val, y_val, X_test, y_test = split_service.apply_manifest(manifest, X, y)
    if train_sample_size is not None and train_sample_size < len(X_train):
        X_train, y_train = X_train[:train_sample_size], y_train[:train_sample_size]
    return (X_train, y_train, X_val, y_val, X_test, y_test), manifest
```

`run_qcnn` is rewritten wholesale in Step 6; leave it alone for now.

`run_single` currently passes `*split` to the baselines, which would now hand them six arrays. Update the two baseline calls in `run_single` to pass only train and test, keeping their existing four-argument contract:

```python
    if with_baselines and config_name == "proposed":
        X_train, y_train, X_val, y_val, X_test, y_test = split
        base = run_classical_baselines(X_train, y_train, X_test, y_test, seed=seed,
                                       target_params=metrics.get("n_params"))
        base.update(run_quantum_baselines(
            X_train, y_train, X_test, y_test, seed=seed, n_qubits=cfg.n_qubits,
            n_epochs=cfg.n_epochs, learning_rate=cfg.learning_rate, use_bce=use_bce))
```

> Removing the classical baselines' own test-as-validation behaviour is Phase 5 (M5.1) work and is explicitly out of scope here. Leave a comment at that call site:
> ```python
>         # NOTE (Phase 5 / M5.1): these baselines still select on the data passed
>         # as their test set. They must be given X_val before any baseline number
>         # enters the manuscript.
> ```

- [ ] **Step 6: Persist per-example results into isolated run directories**

M0.5 requires per-example predictions (for the paired McNemar tests in Phase 5) and requires that new clean-protocol outputs are distinguishable from the historical `Results/experiments/` tree.

First add the assertion to `tests/test_protocol.py`:

```python
def test_runner_persists_per_example_predictions():
    source = _read("experiments/run_experiments.py")
    assert "save_predictions" in source, (
        "per-example predictions are required for the paired tests in Phase 5 "
        "(UPGRADE_PLAN.md M0.5)")
    assert "run_artifacts.run_dir" in source or "run_dir(" in source
```

Run: `python -m pytest tests/test_protocol.py::test_runner_persists_per_example_predictions -v`
Expected: FAIL on the `save_predictions` assertion.

Then in `experiments/run_experiments.py`, extend the import to:

```python
from QCNN.utils import run_artifacts
from QCNN.utils.run_artifacts import TestEvaluationGuard
```

Change `run_qcnn` to accept the run directory and the test sample IDs, and to record status and predictions:

```python
def run_qcnn(cfg: QuantumNativeConfig, split, use_bce: bool, log_path: str,
             directory: str, test_sample_ids) -> dict:
    """Train the QCNN on a prepared split and return its metric dict."""
    X_train, y_train, X_val, y_val, X_test, y_test = split
    model = PureQuantumNativeCNN(cfg)
    n_params = int(sum(np.prod(p.shape) for p in model.quantum_params.values()))
    trainer = QuantumNativeTrainer(learning_rate=cfg.learning_rate, use_bce=use_bce)

    run_artifacts.start_run(
        directory,
        config={k: v for k, v in vars(cfg).items() if isinstance(v, (int, float, str, bool, type(None)))},
        split_id=cfg.split_id,
        seed=cfg.seed,
        environment={"python": platform.python_version(), "pennylane": qml.version()},
    )

    # Trainer is very chatty; capture its output to a per-run log file.
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    guard = TestEvaluationGuard()
    try:
        with open(log_path, "w") as fh, contextlib.redirect_stdout(fh):
            trained = trainer.train_pure_quantum_cnn(
                model, X_train, y_train, X_val, y_val,
                log_filepath=log_path, summary_filepath=None,
                weights_path=run_artifacts.weights_path(directory),
            )
            raw = guard.evaluate(predict_raw_outputs, trained, X_test, already_preprocessed=False)
    except Exception as exc:
        run_artifacts.fail_run(directory, error=repr(exc))
        raise

    run_artifacts.save_predictions(directory, test_sample_ids, y_test, raw)
    metrics = compute_classification_metrics(y_test, raw)
    metrics["n_params"] = n_params
    run_artifacts.complete_run(directory, metrics={
        k: v for k, v in metrics.items() if isinstance(v, (int, float))})
    return metrics
```

Add `import platform` and `import pennylane as qml` to the module imports.

`cfg.split_id` and the manifest are already produced by the `prepare_split` rewrite in Step 5. Update `run_single` to unpack the pair and build the isolated directory:

```python
    split, manifest = prepare_split(cfg, classes, dataset_dir, train_sample_size)

    run_dir = os.path.join(out_dir, config_name)
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, f"seed_{seed}.log")

    # Clean-protocol artifacts live under Results/runs/, kept separate from the
    # historical Results/experiments/ tree produced under the leaked protocol.
    directory = run_artifacts.run_dir(_fmt_pair(classes), config_name, seed)
    test_ids = [manifest["sample_ids"][i] for i in manifest["test_idx"]]

    metrics = run_qcnn(cfg, split, use_bce, log_path, directory, test_ids)
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `python -m pytest tests/test_protocol.py -v`
Expected: `11 passed`

- [ ] **Step 8: Smoke-run the rewired pipeline end to end**

Run: `python -m experiments.run_experiments --quick`
Expected: completes without traceback; `Results/manifests/` now contains at least one `*_seed*.json` file.

Verify the manifest is well-formed:

Run: `python -c "import glob, json; from QCNN.utils import splits; p=sorted(glob.glob('Results/manifests/*.json'))[0]; m=splits.load_manifest(p); print(p, m['id'][:12], len(m['train_idx']), len(m['val_idx']), len(m['test_idx']))"`
Expected: a line showing the three split sizes in roughly 60/15/25 proportion.

Confirm the isolated run artifacts appeared:

Run: `python -c "import glob; print(sorted(glob.glob('Results/runs/*/*/seed_*/status.json'))[:3])"`
Expected: at least one `status.json` path.

- [ ] **Step 9: Commit**

```bash
git add QCNN/utils/seeding.py main.py experiments/run_experiments.py tests/test_protocol.py Results/manifests
git commit -m "Fix: route training and experiments through recorded split manifests"
```

---

## Task 10: Single source of truth for the circuit

Extracts the frozen topology into one builder and deletes the hand-copied duplicate in `noise_sim.py` (F7). The Task 4 expectation fixtures are the proof this is behaviour-preserving.

**Files:**
- Create: `QCNN/circuits.py`
- Modify: `QCNN/models/QCNNModel.py`
- Modify: `noise_sim.py`
- Create: `tests/test_circuit_source.py`

**Interfaces:**
- Consumes: `freeze.circuit_signature`, `freeze.headline_expectations`, the committed fixtures.
- Produces:
  - `QCNN.circuits.build_circuit(x, params, cfg, hooks=None) -> qml.measurements.ExpectationMP`
    where `params` is the **unflattened dict** (`{'quantum_conv_kernel_0': ..., ...}`) and `cfg` is a `QuantumNativeConfig`.
  - `QCNN.circuits.CircuitHooks` — evaluation-only injection points, all optional and all `None` by default:
    `after_encoding(wires)`, `after_conv_window(wires)`, `after_pool(keep, discard)`, `after_classifier(wires)`, `cnot(a, b)`, `before_readout(readout)`.

  With `hooks=None` the builder must be **exactly** the current `_pure_quantum_forward`. Hooks never enter the headline path (§A3).

- [ ] **Step 1: Write the failing consolidation tests**

Create `tests/test_circuit_source.py`:

```python
"""One circuit definition, consumed by every path (UPGRADE_PLAN.md 0.4, F7).

noise_sim.py previously re-implemented the forward pass by hand, so the noise
study could silently drift away from the model it claimed to measure.
"""
import inspect
import json

import numpy as np
import pytest

from QCNN import circuits, freeze


def test_model_delegates_to_the_shared_builder():
    from QCNN.models.QCNNModel import PureQuantumNativeCNN
    source = inspect.getsource(PureQuantumNativeCNN._pure_quantum_forward)
    assert "circuits.build_circuit" in source or "build_circuit(" in source


def test_noise_sim_has_no_duplicate_topology():
    with open("noise_sim.py", encoding="utf-8") as fh:
        source = fh.read()
    assert "build_circuit" in source, "noise_sim.py must consume QCNN/circuits.py"
    for copied in ("get_conv_windows", "make_pairing", "quantum_unitary_pooling"):
        assert copied not in source, (
            "noise_sim.py still rebuilds the topology by hand ({})".format(copied))


def test_hooks_are_off_by_default():
    hooks = circuits.CircuitHooks()
    assert hooks.after_encoding is None
    assert hooks.after_conv_window is None
    assert hooks.after_pool is None
    assert hooks.after_classifier is None
    assert hooks.before_readout is None


def test_signature_survives_the_consolidation(headline_model):
    with open(freeze.SIGNATURE_FIXTURE) as fh:
        reference = json.load(fh)
    assert freeze.signature_hash(freeze.circuit_signature(headline_model)) == reference["hash"]


@pytest.mark.slow
def test_expectations_survive_the_consolidation(headline_model, archived_params, regression_inputs):
    reference = np.load(freeze.EXPECTATION_FIXTURE)
    actual = freeze.headline_expectations(headline_model, archived_params, regression_inputs)
    np.testing.assert_allclose(
        actual, reference["expectations"], atol=freeze.REGRESSION_TOL, rtol=0.0)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_circuit_source.py -v`
Expected: collection error — `ModuleNotFoundError: No module named 'QCNN.circuits'`

- [ ] **Step 3: Implement the canonical builder**

Create `QCNN/circuits.py`. The body is a verbatim transcription of `QCNNModel._pure_quantum_forward` with hook call sites added; do not "clean up" anything while moving it.

```python
"""The canonical frozen FQCNN circuit (UPGRADE_PLAN.md 0.4).

This is the single topology consumed by training, inference, noise simulation,
hardware transpilation, drawing, and resource counting. It is frozen under A1:
no gate may be added, removed, reordered, or reparameterised here.

``CircuitHooks`` are evaluation-layer injection points used by noise and probe
studies. With ``hooks=None`` — always the case for the headline model — the
emitted operations are exactly the frozen circuit, which is what the committed
expectation fixtures verify.
"""
import math

import pennylane as qml

from QCNN.encoding import PureQuantumEncoder
from QCNN.layers import QuantumNativeConvolution
from QCNN.layers import QuantumNativePooling


class CircuitHooks:
    """Optional evaluation-only callbacks. Never part of the model definition."""

    def __init__(self, after_encoding=None, after_conv_window=None, after_pool=None,
                 after_classifier=None, cnot=None, before_readout=None):
        self.after_encoding = after_encoding
        self.after_conv_window = after_conv_window
        self.after_pool = after_pool
        self.after_classifier = after_classifier
        self.cnot = cnot
        self.before_readout = before_readout


_NO_HOOKS = CircuitHooks()


def build_circuit(x, params, cfg, hooks=None):
    """Emit the frozen circuit and return its terminal expectation value.

    Args:
        x: encoded input (amplitude vector, or raw features for feature maps).
        params: unflattened parameter dict keyed as in ``QCNNModel.quantum_params``.
        cfg: a ``QuantumNativeConfig``.
        hooks: optional ``CircuitHooks``; ``None`` gives the frozen path.
    """
    hooks = hooks or _NO_HOOKS
    emit_cnot = hooks.cnot or (lambda a, b: qml.CNOT(wires=[a, b]))

    all_qubits = list(range(cfg.n_qubits))

    if cfg.encoding_type in ('amplitude', 'patch'):
        PureQuantumEncoder.amplitude_encoding(x, all_qubits)
    else:
        PureQuantumEncoder.quantum_feature_map(x, all_qubits)
    if hooks.after_encoding:
        hooks.after_encoding(all_qubits)

    active_qubits = all_qubits.copy()
    current_image_size = cfg.image_size

    for layer in range(cfg.n_conv_layers):
        n_current = len(active_qubits)
        if n_current >= 4:
            width = int(math.sqrt(n_current))
            while n_current % width != 0:
                width -= 1
            height = n_current // width
            w, h = max(width, height), min(width, height)
            base_windows = QuantumNativeConvolution.get_conv_windows(w, h)
            kernel = params[f'quantum_conv_kernel_{layer}']
            rotations = getattr(cfg, 'kernel_rotations', 'su2')
            entanglement = getattr(cfg, 'conv_entanglement', 'full')
            for rel_window in base_windows:
                if max(rel_window) < n_current:
                    window_qubits = [active_qubits[i] for i in rel_window]
                    QuantumNativeConvolution.quantum_conv2d_kernel(
                        kernel, window_qubits,
                        rotations=rotations, entanglement=entanglement)
                    if hooks.after_conv_window:
                        hooks.after_conv_window(window_qubits)

        if layer < cfg.n_conv_layers - 1:
            n_qubits_current = len(active_qubits)
            if n_qubits_current < 2:
                break
            pairs = QuantumNativePooling.make_pairing(active_qubits)
            if len(pairs) == 0:
                break
            keep = [k for (k, _) in pairs]
            discard = [d for (_, d) in pairs]
            pool_key = f'quantum_pooling_{layer}'
            QuantumNativePooling.apply_pooling(
                getattr(cfg, 'pooling_mode', 'unitary'),
                params[pool_key],
                input_qubits=keep,
                output_qubits=discard
            )
            if hooks.after_pool:
                hooks.after_pool(keep, discard)
            active_qubits = keep
            if current_image_size > 2:
                current_image_size = max(2, current_image_size // 2)

    classifier_params = params['quantum_classifier']
    n_active = len(active_qubits)
    readout = active_qubits[0]

    for i, q in enumerate(active_qubits[:min(n_active, 4)]):
        qml.RX(classifier_params[i * 2 % 32], wires=q)
        qml.RY(classifier_params[(i * 2 + 1) % 32], wires=q)
        qml.RZ(classifier_params[(i * 2 + 8) % 32], wires=q)

    for i in range(n_active - 1):
        emit_cnot(active_qubits[i], active_qubits[i + 1])
    if n_active >= 2:
        emit_cnot(active_qubits[n_active - 1], active_qubits[0])

    for i, q in enumerate(active_qubits[:min(n_active, 4)]):
        qml.RX(classifier_params[(i * 2 + 16) % 32], wires=q)
        qml.RY(classifier_params[(i * 2 + 17) % 32], wires=q)

    if n_active >= 2:
        emit_cnot(active_qubits[0], active_qubits[min(n_active - 1, 1)])
    qml.RZ(classifier_params[31], wires=readout)

    if hooks.after_classifier:
        hooks.after_classifier(active_qubits)
    if hooks.before_readout:
        hooks.before_readout(readout)

    return qml.expval(qml.PauliZ(readout))
```

- [ ] **Step 4: Delegate from the model**

In `QCNN/models/QCNNModel.py`, add to the imports:

```python
from QCNN import circuits
```

and replace the entire body of `_pure_quantum_forward` (lines 107–179) with:

```python
    def _pure_quantum_forward(self, x: np.ndarray, params: dict) -> float:
        return circuits.build_circuit(x, params, self.config)
```

Delete the now-unused `import math` at the top of `QCNNModel.py` only if no other method uses it (check with `grep -n "math\." QCNN/models/QCNNModel.py`).

- [ ] **Step 5: Verify the consolidation preserved behaviour**

This is the gate that makes the refactor legitimate under §A2.

Run: `python -m pytest tests/test_freeze_architecture.py tests/test_freeze_expectations.py -v`
Expected: `6 passed`

If the expectation test fails, the transcription is not verbatim — diff `QCNN/circuits.py` against `git show HEAD~1:QCNN/models/QCNNModel.py` rather than regenerating the fixture.

- [ ] **Step 6: Rewire `noise_sim.py`**

Replace `_make_noisy_circuit` (lines 245–328) with a hook-driven version that reuses the shared builder. The noise placement is unchanged from the current implementation: one-qubit noise after encoding, after each convolution window, on keep+discard after pooling, and on the active set after the classifier; two-qubit noise on the classifier's CNOTs; readout bit-flip on the readout wire.

```python
def _make_noisy_circuit(n_qubits, n_conv, n_pool, flat_params, spec, cfg):
    from QCNN import circuits

    dev = qml.device('default.mixed', wires=n_qubits)

    # Hoisted out of the QNode: identical on every execution.
    params = _unflatten_params(flat_params, n_conv, n_pool, n_qubits)
    all_qubits = list(range(n_qubits))

    def noisy_cnot(a, b):
        qml.CNOT(wires=[a, b])
        _apply_2q_noise(a, b, spec)

    hooks = circuits.CircuitHooks(
        after_encoding=lambda wires: _apply_1q_noise(wires, spec),
        after_conv_window=lambda wires: _apply_1q_noise(wires, spec),
        after_pool=lambda keep, discard: _apply_1q_noise(keep + discard, spec),
        after_classifier=lambda wires: _apply_1q_noise(wires, spec),
        cnot=noisy_cnot,
        before_readout=lambda readout: (
            qml.BitFlip(spec['readout'], wires=readout) if spec.get('readout', 0) > 0 else None),
    )

    @qml.qnode(dev, interface='numpy')
    def circuit(x):
        # ``x`` may be a single amplitude vector or a batch; AmplitudeEmbedding
        # broadcasts over the leading batch dim.
        return circuits.build_circuit(x, params, cfg, hooks=hooks)

    return circuit
```

> The hook ordering differs from the old hand-copied version in one respect: the classifier's post-noise and the readout bit-flip now fire through `after_classifier` / `before_readout` *after* the final `RZ`, matching the original sequence. Confirm by reading the original in `git show HEAD~1:noise_sim.py`.

Update `_make_noisy_circuit`'s call site to pass `cfg`. Find it with:

Run: `grep -n "_make_noisy_circuit(" noise_sim.py`

and add the config the run function already builds as the new final argument.

Also replace `noise_sim.py`'s local `_seed_everything`, if present, with `from QCNN.utils.seeding import seed_everything`.

- [ ] **Step 7: Run the full suite**

Run: `python -m pytest tests/ -v`
Expected: all tests pass, including `tests/test_circuit_source.py` (5 passed).

- [ ] **Step 8: Smoke-run the noise simulator**

Run: `python noise_sim.py --noise-model depolarizing --classes 0 1 --max-test 8`
Expected: completes and writes a figure under `Results/Graphs/`. If it reports missing trained weights, that is the archived-weights path issue, not a topology error — check the message before treating it as a failure.

- [ ] **Step 9: Commit**

```bash
git add QCNN/circuits.py QCNN/models/QCNNModel.py noise_sim.py tests/test_circuit_source.py
git commit -m "Refactor: consolidate the frozen circuit into a single builder"
```

---

## Task 11: Paper↔code reconciliation table

The M0.7 deliverable. Every equation and architecture statement gets a recorded disposition. This document drives the Phase 10 rewrite; no `.tex` edits happen here.

**Files:**
- Create: `docs/paper_code_reconciliation.md`

**Interfaces:**
- Consumes: `tests/fixtures/effective_params.json` (measured counts), `tests/fixtures/headline_signature.json` (executed topology).
- Produces: the reconciliation table referenced by Phase 10 (M10.3).

- [ ] **Step 1: Read the measured counts**

Run: `python -c "import json; a=json.load(open('tests/fixtures/effective_params.json')); print('allocated', a['n_allocated']); print('syntactically used', a['n_syntactically_used']); print('effective', a['n_effective']); print({k: v['n_effective'] for k, v in a['per_group'].items()})"`

Record the output; it fills the bracketed slots below.

- [ ] **Step 2: Write the reconciliation table**

Create `docs/paper_code_reconciliation.md`. Replace each `<measured:...>` marker with the value printed in Step 1 — nothing else in the document is a placeholder.

```markdown
# Paper ↔ Code Reconciliation

**Governing rule:** `UPGRADE_PLAN.md` §A1 — where `fqcnn.tex` and the executed code
disagree, the paper moves to match the code. This table records the disposition of
every mismatch found in Phase 0. Phase 10 (M10.3) applies these dispositions to the
manuscript; no `.tex` edit is made here.

**Evidence sources:** `tests/fixtures/headline_signature.json` (executed topology),
`tests/fixtures/effective_params.json` (gradient audit),
`tests/fixtures/headline_expectations.npz` (frozen behaviour).

| # | Paper says | Code does | Disposition |
|---|---|---|---|
| 1 | Sec. III-B / VI-D: two-stage encoding (`Ry Rz H` then ring `CNOT·Rz·CNOT`), listed as contribution #4 | `AmplitudeEmbedding` only; the two-stage map is never executed on the amplitude path | **Delete** the two-stage map from the main-model description and **remove contribution #4**. Optionally re-scope as the `enc_augmented` ablation arm in Phase 6. |
| 2 | Sec. III-C: a four-level convolutional hierarchy | One effective convolution stage at n=10; layers 1–3 never apply their kernels (5×1 grid yields no windows) | **Rewrite** as one convolution stage plus a two-stage pooling cascade. Scope "hierarchical" to the pooling cascade. The deeper hierarchy becomes the n=16 scaling-family instance in Phase 8. |
| 3 | Sec. III-D pooling `CRY·CRZ·RY`; Sec. VI-B pooling `CRZ·CRY·CRZ` | `CRY(a)` then `CRZ(b)` from discard→keep, `RY(0.02)` on discard, `RY(c)` on keep | **One canonical definition** matching the code, stated once. The paper currently gives two conflicting definitions of its own contribution. |
| 4 | Implied: `RY(0.02)` on the discarded wire is a harmless disentangling touch | Changing it to `RY(0.03)` moves `⟨Z⟩` beyond 1e-10 (verified in Task 4 Step 7) — the discarded wire is re-entangled by *later* pooling stages through the keep/discard pairing | **Document as an operative gate**, not an inert one. The §A2 inert-gate removal exception does **not** apply. Disclose the constant and its effect. |
| 5 | Sec. III-E: classifier with `n_r + 1` parameters | 32-slot RX/RY/RZ block with modular index reuse (`% 32`), plus a ring of CNOTs and a final `RZ` on the readout wire | **Describe the executed block honestly**, including the modular reuse, with the effective counts from row 7. |
| 6 | F2: pooling pairs all wires | `make_pairing` drops the last wire of an odd active set; at 5 active qubits one wire is retired with no information-transfer gate | **Disclose** in the pooling definition: an unpaired wire is retired by exclusion from subsequent operations. On a still-pure global state this is deferred bookkeeping, not an extra operation. |
| 7 | "269 trainable parameters" | 269 allocated slots; `<measured:n_syntactically_used>` appear as gate arguments; `<measured:n_effective>` have a nonzero gradient on at least one fixed input | **Report both numbers**: "269 allocated / `<measured:n_effective>` effective", with one sentence of explanation and a pointer to the committed audit. The parameter *vector* keeps 269 slots so the archived run's seeded RNG stream reproduces. |
| 8 | Sec. IV-C: gradients via parameter-shift | Simulation uses adjoint differentiation (`diff_method='best'` on `lightning.qubit`) | **State both**: adjoint differentiation in simulation; parameter-shift as the hardware-execution path, with its `2·\|Θ_eff\|` circuit cost quantified in Phase 8. |
| 9 | Sec. V: 11,430 / 8,001 / 3,429 samples | `training_log.txt`: 12,665 / 8,865 / 3,800, under a 70/30 split with no validation set | **Regenerate** all dataset counts from the executed pipeline under the new 60/15/25 manifest protocol (Phase 1 clean retrain). |
| 10 | "Validation accuracy 98.7%" (Abstract, Sec. V) | No validation split existed; 98.86% was a best-of-24-epochs **test** score used for checkpointing, LR plateau, and early stopping | **Replace** with the clean-protocol distribution from the Phase 1 retrain across seeds. Report the lower number if it is lower (§A6). |
| 11 | Sec. III / F5: "2×2 spatial window", "translational equivariance over the image" | Under global amplitude encoding a qubit indexes a bit-plane of the flattened pixel address, not a pixel neighbourhood | **Reframe as locality in register/index space** (bounded entanglement spread, constant per-window depth, weight sharing over windows). Report the measured translation sensitivity from Phase 3.6 instead of asserting an inductive bias the model does not have. |
| 12 | Abstract, Secs. I, VI-A: unitary pooling "preserves information that measurement pooling irrevocably loses" | By deferred measurement the retained register's state — and every downstream observable — is identical to measure-and-condition pooling | **Delete the information-loss claim.** Replace with the Phase 2 simulation theorem: exact reproduction of measurement-pooling statistics with no mid-circuit measurement, no reset, and no classical feed-forward, plus end-to-end adjoint differentiability. |
| 13 | Sec. VI: calibration claims based on min-max normalised scores | Scores are min-max normalised over the test set | **Remove** the unsupported calibration claim. Phase 5.4 defines defensible probability outputs and reports ECE. |
| 14 | Reproducibility paragraph: implied environment | `requirements-lock.txt` claimed Python 3.14.5 / PennyLane 0.44.0 / NumPy 2.4.2 / scikit-learn 1.8.0; the results were produced on Python 3.9.13 / PennyLane 0.38.0 / NumPy 1.26.4 / scikit-learn 1.6.1 | **Corrected in Phase 0 Task 12.** State the true stack, including the simulator, in the reproducibility paragraph. |

## Unitarity statement (verified)

`tests/test_freeze_architecture.py::test_main_path_contains_no_non_unitary_operation`
confirms the executed headline tape contains no mid-circuit measurement, no classical
feed-forward, and no non-unitary channel, and terminates in exactly one expectation
value. The measurement-pooling arm is detected by the same audit as a positive
control, so the guard cannot pass vacuously. These are the defensible claims that
replace row 12's deleted assertion.
```

- [ ] **Step 3: Verify no placeholder markers remain**

Run: `grep -n "<measured:" docs/paper_code_reconciliation.md`
Expected: no output.

- [ ] **Step 4: Commit**

```bash
git add docs/paper_code_reconciliation.md
git commit -m "Docs: record paper-code reconciliation dispositions"
```

---

## Task 12: Reproducibility repairs and the M0 gate

Closes M0.8 and proves the whole milestone.

**Files:**
- Modify: `requirements-lock.txt`
- Modify: `requirements.txt`
- Modify: `setup_env.ps1`
- Modify: `setup_env.bat`
- Modify: `reproduce.sh`
- Modify: `QCNN/models/QCNNModel.py`
- Delete: `Results/Weights/quantum_model_params(old).npz`, `output.txt`

**Interfaces:**
- Consumes: the full test suite from Tasks 1–11.
- Produces: a `reproduce.sh` whose first action is the freeze/protocol gate.

- [ ] **Step 1: Regenerate the environment lock from the live environment**

Run: `python -m pip freeze > /tmp/live_freeze.txt && grep -iE "^(pennylane|numpy|scikit-learn|matplotlib|seaborn|scipy|autoray|pandas|pillow|pytest)" /tmp/live_freeze.txt`
Expected: the true installed versions, including `PennyLane==0.38.0` and `numpy==1.26.4`.

Rewrite `requirements-lock.txt` using those exact versions:

```
# Exact environment used to produce the reported FQCNN results.
# Regenerated from the live training environment (UPGRADE_PLAN.md 0.6 / F9);
# the previous contents claimed a Python 3.14 / PennyLane 0.44 stack that never
# produced any result in this repository.
# Reproduce with:  pip install -r requirements-lock.txt
# Python 3.9.13
PennyLane==0.38.0
PennyLane-Lightning==0.38.0
numpy==1.26.4
scikit-learn==1.6.1
matplotlib==3.9.4
pytest==7.4.4
```

Then append every remaining line printed by the `grep` above (seaborn, scipy, autoray, pandas, Pillow) with its **actual** installed version. Do not carry over a version that `pip freeze` did not print.

- [ ] **Step 2: Align the floors in `requirements.txt`**

Change the first two pins so the floor cannot pull in a simulator that never produced these results:

```
pennylane >= 0.38.0, < 0.39
pennylane-lightning >= 0.38.0, < 0.39
numpy >= 1.23.0, < 2.0
```

Leave the remaining lines and the optional-extras comment unchanged.

- [ ] **Step 3: Align the setup scripts**

In both `setup_env.ps1` and `setup_env.bat`, ensure the install step uses the lock file. Read each file first, then change the `pip install -r requirements.txt` invocation to:

```
pip install -r requirements-lock.txt
```

If a script hard-codes a Python version string, change it to `3.9`.

- [ ] **Step 4: Delete the permitted dead artifacts**

`UPGRADE_PLAN.md` 0.6 permits exactly these three, all verified unreferenced:

Run: `git rm "Results/Weights/quantum_model_params(old).npz" output.txt`
Expected: two `rm` lines.

Then delete the unused MSE duplicate `quantum_loss_function` from `QCNN/models/QCNNModel.py` (lines 229–238 in the pre-Task-10 file). Confirm it is unreferenced first:

Run: `grep -rn "quantum_loss_function" --include=*.py .`
Expected: only the definition itself. If anything else appears, stop and leave the method in place.

- [ ] **Step 5: Make `reproduce.sh` run the gate first**

Replace the interpreter default and insert the gate as step 0. Change:

```bash
PY="${PYTHON:-.venv/bin/python}"
```

to:

```bash
# No .venv is committed; default to whatever python is on PATH and let the
# integrity gate below fail loudly if it is the wrong interpreter.
PY="${PYTHON:-python}"
```

and insert immediately after the `"$PY" --version` line:

```bash
echo "=== [0/4] Freeze and protocol integrity gate ==="
# Nothing downstream is trustworthy if these fail (UPGRADE_PLAN.md 0.1, 0.6).
"$PY" -m pytest tests/ -q

```

Then renumber the existing step banners from `[1/3]`, `[2/3]`, `[3/3]` to `[1/4]`, `[2/4]`, `[3/4]`.

- [ ] **Step 6: Run the complete suite**

Run: `python -m pytest tests/ -v`
Expected: all tests pass. Record the count.

- [ ] **Step 7: Run the reproduction gate end to end**

Run: `bash reproduce.sh --quick`
Expected: the `[0/4]` gate passes before any experiment runs; the remaining quick steps complete.

- [ ] **Step 8: Verify the M0 exit criteria**

Confirm each roadmap M0 exit check, and record the result of each command:

```bash
# M0.2 — freeze guards catch drift (proven in Task 2 Step 7 and Task 4 Step 7)
python -m pytest tests/test_freeze_architecture.py tests/test_freeze_expectations.py -q

# M0.3 — one authoritative effective-parameter artifact
python -c "import json; a=json.load(open('tests/fixtures/effective_params.json')); print('allocated', a['n_allocated'], 'effective', a['n_effective'])"

# M0.4 — split disjointness and single test evaluation
python -m pytest tests/test_splits.py tests/test_protocol.py -q

# M0.5 — two runs cannot collide, archived weights are never a write target
python -m pytest tests/test_run_artifacts.py -q

# M0.6 — one circuit source
python -m pytest tests/test_circuit_source.py -q

# M0.8 — the lock matches the live environment
python -c "import pennylane, numpy, sklearn; print(pennylane.__version__, numpy.__version__, sklearn.__version__)"
grep -E "^(PennyLane|numpy|scikit-learn)==" requirements-lock.txt
```

Expected: every pytest command reports `passed`; the printed versions match the lock exactly.

- [ ] **Step 9: Commit**

```bash
git add requirements.txt requirements-lock.txt setup_env.ps1 setup_env.bat reproduce.sh QCNN/models/QCNNModel.py
git commit -m "Fix: truthful environment lock and gated reproduction entry point"
```

- [ ] **Step 10: Tag the milestone gate**

```bash
git tag -a phase-0-gate -m "M0 gate: freeze guards, clean split protocol, single circuit source, truthful environment"
```

---

## What Phase 0 deliberately does not do

Recorded so the next session does not mistake these for oversights:

- **No retraining.** The clean-protocol headline retrain is Phase 1 work (M1 gate).
- **No circuit edits.** Including the `RY(0.02)` removal — Task 4 Step 7 shows the gate is *not* inert, so the §A2 exception does not apply.
- **No baseline protocol fix.** The classical baselines still select on the data passed as their test set; that is M5.1, flagged in the code at its call site.
- **No `.tex` edits.** Task 11 records dispositions; Phase 10 applies them.
- **No batching, caching, parallelism, or cost estimation.** All Phase 1.
- **No pooling arms beyond what already exists.** `pool_measurement`'s correction to the theorem's channel is M2.1.
