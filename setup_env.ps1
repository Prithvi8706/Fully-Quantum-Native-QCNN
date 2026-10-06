# ===============================
# QCNN Project Environment Setup (PowerShell)
# ===============================
# This script sets up a Python environment for running the Quantum Convolutional Neural Network (QCNN) project.
# Run this script from your project's root directory.

Write-Host "=== QCNN Setup Script Starting ===" -ForegroundColor Cyan

# Recommended: Create a fresh virtual environment
if (-not (Test-Path ".venv")) {
    Write-Host "Making .venv folder..."
    python -m venv .venv
} else {
    Write-Host ".venv already exists."
}

# Activate the virtual environment
Write-Host "Activating .venv..." -ForegroundColor Yellow
& .\.venv\Scripts\Activate.ps1

# Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# --- PINNED DEPENDENCIES ---
# Install from the lock, not from unpinned latest: the reported results were
# produced on PennyLane 0.38 / NumPy 1.26, and installing whatever is current
# gives a different simulator (UPGRADE_PLAN.md 0.6 / F9).
Write-Host "Installing pinned dependencies from requirements-lock.txt..." -ForegroundColor Yellow
pip install -r requirements-lock.txt

# --- VERIFY INSTALLATION ---
Write-Host "--- Installed PennyLane devices ---" -ForegroundColor Green
python -c "import pennylane as qml; print(qml.list_available_devices())"

Write-Host "--- Verifying the pinned stack ---" -ForegroundColor Green
python -c "import platform, pennylane, numpy, sklearn; print('python', platform.python_version()); print('pennylane', pennylane.__version__); print('numpy', numpy.__version__); print('scikit-learn', sklearn.__version__)"

Write-Host "=== QCNN Setup Complete! ===" -ForegroundColor Cyan
Write-Host "To activate the environment next time, run: .\.venv\Scripts\Activate.ps1"
Pause
