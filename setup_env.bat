@echo off
REM ===============================
REM QCNN Project Environment Setup (Windows Batch)
REM ===============================
REM This script sets up a Python environment for running the Quantum Convolutional Neural Network (QCNN) project.
REM Run this script from your project's root directory.

echo === QCNN Setup Script Starting ===

REM Recommended: Create a fresh virtual environment
echo Making .venv folder if it doesn't exist
if not exist .venv (
    python -m venv .venv
) else (
    echo .venv already exists.
)

REM Activate the virtual environment
echo Activating .venv...
call .venv\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM --- PINNED DEPENDENCIES ---
REM Install from the lock, not from unpinned latest: the reported results were
REM produced on PennyLane 0.38 / NumPy 1.26, and installing whatever is current
REM gives a different simulator (UPGRADE_PLAN.md 0.6 / F9).
echo Installing pinned dependencies from requirements-lock.txt...
pip install -r requirements-lock.txt

REM --- VERIFY INSTALLATION ---
echo --- Installed PennyLane devices ---
python -c "import pennylane as qml; print(qml.list_available_devices())"

echo --- Verifying the pinned stack ---
python -c "import platform, pennylane, numpy, sklearn; print('python', platform.python_version()); print('pennylane', pennylane.__version__); print('numpy', numpy.__version__); print('scikit-learn', sklearn.__version__)"

echo === QCNN Setup Complete! ===
echo To activate the environment next time, run: .venv\Scripts\activate
pause
