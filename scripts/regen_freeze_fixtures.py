"""Regenerate the committed freeze fixtures from the archived headline model.

Run this ONLY when a change to the frozen circuit has been approved under
UPGRADE_PLAN.md A2. Regenerating fixtures to make a failing guard pass defeats
the entire purpose of the freeze.

Usage:  python scripts/regen_freeze_fixtures.py
"""
import json
import os
import sys

import numpy as np

# Running this by file path puts scripts/ on sys.path, not the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from QCNN import freeze  # noqa: E402


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
