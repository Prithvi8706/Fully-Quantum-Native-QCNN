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
import tempfile
import time

import numpy as np
from sklearn.model_selection import train_test_split

# train / validation / test. Fixed by the upgrade plan; not a tunable.
SPLIT_FRACTIONS = (0.60, 0.15, 0.25)


def manifest_id(manifest: dict) -> str:
    """Stable identity of a split, over its indices and provenance."""
    payload = {
        'dataset_id': manifest['dataset_id'],
        'class_mapping': manifest['class_mapping'],
        'seed': manifest['seed'],
        'fractions': list(manifest['fractions']),
        'sample_ids': manifest['sample_ids'],
        'train_idx': manifest['train_idx'],
        'val_idx': manifest['val_idx'],
        'test_idx': manifest['test_idx'],
    }
    blob = json.dumps(payload, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(blob.encode('utf-8')).hexdigest()


def class_mapping_for(classes) -> dict:
    """The label mapping a manifest must record, matching ``encode_labels``.

    ``QCNN.utils.data_preprocessing.encode_labels`` maps ``sorted(unique(y))[0]``
    to ``-1`` and ``sorted(unique(y))[1]`` to ``+1``. A manifest built from
    command-line argument order instead would name the wrong positive class,
    which silently misattributes precision, recall and F1.
    """
    lo, hi = sorted(classes)
    return {str(lo): -1, str(hi): 1}


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

    train_frac, val_frac, _ = SPLIT_FRACTIONS
    positions = np.arange(n_total)

    # Integer counts, not fractions: 0.15 / (0.15 + 0.25) evaluates to
    # 0.37499999999999994 in binary floating point, which sklearn floors to one
    # sample short of the intended validation size.
    n_train = int(round(train_frac * n_total))
    n_val = int(round(val_frac * n_total))

    train_idx, holdout_idx = train_test_split(
        positions,
        train_size=n_train,
        random_state=seed,
        stratify=labels,
    )
    val_idx, test_idx = train_test_split(
        holdout_idx,
        train_size=n_val,
        random_state=seed,
        stratify=labels[holdout_idx],
    )

    canonical_sample_ids = []
    for sample_id in sample_ids:
        if isinstance(sample_id, bool) or not isinstance(sample_id, (int, str)):
            raise ValueError('sample IDs must be non-empty strings or integers')
        if isinstance(sample_id, str) and not sample_id:
            raise ValueError('sample IDs must be non-empty strings or integers')
        canonical_sample_ids.append(sample_id)
    if len(set(canonical_sample_ids)) != len(canonical_sample_ids):
        raise ValueError('sample IDs must be unique')

    manifest = {
        'dataset_id': dataset_id,
        'class_mapping': class_mapping,
        'seed': int(seed),
        'fractions': list(SPLIT_FRACTIONS),
        'n_total': int(n_total),
        'sample_ids': canonical_sample_ids,
        'train_idx': sorted(int(i) for i in train_idx),
        'val_idx': sorted(int(i) for i in val_idx),
        'test_idx': sorted(int(i) for i in test_idx),
    }
    manifest['id'] = manifest_id(manifest)
    verify_manifest(manifest)
    return manifest


def verify_manifest(manifest: dict) -> None:
    """Raise ``ValueError`` unless the split is disjoint and exhaustive."""
    sample_ids = manifest.get('sample_ids')
    if not isinstance(sample_ids, list) or len(sample_ids) != manifest['n_total']:
        raise ValueError('sample IDs must contain one entry per sample')
    for sample_id in sample_ids:
        if isinstance(sample_id, bool) or not isinstance(sample_id, (int, str)):
            raise ValueError('sample IDs must be non-empty strings or integers')
        if isinstance(sample_id, str) and not sample_id:
            raise ValueError('sample IDs must be non-empty strings or integers')
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError('sample IDs must be unique')

    train = set(manifest['train_idx'])
    val = set(manifest['val_idx'])
    test = set(manifest['test_idx'])

    if len(train) + len(val) + len(test) != len(train | val | test):
        raise ValueError('split partitions are not disjoint')
    if train | val | test != set(range(manifest['n_total'])):
        raise ValueError('split partitions do not cover all {} samples'.format(manifest['n_total']))


def save_manifest(manifest: dict, path: str) -> None:
    """Persist a manifest with an atomic replace.

    The experiment runner prepares the same dataset/seed manifest in several
    worker processes when ``--jobs`` is greater than one.  Writing directly to
    the destination lets one worker observe another worker's truncated JSON.
    A per-directory temporary file followed by ``os.replace`` means readers
    see either the old complete manifest or the new complete manifest, never a
    partially written file.  The temporary file is deliberately created beside
    the destination so the replace remains on one filesystem.
    """
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    directory = os.path.dirname(os.path.abspath(path)) or '.'
    basename = os.path.basename(path)
    temporary_path = None
    try:
        fd, temporary_path = tempfile.mkstemp(
            prefix=f'.{basename}.', suffix='.tmp', dir=directory)
        with os.fdopen(fd, 'w', encoding='utf-8', newline='') as fh:
            json.dump(manifest, fh, indent=2, sort_keys=True)
            fh.write('\n')
            fh.flush()
            os.fsync(fh.fileno())
        # Windows refuses to replace a file while a concurrent reader still
        # holds it open.  Readers never observe a partial file because the
        # replacement remains atomic; retry the short sharing window rather
        # than turning harmless read/write contention into a failed worker.
        for attempt in range(20):
            try:
                os.replace(temporary_path, path)
                break
            except PermissionError:
                if attempt == 19:
                    raise
                time.sleep(0.005)
        temporary_path = None
    finally:
        if temporary_path is not None:
            try:
                os.unlink(temporary_path)
            except FileNotFoundError:
                pass


def load_manifest(path: str) -> dict:
    # A concurrent atomic replacement can briefly make the destination
    # unavailable on Windows while another process still has the old file open.
    # Bound the retry so a genuinely missing/locked manifest remains an
    # actionable error.
    for attempt in range(20):
        try:
            with open(path, encoding='utf-8') as fh:
                manifest = json.load(fh)
            break
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.005)
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
