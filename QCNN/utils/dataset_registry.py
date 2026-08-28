"""Thin, checksummed dataset registry for the Q1 fast-track study.

Only MNIST, Fashion-MNIST, and Kuzushiji-MNIST are in scope. The registry keeps
source identity separate from the binary class task and preserves the original
source partition in every sample ID.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import gzip
import hashlib
import json
from pathlib import Path
import struct
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.request import urlopen

import numpy as np

from .data_preprocessing import preprocess_for_quantum


@dataclass(frozen=True)
class SourceFile:
    partition: str
    kind: str
    filename: str
    url: Optional[str]
    md5: Optional[str]
    compressed: bool = True
    sha256: Optional[str] = None


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    display_name: str
    version: str
    directory: str
    homepage: str
    license: str
    files: Tuple[SourceFile, ...]
    labels: Tuple[str, ...]


_IDX_NAMES = {
    ("train", "images"): "train-images-idx3-ubyte",
    ("train", "labels"): "train-labels-idx1-ubyte",
    ("test", "images"): "t10k-images-idx3-ubyte",
    ("test", "labels"): "t10k-labels-idx1-ubyte",
}


def _remote_files(
    base_url: str,
    checksums: Dict[str, str],
    sha256s: Optional[Dict[str, str]] = None,
) -> Tuple[SourceFile, ...]:
    sha256s = sha256s or {}
    return tuple(
        SourceFile(
            partition=partition,
            kind=kind,
            filename=f"{filename}.gz",
            url=f"{base_url}/{filename}.gz",
            md5=checksums[f"{filename}.gz"],
            sha256=sha256s.get(f"{filename}.gz"),
        )
        for (partition, kind), filename in _IDX_NAMES.items()
    )


DATASETS: Dict[str, DatasetSpec] = {
    "mnist": DatasetSpec(
        key="mnist",
        display_name="MNIST",
        version="official IDX snapshot",
        directory="MNIST",
        homepage="https://yann.lecun.com/exdb/mnist/",
        license="source terms and citation apply",
        files=tuple(
            SourceFile(
                partition, kind, filename.replace("-idx", ".idx"),
                None, None, compressed=False,
            )
            for (partition, kind), filename in _IDX_NAMES.items()
        ),
        labels=tuple(str(value) for value in range(10)),
    ),
    "fashion_mnist": DatasetSpec(
        key="fashion_mnist",
        display_name="Fashion-MNIST",
        version="Zalando Research official IDX release",
        directory="FashionMNIST",
        homepage="https://github.com/zalandoresearch/fashion-mnist",
        license="MIT repository license; dataset citation required",
        files=_remote_files(
            # Use the TLS-enabled S3 endpoint; the website endpoint is HTTP-only.
            "https://fashion-mnist.s3.eu-central-1.amazonaws.com",
            {
                "train-images-idx3-ubyte.gz": "8d4fb7e6c68d591d4c3dfef9ec88bf0d",
                "train-labels-idx1-ubyte.gz": "25c81989df183df01b3e8a0aad5dffbe",
                "t10k-images-idx3-ubyte.gz": "bef4ecab320f06d8554ea6380940ec79",
                "t10k-labels-idx1-ubyte.gz": "bb300cfdad3c16e7a12a480ee83cd310",
            },
            {
                "train-images-idx3-ubyte.gz": "3aede38d61863908ad78613f6a32ed271626dd12800ba2636569512369268a84",
                "train-labels-idx1-ubyte.gz": "a04f17134ac03560a47e3764e11b92fc97de4d1bfaf8ba1a3aa29af54cc90845",
                "t10k-images-idx3-ubyte.gz": "346e55b948d973a97e58d2351dde16a484bd415d4595297633bb08f03db6a073",
                "t10k-labels-idx1-ubyte.gz": "67da17c76eaffca5446c3361aaab5c3cd6d1c2608764d35dfb1850b086bf8dd5",
            },
        ),
        labels=(
            "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
        ),
    ),
    "kmnist": DatasetSpec(
        key="kmnist",
        display_name="Kuzushiji-MNIST",
        version="ROIS-CODH official KMNIST IDX release",
        directory="KMNIST",
        homepage="https://codh.rois.ac.jp/kmnist/",
        license="CC BY-SA 4.0",
        files=_remote_files(
            "https://codh.rois.ac.jp/kmnist/dataset/kmnist",
            {
                "train-images-idx3-ubyte.gz": "bdb82020997e1d708af4cf47b453dcf7",
                "train-labels-idx1-ubyte.gz": "e144d726b3acfaa3e44228e80efcd344",
                "t10k-images-idx3-ubyte.gz": "5c965bf0a639b31b8f53240b1b52f4d7",
                "t10k-labels-idx1-ubyte.gz": "7320c461ea6c1c855c0b718fb2a4b134",
            },
            {
                "train-images-idx3-ubyte.gz": "51467d22d8cc72929e2a028a0428f2086b092bb31cfb79c69cc0a90ce135fde4",
                "train-labels-idx1-ubyte.gz": "e38f9ebcd0f3ebcdec7fc8eabdcdaef93bb0df8ea12bee65224341c8183d8e17",
                "t10k-images-idx3-ubyte.gz": "edd7a857845ad6bb1d0ba43fe7e794d164fe2dce499a1694695a792adfac43c5",
                "t10k-labels-idx1-ubyte.gz": "20bb9a0ef54c7db3efc55a92eef5582c109615df22683c380526788f98e42a1c",
            },
        ),
        labels=tuple(f"KMNIST class {value}" for value in range(10)),
    ),
}


Q1_TASKS = (
    ("mnist", (0, 1), "headline continuity / easy calibration"),
    ("mnist", (3, 5), "harder in-domain task"),
    ("fashion_mnist", (0, 6), "T-shirt/top versus Shirt"),
    ("kmnist", (2, 3), "distinct Kuzushiji character classes"),
)


def get_spec(dataset: str) -> DatasetSpec:
    try:
        return DATASETS[dataset]
    except KeyError as exc:
        raise ValueError(
            f"unknown dataset {dataset!r}; choose from {', '.join(sorted(DATASETS))}"
        ) from exc


def dataset_directory(dataset: str, data_root: Path = Path("datasets")) -> Path:
    return Path(data_root) / get_spec(dataset).directory


def _digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_source(dataset: str, data_root: Path = Path("datasets")) -> List[dict]:
    spec = get_spec(dataset)
    directory = dataset_directory(dataset, data_root)
    records = []
    for source in spec.files:
        path = directory / source.filename
        if not path.is_file():
            raise FileNotFoundError(f"missing {dataset} source file: {path}")
        md5 = _digest(path, "md5")
        if source.md5 is not None and md5 != source.md5:
            raise ValueError(
                f"checksum mismatch for {path}: expected {source.md5}, observed {md5}"
            )
        sha256 = _digest(path, "sha256")
        if source.sha256 is not None and sha256 != source.sha256:
            raise ValueError(
                f"SHA-256 mismatch for {path}: expected {source.sha256}, observed {sha256}"
            )
        records.append({
            "partition": source.partition,
            "kind": source.kind,
            "path": path.as_posix(),
            "url": source.url,
            "bytes": path.stat().st_size,
            "md5": md5,
            "sha256": sha256,
            "sha256_expected": source.sha256,
        })
    return records


def fetch_dataset(dataset: str, data_root: Path = Path("datasets")) -> List[dict]:
    spec = get_spec(dataset)
    directory = dataset_directory(dataset, data_root)
    directory.mkdir(parents=True, exist_ok=True)
    for source in spec.files:
        path = directory / source.filename
        if path.exists():
            continue
        if source.url is None:
            raise FileNotFoundError(
                f"{dataset} is local-only in this registry; expected {path}"
            )
        temporary = path.with_suffix(path.suffix + ".part")
        if temporary.exists():
            raise FileExistsError(f"incomplete download already exists: {temporary}")
        try:
            with urlopen(source.url, timeout=120) as response, temporary.open("xb") as output:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
            md5 = _digest(temporary, "md5")
            if source.md5 is not None and md5 != source.md5:
                raise ValueError(
                    f"download checksum mismatch for {source.filename}: "
                    f"expected {source.md5}, observed {md5}"
                )
            sha256 = _digest(temporary, "sha256")
            if source.sha256 is not None and sha256 != source.sha256:
                raise ValueError(
                    f"download SHA-256 mismatch for {source.filename}: "
                    f"expected {source.sha256}, observed {sha256}"
                )
            temporary.replace(path)
        except Exception:
            if temporary.exists():
                temporary.unlink()
            raise
    return validate_source(dataset, data_root)


def _source_file(spec: DatasetSpec, partition: str, kind: str) -> SourceFile:
    matches = [
        source for source in spec.files
        if source.partition == partition and source.kind == kind
    ]
    if len(matches) != 1:
        raise ValueError(f"{spec.key} has no unique {partition}/{kind} source")
    return matches[0]


def _open_binary(path: Path, compressed: bool):
    return gzip.open(path, "rb") if compressed else path.open("rb")


def load_raw(
    dataset: str,
    partition: str = "train",
    data_root: Path = Path("datasets"),
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    if partition not in {"train", "test"}:
        raise ValueError("partition must be 'train' or 'test'")
    spec = get_spec(dataset)
    directory = dataset_directory(dataset, data_root)
    image_source = _source_file(spec, partition, "images")
    label_source = _source_file(spec, partition, "labels")
    image_path = directory / image_source.filename
    label_path = directory / label_source.filename

    with _open_binary(label_path, label_source.compressed) as handle:
        header = handle.read(8)
        if len(header) != 8:
            raise ValueError(f"truncated IDX label header: {label_path}")
        magic, label_count = struct.unpack(">II", header)
        if magic != 2049:
            raise ValueError(f"invalid IDX label magic {magic} in {label_path}")
        labels = np.frombuffer(handle.read(), dtype=np.uint8)

    with _open_binary(image_path, image_source.compressed) as handle:
        header = handle.read(16)
        if len(header) != 16:
            raise ValueError(f"truncated IDX image header: {image_path}")
        magic, image_count, rows, columns = struct.unpack(">IIII", header)
        if magic != 2051 or (rows, columns) != (28, 28):
            raise ValueError(
                f"invalid IDX image header magic={magic} shape={rows}x{columns} in {image_path}"
            )
        images = np.frombuffer(handle.read(), dtype=np.uint8)

    if label_count != image_count or len(labels) != label_count:
        raise ValueError("IDX image/label counts do not agree")
    expected_values = image_count * rows * columns
    if len(images) != expected_values:
        raise ValueError(
            f"IDX image payload has {len(images)} values; expected {expected_values}"
        )
    images = images.reshape(image_count, rows * columns)
    sample_ids = [f"{dataset}:{partition}:{index:05d}" for index in range(image_count)]
    return images, labels, sample_ids


def load_binary(
    dataset: str,
    classes: Sequence[int],
    partition: str = "train",
    data_root: Path = Path("datasets"),
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    if len(classes) != 2 or len(set(classes)) != 2:
        raise ValueError("binary task requires two distinct classes")
    spec = get_spec(dataset)
    if any(type(value) is not int or value < 0 or value >= len(spec.labels) for value in classes):
        raise ValueError(f"invalid classes {classes!r} for {dataset}")
    images, labels, sample_ids = load_raw(dataset, partition, data_root)
    mask = np.isin(labels, np.asarray(classes, dtype=np.uint8))
    indices = np.flatnonzero(mask)
    return images[indices], labels[indices], [sample_ids[index] for index in indices]


def load_binary_quantum(
    dataset: str,
    classes: Sequence[int],
    *,
    partition: str = "train",
    data_root: Path = Path("datasets"),
    n_qubits: int = 10,
    image_size: int = 28,
    normalization: str = "minmax",
    encoding_type: str = "amplitude",
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    images, labels, sample_ids = load_binary(dataset, classes, partition, data_root)
    features, encoded = preprocess_for_quantum(
        images,
        labels,
        n_qubits=n_qubits,
        image_size=image_size,
        normalization=normalization,
        encoding_type=encoding_type,
    )
    return features, encoded, sample_ids


def provenance(dataset: str, data_root: Path = Path("datasets")) -> dict:
    spec = get_spec(dataset)
    return {
        "dataset": spec.key,
        "display_name": spec.display_name,
        "version": spec.version,
        "homepage": spec.homepage,
        "license": spec.license,
        "labels": list(spec.labels),
        "source_files": validate_source(dataset, data_root),
        "sample_id_format": f"{dataset}:<source_partition>:<zero-padded-source-index>",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    fetch = subparsers.add_parser("fetch")
    fetch.add_argument("--dataset", nargs="+", choices=sorted(DATASETS), required=True)
    fetch.add_argument("--data-root", default="datasets")
    report = subparsers.add_parser("provenance")
    report.add_argument("--dataset", nargs="+", choices=sorted(DATASETS), required=True)
    report.add_argument("--data-root", default="datasets")
    report.add_argument("--output", required=True)
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = _parser().parse_args(argv)
    data_root = Path(args.data_root)
    if args.command == "fetch":
        for dataset in args.dataset:
            records = fetch_dataset(dataset, data_root)
            print(f"{dataset}: {len(records)} source files verified")
        return 0
    payload = {
        "schema": {"name": "fqcnn_q1_dataset_provenance", "version": 1},
        "datasets": [provenance(dataset, data_root) for dataset in args.dataset],
        "frozen_tasks": [
            {"dataset": dataset, "classes": list(classes), "role": role}
            for dataset, classes, role in Q1_TASKS
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
