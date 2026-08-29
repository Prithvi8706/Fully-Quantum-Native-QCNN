import hashlib
import json
import zipfile

import pytest

from scripts import build_q1_submission_package as package


def test_referenced_graphics_are_normalized_and_deduplicated():
    tex = r"""
    \includegraphics[width=0.5\linewidth]{figs_final/fig1.png}
    \includegraphics{figs_final/q1_comparison}
    \includegraphics{figs_final/fig1.png}
    """
    assert package.referenced_graphics(tex) == [
        "figs_final/fig1.png",
        "figs_final/q1_comparison.png",
    ]


@pytest.mark.parametrize("value", ["../secret", "/absolute", "C:/secret"])
def test_unsafe_archive_paths_are_rejected(value):
    with pytest.raises(ValueError, match="unsafe archive path"):
        package.safe_member_path(value)


def test_archive_verifier_checks_manifest_and_file_hashes(tmp_path):
    source = tmp_path / "paper.txt"
    source.write_bytes(b"bounded evidence\n")
    records = package.file_records(tmp_path, ["paper.txt"])
    manifest = {
        "integrity": {
            "files": records,
            "file_count": 1,
            "total_bytes": source.stat().st_size,
        }
    }
    output = tmp_path / "package.zip"
    package.write_archive(tmp_path, output, "release", records, manifest)
    package.verify_archive(output, "release", manifest)

    with zipfile.ZipFile(output) as archive:
        archived_manifest = json.loads(
            archive.read(f"release/{package.MANIFEST_NAME}")
        )
    assert archived_manifest == manifest
    assert records[0]["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()


def test_pdf_guard_rejects_non_pdf_content(tmp_path):
    (tmp_path / "fqcnn.tex").write_text("paper", encoding="utf-8")
    (tmp_path / "fqcnn.pdf").write_bytes(b"not a pdf")
    with pytest.raises(ValueError, match="valid PDF header"):
        package.assert_fresh_pdf(tmp_path)


def test_secret_scan_reports_location_without_secret_material(tmp_path):
    key = "AK" + "IA" + "A" * 16
    (tmp_path / "safe.md").write_text("no credentials", encoding="utf-8")
    (tmp_path / "unsafe.txt").write_text(key, encoding="utf-8")
    findings = package.scan_for_secrets(
        tmp_path, ["safe.md", "unsafe.txt"]
    )
    assert findings == [{"path": "unsafe.txt", "type": "aws_access_key"}]
    assert key not in repr(findings)
