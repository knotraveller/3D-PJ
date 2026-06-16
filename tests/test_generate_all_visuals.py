from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from tools.generate_all_visuals import is_ref_directory, output_path_for_sample


def test_is_ref_directory_requires_ref_number_name(tmp_path: Path) -> None:
    valid = tmp_path / "ref_000"
    invalid = tmp_path / "reference_000"
    valid.mkdir()
    invalid.mkdir()

    assert is_ref_directory(valid)
    assert not is_ref_directory(invalid)
    assert not is_ref_directory(tmp_path / "ref_001")


def test_output_path_preserves_relative_sample_structure(tmp_path: Path) -> None:
    image_root = tmp_path / "renders"
    sample_dir = image_root / "asset_123" / "ref_090"
    output_root = tmp_path / "outputs" / "all_visuals" / "timestamp"

    result = output_path_for_sample(sample_dir, image_root, output_root)

    assert result == output_root / "asset_123" / "ref_090.png"
