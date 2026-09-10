from __future__ import annotations

from pathlib import Path

import pytest

from app.config import load_taxonomy

REPO_STYLES = Path(__file__).resolve().parents[2] / "styles.yaml"


def test_repo_styles_file_parses() -> None:
    tax = load_taxonomy(REPO_STYLES)
    assert len(tax.styles) >= 2
    assert "urban_streetwear" in tax.names
    assert all(s.prompt for s in tax.styles)
    assert len(tax.prompts) == len(tax.names)


def test_taxonomy_lookup() -> None:
    tax = load_taxonomy(REPO_STYLES)
    assert tax.by_name("gothic").display_name == "Gothic"
    assert tax.by_name("does_not_exist") is None


def test_empty_styles_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "styles.yaml"
    bad.write_text("version: 1\nstyles: {}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_taxonomy(bad)


def test_display_name_defaulted(tmp_path: Path) -> None:
    f = tmp_path / "styles.yaml"
    f.write_text("styles:\n  wild_thing:\n    prompt: a wild outfit\n", encoding="utf-8")
    tax = load_taxonomy(f)
    assert tax.by_name("wild_thing").display_name == "Wild Thing"
