"""Tests for the stratified split and the saved index files."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from ml import config
from ml.data import load
from ml.preprocessing import split


@pytest.fixture(scope="module")
def maths() -> pd.DataFrame:
    frame, _ = load.load_subject("mat")
    return frame


def test_split_sizes_match_the_documented_ratio(maths: pd.DataFrame) -> None:
    train, test = split.stratified_split(maths)

    assert len(train) == 316
    assert len(test) == 79
    assert len(train) + len(test) == len(maths)


def test_indices_are_disjoint_sorted_and_complete(maths: pd.DataFrame) -> None:
    train, test = split.stratified_split(maths)

    assert len(set(train) & set(test)) == 0
    assert list(train) == sorted(train)
    assert list(test) == sorted(test)
    assert set(train) | set(test) == set(range(len(maths)))


def test_same_seed_reproduces_the_same_split(maths: pd.DataFrame) -> None:
    first = split.stratified_split(maths, seed=config.SEED)
    second = split.stratified_split(maths, seed=config.SEED)
    other = split.stratified_split(maths, seed=7)

    assert np.array_equal(first[1], second[1])
    assert not np.array_equal(first[1], other[1])


def test_every_band_keeps_its_share(maths: pd.DataFrame) -> None:
    _, test = split.stratified_split(maths)
    bands = split.grade_bands(maths)

    for position in range(len(config.RISK_BAND_LABELS)):
        in_band = bands == position
        share = in_band.iloc[test].sum() / in_band.sum()
        assert share == pytest.approx(config.TEST_SIZE, abs=0.06)


def test_grade_bands_follow_the_configured_edges() -> None:
    frame = pd.DataFrame({config.TARGET: [9, 10, 13, 14, 16, 17, 20]})

    assert split.grade_bands(frame).tolist() == [0, 1, 1, 2, 2, 3, 3]


def test_payload_records_provenance(maths: pd.DataFrame) -> None:
    train, test = split.stratified_split(maths)
    payload = split.split_payload("mat", maths, train, test)

    assert payload["rows"] == 395
    assert payload["seed"] == config.SEED
    assert len(payload["dataset_sha256"]) == 64
    assert sum(entry["total"] for entry in payload["bands"].values()) == 395
    assert len(payload["train_indices"]) == 316
    assert len(payload["test_indices"]) == 79


def test_saved_split_round_trips(maths: pd.DataFrame, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(split, "SPLITS_DIR", tmp_path / "splits")
    train, test = split.stratified_split(maths)
    payload = split.split_payload("mat", maths, train, test)
    path = split.save_split("mat", payload)

    assert path.exists()
    assert json.loads(path.read_text(encoding="utf-8"))["rows"] == 395
    assert split.load_split("mat")["test_indices"] == payload["test_indices"]


def test_load_split_rejects_overlapping_indices(maths, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(split, "SPLITS_DIR", tmp_path / "splits")
    train, test = split.stratified_split(maths)
    payload = split.split_payload("mat", maths, train, test)
    payload["test_indices"] = payload["train_indices"][:10]
    split.save_split("mat", payload)

    with pytest.raises(ValueError, match="both train and test"):
        split.load_split("mat")


def test_single_row_band_cannot_be_stratified() -> None:
    frame = pd.DataFrame({config.TARGET: [5, 12, 12, 18, 18]})

    with pytest.raises(ValueError, match="Cannot stratify"):
        split.stratified_split(frame)
