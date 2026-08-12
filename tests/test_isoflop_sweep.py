from __future__ import annotations

import csv

import pytest

from scripts.isoflop_sweep import FIELDNAMES, append_row, load_completed_cells

SEED = 1337  # --seed's default
PRE_REPEATS_FIELDNAMES = [name for name in FIELDNAMES if name != "seed"]

# One row in the pre-repeats layout, values in that header's order.
PRE_REPEATS_ROW = dict(
    zip(PRE_REPEATS_FIELDNAMES, [1e14, 32, 4, 1, 32, 1_000_000, 800_000, 100_000, 400, 7.05, 5.78, 62.5])
)
ROW = {**PRE_REPEATS_ROW, "seed": SEED}


def write_csv(path, fieldnames, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_load_completed_cells_keys_on_budget_width_and_seed(tmp_path):
    csv_path = tmp_path / "results.csv"
    write_csv(csv_path, FIELDNAMES, [ROW, {**ROW, "seed": SEED + 1, "elapsed_s": 70.0}])

    assert load_completed_cells(csv_path) == {(1e14, 32, SEED): 62.5, (1e14, 32, SEED + 1): 70.0}


def test_load_completed_cells_refuses_a_pre_seed_column_csv(tmp_path):
    # The bug this guards: a CSV has one header for the whole file and
    # append_row writes values in FIELDNAMES order underneath it, so
    # resuming into a pre-repeats CSV wrote the seed into the n_layer
    # column and shifted every field after it.
    csv_path = tmp_path / "results.csv"
    write_csv(csv_path, PRE_REPEATS_FIELDNAMES, [PRE_REPEATS_ROW])

    with pytest.raises(ValueError, match="point --out-dir at a new directory"):
        load_completed_cells(csv_path)


def test_load_completed_cells_refuses_an_unrecognized_header(tmp_path):
    csv_path = tmp_path / "results.csv"
    write_csv(csv_path, ["flops_budget", "width", "something_else"], [])

    with pytest.raises(ValueError, match="has header"):
        load_completed_cells(csv_path)


def test_load_completed_cells_ignores_missing_file(tmp_path):
    assert load_completed_cells(tmp_path / "absent.csv") == {}


def test_append_row_round_trips_through_load_completed_cells(tmp_path):
    # A fresh --out-dir is the path that matters: append_row writes the
    # current header itself, so resume reads back exactly what it wrote.
    csv_path = tmp_path / "results.csv"

    append_row(csv_path, ROW)
    append_row(csv_path, {**ROW, "seed": SEED + 1, "width": 40, "elapsed_s": 71.0})

    assert load_completed_cells(csv_path) == {(1e14, 32, SEED): 62.5, (1e14, 40, SEED + 1): 71.0}


def test_narrow_head_dims_avoid_flex_attention():
    # FlexAttention's kernel rejects head_dim < 16 ("NYI: embedding dimension
    # ... must be at least 16"), which is what killed widths 4 and 8 in the
    # v3 sweep. Those must take the dense mask instead of being dropped.
    from src.model.gpt import FLEX_MIN_HEAD_DIM
    from src.model.scaling import gpt_shape_for_width

    for width in (4, 8):
        _, n_head, n_embd = gpt_shape_for_width(width)
        assert n_embd // n_head < FLEX_MIN_HEAD_DIM, f"width {width} was expected to be below the limit"
    for width in (16, 24, 32, 64, 96, 512):
        _, n_head, n_embd = gpt_shape_for_width(width)
        assert n_embd // n_head >= FLEX_MIN_HEAD_DIM, f"width {width} should still use FlexAttention"
