from datetime import date, timedelta

import pytest

from orinoco_forecast.data.splits import walk_forward_splits


def test_walk_forward_splits_are_chronological() -> None:
    dates = [date(2022, 1, 1) + timedelta(days=index) for index in range(400)]
    splits = walk_forward_splits(
        dates,
        test_start=date(2023, 1, 1),
        horizon=14,
        step_size=30,
        n_windows=3,
    )

    assert [split.cutoff for split in splits] == [
        date(2022, 10, 18),
        date(2022, 11, 17),
        date(2022, 12, 17),
    ]
    assert splits[-1].validation_start == date(2022, 12, 18)
    assert splits[-1].validation_end == date(2023, 1, 1) - timedelta(days=1)


def test_walk_forward_splits_reject_unsorted_dates() -> None:
    with pytest.raises(ValueError, match="ordenado"):
        walk_forward_splits(
            [date(2022, 1, 2), date(2022, 1, 1)],
            test_start=date(2023, 1, 1),
            horizon=14,
            step_size=30,
            n_windows=1,
        )
