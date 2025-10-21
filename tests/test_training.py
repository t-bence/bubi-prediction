from datetime import datetime, timedelta
from unittest.mock import patch

from includes.training import get_train_cutoff


def test_get_train_cutoff_valid_iso_format():
    date_string = "2024-03-13T12:30:00"
    result = get_train_cutoff(date_string)
    expected = datetime(2024, 3, 13, 12, 30, 0)
    assert result == expected


def test_get_train_cutoff_valid_iso_format_with_timezone():
    date_string = "2024-03-13T12:30:00+00:00"
    result = get_train_cutoff(date_string)
    expected = datetime.fromisoformat(date_string)
    assert result == expected


def test_get_train_cutoff_empty_string():
    fixed_now = datetime(2024, 3, 13, 12, 0, 0)
    with patch("includes.training.datetime") as mock_datetime:
        mock_datetime.now.return_value = fixed_now
        mock_datetime.fromisoformat.side_effect = ValueError

        result = get_train_cutoff("")
        expected = fixed_now - timedelta(days=14)
        assert result == expected


def test_get_train_cutoff_invalid_date():
    fixed_now = datetime(2024, 3, 13, 12, 0, 0)
    with patch("includes.training.datetime") as mock_datetime:
        mock_datetime.now.return_value = fixed_now
        mock_datetime.fromisoformat.side_effect = ValueError

        result = get_train_cutoff("not-a-date")
        expected = fixed_now - timedelta(days=14)
        assert result == expected


def test_get_train_cutoff_returns_datetime():
    date_string = "2024-01-01T00:00:00"
    result = get_train_cutoff(date_string)
    assert isinstance(result, datetime)
