import datetime as dt

import pandas as pd

from includes.inference import (
    create_prediction_time_dataframe,
    create_predictions_spark_dataframe,
)


def test_create_prediction_time_dataframe_with_fixed_time():
    test_time = dt.datetime(2024, 3, 13, 14, 23, 45, tzinfo=dt.UTC)
    result = create_prediction_time_dataframe(test_time)

    assert len(result) == 144
    assert result["ts"].iloc[0].tz is not None
    assert result["ts"].iloc[0].hour == 14
    assert result["ts"].iloc[0].minute == 20
    assert result["ts"].iloc[0].second == 0
    assert result["ts"].iloc[0].microsecond == 0

    assert result["ts"].iloc[-1].hour == 14
    assert result["ts"].iloc[-1].minute == 10


def test_create_prediction_time_dataframe_floors_to_10_minutes():
    test_time = dt.datetime(2024, 3, 13, 14, 27, 30, tzinfo=dt.UTC)
    result = create_prediction_time_dataframe(test_time)

    assert result["ts"].iloc[0].minute == 20


def test_create_prediction_time_dataframe_10_minute_intervals():
    test_time = dt.datetime(2024, 3, 13, 14, 0, 0, tzinfo=dt.UTC)
    result = create_prediction_time_dataframe(test_time)

    time_diffs = result["ts"].diff().dropna()
    assert all(time_diffs == dt.timedelta(minutes=10))


def test_create_prediction_time_dataframe_without_provided_time():
    result = create_prediction_time_dataframe()

    assert len(result) == 144
    assert result["ts"].iloc[0].tz is not None


def test_create_predictions_spark_dataframe(spark):
    test_time = dt.datetime(2024, 3, 13, 14, 0, 0, tzinfo=dt.UTC)
    pdf = create_prediction_time_dataframe(test_time)
    bikes = pd.Series([10, 20, 30] + [0] * 141)
    model_version = "1"
    prediction_date = dt.datetime(2024, 3, 13, 14, 0, 0, tzinfo=dt.UTC)

    result = create_predictions_spark_dataframe(
        pdf, bikes, model_version, prediction_date, spark
    )

    assert result.count() == 144
    assert "ts" in result.columns
    assert "bikes" in result.columns
    assert "model_version" in result.columns
    assert "prediction_date" in result.columns

    pandas_result = result.toPandas()
    assert pandas_result["bikes"].iloc[0] == 10
    assert pandas_result["bikes"].iloc[1] == 20
    assert pandas_result["bikes"].iloc[2] == 30
    assert pandas_result["model_version"].iloc[0] == "1"


def test_create_predictions_spark_dataframe_timezone_preserved(spark):
    test_time = dt.datetime(2024, 3, 13, 14, 23, 45, tzinfo=dt.UTC)
    pdf = create_prediction_time_dataframe(test_time)
    bikes = pd.Series([5] * 144)
    model_version = "2"
    prediction_date = dt.datetime(2024, 3, 13, 15, 0, 0, tzinfo=dt.UTC)

    result = create_predictions_spark_dataframe(
        pdf, bikes, model_version, prediction_date, spark
    )

    pandas_result = result.toPandas()
    assert pandas_result["ts"].iloc[0].hour == 14
    assert pandas_result["ts"].iloc[0].minute == 20
