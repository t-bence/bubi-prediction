import datetime as dt

import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql import DataFrame, SparkSession


def create_prediction_time_dataframe(now: dt.datetime | None = None) -> pd.DataFrame:
    HOURS_IN_DAY = 24
    MINUTES_IN_HOUR = 60
    PREDICTION_PERIOD_MINUTES = 10
    periods = int(HOURS_IN_DAY * MINUTES_IN_HOUR / PREDICTION_PERIOD_MINUTES - 1)

    if now is None:
        now = dt.datetime.now(dt.UTC)
    floored_minute = (now.minute // 10) * 10
    now = now.replace(minute=floored_minute, second=0, microsecond=0)
    future_times = [now + dt.timedelta(minutes=10 * i) for i in range(periods + 1)]
    pdf = pd.DataFrame({"ts": future_times})
    pdf["ts"] = pdf["ts"].dt.tz_localize(
        None
    )  # remove time zone for compatbility reasons

    return pdf


def create_predictions_spark_dataframe(
    pdf: pd.DataFrame,
    bikes: pd.Series,
    model_version: str,
    prediction_date: dt.datetime,
    spark: SparkSession,
) -> DataFrame:
    pdf = pdf.copy()
    pdf["bikes"] = bikes
    preds_df = (
        spark.createDataFrame(pdf)
        .withColumn("model_version", F.lit(model_version))
        .withColumn("prediction_date", F.lit(prediction_date))
    )
    return preds_df
