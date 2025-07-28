from typing import Optional

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def temporal_deduplication(
    df: DataFrame, column: str, minutes: int, other_cols: Optional[list[str]] = []
) -> DataFrame:
    """Deduplicate so that there is only one row for every n minutes

    Parameters
    ----------
    df : DataFrame
        The input dataframe
    column : str
        Name of the timestamp column that serves as base for deduplication
    minutes : int
        Length of period in minutes for unique rows
    other_cols : Optional[list[str]]
        Other columns to use with deduplication

    Returns
    -------
    DataFrame
        Input DataFrame with duplicate entries removed from every `minutes` minutes
    """

    all_cols = [column] + other_cols
    seconds = minutes * 60
    watermark = f"{5 * minutes} minutes"

    return (
        df.withColumn(
            column,
            F.expr(
                f"timestamp_seconds(floor(unix_timestamp({column}) / {seconds}) * {seconds})"
            ),
        )
        .withWatermark(column, watermark)
        .dropDuplicates(all_cols)
    )
