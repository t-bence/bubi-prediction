from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def extract_json_fields(df: DataFrame) -> DataFrame:
    """
    Extracts relevant fields from a nested JSON structure in a Spark DataFrame.

    This function processes a DataFrame containing nested JSON data about bike stations,
    extracting timestamp information from the filename and flattening the nested structure
    to obtain details about each station.

    Parameters
    ----------
    df : DataFrame
        Input Spark DataFrame containing the nested JSON data.

    Returns:
    DataFrame
        A Spark DataFrame with the following columns:
            - station_id (int): Unique identifier of the bike station.
            - bikes (int): Number of bikes available at the station.
            - maintenance (bool): Maintenance status of the station.
            - station_name (str): Name of the bike station.
            - lat (float): Latitude of the station.
            - lng (float): Longitude of the station.
            - ts (timestamp): Parsed timestamp from the filename.
    """

    DATE_FORMAT = "yyyy-MM-dd'T'HH-mm-ss'Z.json'"

    return (
        df.select(
            F.to_timestamp("filename", DATE_FORMAT).alias("ts"),
            F.explode(F.col("countries")[0].cities[0].places).alias("places"),
        )
        .filter(F.col("places.spot"))  # keep stations only, not bikes left around
        .select(
            "ts",
            F.col("places.number").alias("station_id"),
            F.col("places.bikes").alias("bikes"),
            F.col("places.maintenance").alias("maintenance"),
            F.col("places.name").alias("station_name"),
            F.col("places.lat").alias("lat"),
            F.col("places.lng").alias("lng"),
        )
    )


def temporal_deduplication(df: DataFrame) -> DataFrame:
    """Deduplicate so that there is only one row for every n minutes

    Parameters
    ----------
    df : DataFrame
        The input dataframe

    Returns
    -------
    DataFrame
        Input DataFrame with duplicate entries removed from every 10 minutes
    """

    SEC_PER_MINUTE = 60
    seconds = 10 * SEC_PER_MINUTE
    watermark = "30 minutes"

    return (
        df.withColumn(
            "ts",
            F.expr(
                f"timestamp_seconds(floor(unix_timestamp(ts) / {seconds}) * {seconds})"
            ),
        )
        .withWatermark("ts", watermark)
        .dropDuplicates(["ts", "station_id"])
    )
