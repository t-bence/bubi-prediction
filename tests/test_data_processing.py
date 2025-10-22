import datetime as dt

from pyspark.sql.types import (
    BooleanType,
    DoubleType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)
from pyspark.testing import assertDataFrameEqual

from includes.data_processing import (
    extract_json_fields,
    extract_timestamp_from_filename,
    temporal_deduplication,
)


def test_extract_timestamp_from_filename(spark):
    import pyspark.sql.functions as F
    from pyspark.sql import Row

    # Test dash format
    dash_row = Row(filename="2024-03-13T02-50-00Z.json")
    # Test colon format
    colon_row = Row(filename="2024-03-13T02:50:00Z.json")
    result = spark.createDataFrame([dash_row, colon_row]).select(
        extract_timestamp_from_filename(F.col("filename")).alias("ts")
    )
    expected = spark.createDataFrame(
        [
            (dt.datetime(2024, 3, 13, 2, 50, 0, tzinfo=dt.UTC),),
            (dt.datetime(2024, 3, 13, 2, 50, 0, tzinfo=dt.UTC),),
        ],
        schema="ts TIMESTAMP",
    )

    assertDataFrameEqual(result, expected)


def test_json_processing(spark) -> None:
    df = spark.read.json("./tests/files/bronze_sample_row.json")
    result = df.transform(extract_json_fields).limit(2)

    schema = StructType(
        [
            StructField("ts", TimestampType(), True),
            StructField("station_id", LongType(), True),
            StructField("bikes", LongType(), True),
            StructField("maintenance", BooleanType(), True),
            StructField("station_name", StringType(), True),
            StructField("lat", DoubleType(), True),
            StructField("lng", DoubleType(), True),
        ]
    )

    data = [
        (
            dt.datetime(2024, 3, 13, 2, 50, 0, tzinfo=dt.UTC),
            2297,
            12,
            False,
            "0213-Millenáris",
            47.51032,
            19.028615,
        ),
        (
            dt.datetime(2024, 3, 13, 2, 50, 0, tzinfo=dt.UTC),
            2298,
            20,
            False,
            "1120-Budapart Gate",
            47.468217,
            19.058447,
        ),
    ]

    expected = spark.createDataFrame(data, schema)

    assertDataFrameEqual(result, expected)


def test_temporal_deduplication(spark):
    schema = StructType(
        [
            StructField("ts", TimestampType(), True),
            StructField("station_id", LongType(), True),
            StructField("bikes", LongType(), True),
        ]
    )

    data = [
        (
            dt.datetime(2024, 3, 13, 2, 40, 0, tzinfo=dt.UTC),
            1,
            1,
        ),
        (
            dt.datetime(2024, 3, 13, 2, 50, 0, tzinfo=dt.UTC),
            1,
            2,
        ),
        (
            dt.datetime(2024, 3, 13, 2, 55, 0, tzinfo=dt.UTC),
            1,
            2,
        ),
    ]
    df = spark.createDataFrame(data, schema)
    assert df.transform(temporal_deduplication).count() == 2
