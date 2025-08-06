from datetime import datetime

from pyspark.sql import SparkSession
from pyspark.sql.types import (
    BooleanType,
    DoubleType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

from includes.data_processing import extract_json_fields


def test_json_processing() -> None:
    spark = SparkSession.builder.getOrCreate()
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
            datetime(2024, 3, 13, 2, 50, 0),
            2297,
            12,
            False,
            "0213-Millenáris",
            47.51032,
            19.028615,
        ),
        (
            datetime(2024, 3, 13, 2, 50, 0),
            2298,
            20,
            False,
            "1120-Budapart Gate",
            47.468217,
            19.058447,
        ),
    ]

    expected = spark.createDataFrame(data, schema)

    from pandas.testing import assert_frame_equal

    assert_frame_equal(result.toPandas(), expected.toPandas())
