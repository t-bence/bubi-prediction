import argparse
import datetime as dt

import pyspark.sql.functions as F
from pyspark.sql import SparkSession

from includes.data_processing import temporal_deduplication
from includes.utilities import get_json_schema, get_table_name

spark = SparkSession.builder.getOrCreate()

parser = argparse.ArgumentParser(description="Data processing arguments")
parser.add_argument("--catalog", type=str, default="", help="Catalog name")
parser.add_argument("--schema", type=str, default="", help="Schema name")
parser.add_argument("--volume", type=str, default="", help="Volume name")
parser.add_argument(
    "--checkpoint_volume", type=str, default="", help="Checkpoint volume name"
)
args, unknown = parser.parse_known_args()

catalog = args.catalog
schema = args.schema
volume = args.volume
checkpoint_volume = args.checkpoint_volume

if not catalog or not schema or not volume or not checkpoint_volume:
    raise ValueError("Catalog, Schema, and Volume must not be empty")

# Create bronze table
# experiment with this...
# spark.conf.set("spark.sql.shuffle.partitions", spark.sparkContext.defaultParallelism)

bronze = (
    spark.readStream.format("cloudFiles")
    .option("cloudFiles.format", "json")
    .schema(get_json_schema())
    .load(f"/Volumes/{catalog}/{schema}/{volume}")
    .withColumn("filename", F.col("_metadata.file_name"))
    .withColumn("ingestion_time", F.lit(dt.datetime.utcnow()))
)


bronze_query = (
    bronze.writeStream.outputMode("append")
    .queryName("json_stream")
    .trigger(availableNow=True)
    .option(
        "checkpointLocation",
        f"/Volumes/{catalog}/{schema}/{checkpoint_volume}/bronze",
    )
    .toTable(get_table_name(catalog, schema, "bronze"))
)

bronze_query.awaitTermination()

# Create silver table
# This will be a batch read now to simplify things


date_length = len("2024-03-14T01-30-02")  # skip trailing Z

silver = (
    spark.readStream.table(get_table_name(catalog, schema, "bronze"))
    .withColumn(
        "ts",
        F.to_timestamp(
            F.substring("filename", 0, date_length), "yyyy-MM-dd'T'HH-mm-ss"
        ),
    )
    .withColumn("data", F.element_at("countries", 1))
    .withColumn("cities", F.col("data.cities"))
    .withColumn("data", F.element_at("cities", 1))
    .select("ts", "data", "ingestion_time")
    .withColumn("places", F.col("data.places"))
    .drop("data")
    .withColumn("places", F.explode("places"))
    # next 2 lines remove random bikes left around
    .filter(F.col("places.spot") == F.lit(True))
    .filter(F.col("places.bike") == F.lit(False))
    .select("ts", "places", "ingestion_time")
    # extract the useful columns
    .withColumn("station_id", F.col("places.number"))
    .withColumn("bikes", F.col("places.bikes"))
    .withColumn("maintenance", F.col("places.maintenance"))
    .withColumn("station_name", F.col("places.name"))
    .withColumn("lat", F.col("places.lat"))
    .withColumn("lng", F.col("places.lng"))
    .drop("places", "ingestion_time")  # ingestion_time is not used currently
    .transform(
        temporal_deduplication, column="ts", minutes=10, other_cols=["station_id"]
    )
)

(
    silver.writeStream.outputMode("append")
    .queryName("silver_stream")
    .trigger(availableNow=True)
    .option(
        "checkpointLocation",
        f"/Volumes/{catalog}/{schema}/{checkpoint_volume}/silver",
    )
    .toTable(get_table_name(catalog, schema, "silver"))
    .awaitTermination()
)

# Features
# Compute the final features

gold = (
    spark.read.table(get_table_name(catalog, schema, "silver"))
    .groupBy("station_id", "ts")
    .agg(F.first("bikes").alias("bikes"))
)

gold.write.mode("overwrite").saveAsTable(get_table_name(catalog, schema, "gold"))
