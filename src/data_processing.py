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

DATE_FORMAT = "yyyy-MM-dd'T'HH-mm-ss'Z.json'"

silver = (
    spark.readStream.table(get_table_name(catalog, schema, "bronze"))
    .select(
        F.to_timestamp("filename", DATE_FORMAT).alias("ts"),
        F.explode(F.col("countries")[0].cities[0].places).alias("bikes"),
    )
    .filter(F.col("bikes.spot"))  # keep stations only, not bikes left around
    .select(
        F.col("places.number").alias("station_id"),
        F.col("places.bikes").alias("bikes"),
        F.col("places.maintenance").alias("maintenance"),
        F.col("places.name").alias("station_name"),
        F.col("places.lat").alias("lat"),
        F.col("places.lng").alias("lng"),
        "ts",
    )
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
