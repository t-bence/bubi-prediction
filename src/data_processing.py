import argparse
import datetime as dt
import logging

import pyspark.sql.functions as F
from pyspark.sql import SparkSession

from includes.data_processing import extract_json_fields, temporal_deduplication
from includes.utilities import get_json_schema, get_table_name

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

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
    logger.error("Catalog, Schema, and Volume must not be empty")
    raise ValueError("Catalog, Schema, and Volume must not be empty")

logger.info("Starting data processing script")
logger.info(
    f"Using catalog={catalog}, schema={schema}, volume={volume}, checkpoint_volume={checkpoint_volume}"
)

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

logger.info("Bronze table stream started")

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

logger.info("Bronze query completed")
logger.info(
    f"Bronze query has written {bronze_query.lastProgress['numInputRows']} rows"
)


# Create silver table
silver = (
    spark.read.table(get_table_name(catalog, schema, "bronze"))
    .transform(extract_json_fields)
    .transform(temporal_deduplication)
)

(silver.write.mode("overwrite").saveAsTable(get_table_name(catalog, schema, "silver")))

logger.info("Silver table created")

# Features
# Compute the final features

gold = (
    spark.read.table(get_table_name(catalog, schema, "silver"))
    .groupBy("station_id", "ts")
    .agg(F.first("bikes").alias("bikes"))
)

gold.write.mode("overwrite").saveAsTable(get_table_name(catalog, schema, "gold"))

logger.info("Gold table created")
