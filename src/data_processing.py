# Databricks notebook source

# MAGIC %pip install -r ../databricks-requirements.txt

# COMMAND ----------

import datetime as dt

import pyspark.sql.functions as F
from pyspark.sql import SparkSession

from includes.data_processing import temporal_deduplication
from includes.utilities import get_json_schema, get_table_name

spark = SparkSession.builder.getOrCreate()

# ruff: noqa: F821
dbutils.widgets.text("catalog", "", "Catalog")
dbutils.widgets.text("schema", "", "Schema")
dbutils.widgets.text("volume", "", "Volume")
dbutils.widgets.text("checkpoint_volume_name", "", "Checkpoint volume")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
volume = dbutils.widgets.get("volume")
checkpoint_volume_name = dbutils.widgets.get("checkpoint_volume_name")

# ruff: enable

if not catalog or not schema or not volume or not checkpoint_volume_name:
    raise ValueError("Catalog, Schema, and Volume must not be empty")

# COMMAND ----------
# MAGIC %md
# MAGIC # Create bronze table
# MAGIC
# MAGIC Do streaming ingestion here

# COMMAND ----------

# experiment with this...
# spark.conf.set("spark.sql.shuffle.partitions", spark.sparkContext.defaultParallelism)

bronze = (
    spark.readStream.format("cloudFiles")
    .option("cloudFiles.format", "json")
    .schema(get_json_schema())
    .load(f"/Volumes/{catalog}/{schema}/{volume}")
    .withColumn("filename", F.col("_metadata.file_name"))
    .withColumn("ingestion_time", dt.now())
)


bronze_query = (
    bronze.writeStream.outputMode("append")
    .queryName("json_stream")
    .trigger(availableNow=True)
    .option(
        "checkpointLocation",
        f"/Volumes/{catalog}/{schema}/{checkpoint_volume_name}/bronze",
    )
    .toTable(get_table_name(catalog, schema, "bronze"))
)

bronze_query.awaitTermination()

# COMMAND ----------
# MAGIC %md
# MAGIC # Create silver table
# MAGIC
# MAGIC This will be a batch read now to simplify things


# COMMAND ----------

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
    .select("ts", "data")
    .withColumn("places", F.col("data.places"))
    .drop("data")
    .withColumn("places", F.explode("places"))
    # next 2 lines remove random bikes left around
    .filter(F.col("places.spot") == F.lit(True))
    .filter(F.col("places.bike") == F.lit(False))
    .select("ts", "places")
    # truncate timestamp to minutes only
    # .withColumn("ts", F.date_trunc("minute", F.col("ts")))
    # drop first few runs that are sometimes not at a ten-minute interval
    # .filter(F.col("ts") >= "2024-03-12T12:00:00.000+00:00")
    # extract the useful columns
    .withColumn("station_id", F.col("places.number"))
    .withColumn("bikes", F.col("places.bikes"))
    .withColumn("maintenance", F.col("places.maintenance"))
    .withColumn("station_name", F.col("places.name"))
    .withColumn("lat", F.col("places.lat"))
    .withColumn("lng", F.col("places.lng"))
    .drop("places")
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
        f"/Volumes/{catalog}/{schema}/{checkpoint_volume_name}/silver",
    )
    .toTable(get_table_name(catalog, schema, "silver"))
    .awaitTermination()
)

# COMMAND ----------
# MAGIC %md
# MAGIC # Features
# MAGIC
# MAGIC Compute the final features

# COMMAND ----------

gold = (
    spark.read.table(get_table_name(catalog, schema, "silver"))
    .groupBy("station_id", "ts")
    .agg(F.first("bikes").alias("bikes"))
)

gold.write.mode("overwrite").saveAsTable(get_table_name(catalog, schema, "gold"))
