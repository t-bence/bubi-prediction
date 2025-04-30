# Databricks notebook source

import pyspark.sql.functions as F
from pyspark.sql import SparkSession

from includes.utilities import get_json_schema, get_table_name

spark = SparkSession.builder.getOrCreate()

dbutils.widgets.text("catalog", "", "Catalog")
dbutils.widgets.text("schema", "", "Schema")
dbutils.widgets.text("volume", "", "Volume")
dbutils.widgets.text("checkpoint_volume_name", "", "Checkpoint volume")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
volume = dbutils.widgets.get("volume")
checkpoint_volume_name = dbutils.widgets.get("checkpoint_volume_name")

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
    .withWatermark("ts", "1 hour")
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
    # fix timestamps to correspond to always XX:10:00 or XX:20:00 or so
    .withColumn(
        "ts", F.expr("timestamp_seconds(floor(unix_timestamp(ts) / 600) * 600)")
    )
    .dropDuplicates(["ts", "station_id"])
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

# station distances

# stations = (silver
#     .select("station_id", "station_name", "lat", "lng")
#     .dropDuplicates()
#     .withColumn("district", F.substring("station_name", 1, 2).cast("int"))
# )

# def dist(long_x, lat_x, long_y, lat_y):
#     return F.acos(
#         F.sin(F.radians(lat_x)) * F.sin(F.radians(lat_y)) +
#         F.cos(F.radians(lat_x)) * F.cos(F.radians(lat_y)) *
#             F.cos(F.radians(long_x) - F.radians(long_y))
#     ) * F.lit(6371.0 * 1000)

# from pyspark.sql import Window

# N_closest_stations = 5

# distanceWindowSpec = (Window
#     .partitionBy("station_id")
#     .orderBy(F.col("distance_meters"))
# )

# closest_stations = (stations
#     .drop("station_name", "district")
#     .crossJoin(stations
#     .drop("station_name", "district")
#     .withColumnRenamed("station_id", "other_station_id")
#     .withColumnRenamed("lat", "other_lat")
#     .withColumnRenamed("lng", "other_lng")
#     )
#     .filter(F.col("station_id") != F.col("other_station_id"))
#     .withColumn("distance_meters",
#         dist(F.col("lng"), F.col("lat"), F.col("other_lng"), F.col("other_lat")).cast("int")
#     )
#     .select("station_id", "other_station_id", "distance_meters")
#     .withColumn("rank", F.dense_rank().over(distanceWindowSpec))
#     .filter(F.col("rank") <= N_closest_stations)
#     .groupBy("station_id")
#     .agg(F.collect_list("other_station_id").alias("closest_stations"))
#     # dense_rank give ties, so we have to filter those to have exactly just five
#     .withColumn("closest_stations", F.slice("closest_stations", 1, N_closest_stations))
# )

# closest_stations.write.mode("overwrite").saveAsTable(
#     get_full_name(catalog, schema, "closest_stations")
# )

# closest_stations.printSchema()

# COMMAND ----------
# MAGIC %md
# MAGIC # Features
# MAGIC
# MAGIC Compute the final features

# COMMAND ----------

# old feature processing code
# features = (silver
#     .join(closest_stations, "station_id", "left")
#     .withColumn("close_station", F.explode("closest_stations"))
#     .drop("closest_stations")
#     .join(silver
#             .withColumnRenamed("station_id", "close_station")
#             .withColumnRenamed("bikes", "close_bikes"),
#         ["close_station", "ts"], "left")
#     .na.fill(0, "close_bikes")
#     .groupBy("station_id", "ts")
#     .agg(
#         F.first("bikes").alias("bikes"),
#         F.collect_list("close_bikes").alias("close_bikes")
#     )
#     .withColumn("close_bikes_1", F.element_at("close_bikes", 1))
#     .withColumn("close_bikes_2", F.element_at("close_bikes", 2))
#     .withColumn("close_bikes_3", F.element_at("close_bikes", 3))
#     .withColumn("close_bikes_4", F.element_at("close_bikes", 4))
#     .withColumn("close_bikes_5", F.element_at("close_bikes", 5))
#     .drop("close_bikes")
#     .select("station_id", "ts", "bikes")
# )


gold = (
    spark.read.table(get_table_name(catalog, schema, "silver"))
    .groupBy("station_id", "ts")
    .agg(F.first("bikes").alias("bikes"))
)

gold.display()

gold.write.mode("overwrite").saveAsTable(get_table_name(catalog, schema, "gold"))
