# Databricks notebook source
import mlflow
import pyspark.sql.functions as F
from mlflow.tracking import MlflowClient
from pyspark.dbutils import dbutils
from pyspark.sql import SparkSession

from includes.utilities import get_table_name

spark = SparkSession.builder.getOrCreate()

dbutils.widgets.text("catalog", "", "Catalog")
dbutils.widgets.text("schema", "", "Schema")
dbutils.widgets.text("model_name", "", "Model name")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
model_name = dbutils.widgets.get("model_name")


if not catalog or not schema or not model_name:
    raise ValueError("None of the parameters may be empty")


mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------
# Evaluation
# If this model performs better (has a better test_mean_absolute_error) than the Champion,
# register it and give it the Champion label

client = MlflowClient()

model_fqn = f"{catalog}.{schema}.{model_name}"

# get challenger version
try:
    challenger_version = client.get_model_version_by_alias(model_fqn, "Challenger")
except Exception:
    print("No Challenger version exists")
    dbutils.notebook.exit(0)

challenger_run = client.get_run(challenger_version.run_id)
challenger_mae = challenger_run.data.metrics.get("test_mean_absolute_error")

print(f"Challenger MAE: {challenger_mae}")

# get champion version
try:
    champion_version = client.get_model_version_by_alias(model_fqn, "Champion")
except Exception:
    print("No Champion exists, continue validation")
    metric_mae_passed = True
else:
    champion_run = client.get_run(champion_version.run_id)
    champion_mae = champion_run.data.metrics.get("test_mean_absolute_error")

    print(f"Champion MAE: {champion_mae}")

    metric_mae_passed = challenger_mae < champion_mae

    print(f"MAE metric passed: {metric_mae_passed}")


client.set_model_version_tag(
    model_fqn,
    challenger_version.version,
    key="metric_mae_passed",
    value=metric_mae_passed,
)


# determine if the challenger model can make predictions
try:
    model_uri = f"models:/{model_name}@Challenger"
    challenger_model = mlflow.pyfunc.load_model(spark, model_uri)

    validation_df = (
        spark.read.table(get_table_name(catalog, schema, "gold")).select(
            F.col("ts").alias("ds")
        )  # here we load data for all stations!
    )

    feature_columns = [F.col(column) for column in validation_df.columns]
    preds = validation_df.withColumn(
        "prediction", challenger_model(F.struct(feature_columns))
    )

    preds.display()

    can_predict = True

except Exception as e:
    print("Unable to predict on features.")
    print(e)
    can_predict = False

client.set_model_version_tag(
    model_fqn, version=challenger_version.version, key="can_predict", value=can_predict
)


if metric_mae_passed and can_predict:
    print("Promoting version to Champion")
    client.set_registered_model_alias(
        name=model_fqn, version=challenger_version.version, alias="Champion"
    )
