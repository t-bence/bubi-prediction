# Databricks notebook source

import mlflow
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

dbutils.widgets.text("catalog", "", "Catalog")
dbutils.widgets.text("schema", "", "Schema")
dbutils.widgets.text("model_name", "", "Model name")
dbutils.widgets.text("previous_run_id", "", "Previous run ID")


catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
model_name = dbutils.widgets.get("model_name")
previous_run_id = dbutils.widgets.get("previous_run_id")


if not catalog or not schema or not model_name or not previous_run_id:
    raise ValueError("None of the parameters may be empty")


mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------
# Evaluation
# If this model performs better (has a better test_mean_absolute_error) than the Baseline,
# register it and give it the Challenger label

from mlflow.tracking import MlflowClient

# Retrieve the test_mean_absolute_error metric from the run with "Baseline" alias
client = MlflowClient()

model_fqn = f"{catalog}.{schema}.{model_name}"

try:
    baseline_version = client.get_model_version_by_alias(model_fqn, "Baseline")
except:
    print("No Baseline found. Exiting.")
    dbutils.notebook.exit(1)


baseline_run = client.get_run(baseline_version.run_id)
baseline_mae = baseline_run.data.metrics.get("test_mean_absolute_error")
print("Baseline test_mean_absolute_error: ", baseline_mae)


# get the last training run
last_run = client.get_run(previous_run_id)
test_mae = last_run.data.metrics.get("test_mean_absolute_error")

if test_mae <= baseline_mae:
    # register model and get version
    version = mlflow.register_model(f"runs:/{previous_run_id}/model", model_fqn)
    # set challenger alias
    client.set_registered_model_alias(
        name=model_fqn, version=version.version, alias="Challenger"
    )
