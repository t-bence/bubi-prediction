# Databricks notebook source

import mlflow
from pyspark.sql import SparkSession
import pyspark.sql.functions as F

from includes.utilities import get_full_name

spark = SparkSession.builder.getOrCreate()

dbutils.widgets.text("catalog", "", "Catalog")
dbutils.widgets.text("schema", "", "Schema")
dbutils.widgets.text("experiment_name", "", "Experiment name")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
experiment_name = dbutils.widgets.get("experiment_name")

if not catalog or not schema or not experiment_name:
	raise ValueError("Catalog, Schema, and Experiment name must not be empty")


def get_latest_model_version(model_name):
    latest_version = 1
    mlflow_client = mlflow.MlflowClient()
    for mv in mlflow_client.search_model_versions(f"name='{model_name}'"):
        version_int = int(mv.version)
        if version_int > latest_version:
            latest_version = version_int
    return latest_version

mlflow.set_registry_uri("databricks-uc")

# End any existing runs (in the case this notebook is being run for a second time)
mlflow.end_run()

# Start an mlflow run, which is needed to log the model
mlflow.start_run()


# Load gold table
gold_df = (spark.read.table(get_full_name(catalog, schema, "gold")))

# COMMAND ----------
# MAGIC %md
# MAGIC # Training
# MAGIC 
# MAGIC Train a model to predict the number of bikes. 
# MAGIC Currently, a Prophet model is trained that can only take into account the time series,
# MAGIC or features that are **known into the future**. So I will not use the other bike numbers.
# MAGIC
# MAGIC Main source: <https://www.databricks.com/blog/2020/01/27/time-series-forecasting-prophet-spark.html>
# MAGIC
# MAGIC Let's train for one station only first

# COMMAND ----------

from prophet import Prophet, serialize
import mlflow.prophet
from mlflow.models import infer_signature

mlflow.autolog(disable=True)

# TODO: make it parametric
mlflow.set_experiment(experiment_name)

mlflow.end_run()

KRISZTINA_TER_ID = 2100

krisztina_pd = (gold_df
    .filter(F.col("station_id") == KRISZTINA_TER_ID)
    .selectExpr("ts AS ds", "bikes AS y")
    .toPandas()
)

def extract_params(pr_model):
    params = {attr: getattr(pr_model, attr) for attr in serialize.SIMPLE_ATTRIBUTES}
    return {k: v for k, v in params.items() if isinstance(v, (int, float, str, bool))}

with mlflow.start_run() as run:
    model = Prophet()

    model.fit(krisztina_pd)

    params = extract_params(model)

    # Prepare future dataframe for prediction
    future = model.make_future_dataframe(periods=10, freq='D')
    forecast = model.predict(future)

    # Infer model signature
    signature = infer_signature(future, forecast)
    model_info = mlflow.prophet.log_model(
         model,
         artifact_path="prophet_model",
         signature=signature,
         input_example=krisztina_pd[["ds"]].head(10)
    )
    mlflow.log_params(params)

# configure predictions
future_pd = model.make_future_dataframe(
    periods=90,
    freq='d',
    include_history=True
)

# make predictions
results_pd = model.predict(future_pd)

# . . .

# return predictions
display(results_pd)