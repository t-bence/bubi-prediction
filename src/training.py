# Databricks notebook source

import mlflow
import mlflow.cli
from pyspark.sql import SparkSession
import pyspark.sql.functions as F

from includes.utilities import get_table_name

spark = SparkSession.builder.getOrCreate()

dbutils.widgets.text("catalog", "", "Catalog")
dbutils.widgets.text("schema", "", "Schema")
dbutils.widgets.text("experiment_name", "", "Experiment name")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
experiment_name = dbutils.widgets.get("experiment_name")

if not catalog or not schema or not experiment_name:
	raise ValueError("Catalog, Schema, and Experiment name must not be empty")


mlflow.set_registry_uri("databricks-uc")

# Load gold table
gold_df = (spark.read.table(get_table_name(catalog, schema, "gold")))

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

mlflow.end_run()

KRISZTINA_TER_ID = 2100

def forecast_station_bikes(key: tuple, history_pd):
    """The function doing the training.
    Tuple `key` will contain the grouping key, `station_id`."""
    import mlflow
    import mlflow.prophet
    from prophet import Prophet, serialize
    from mlflow.models import infer_signature
    import pandas as pd

    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_experiment(experiment_name)
    mlflow.autolog(disable=True)

    station_id = int(key[0])

    def extract_params(pr_model):
        params = {attr: getattr(pr_model, attr) for attr in serialize.SIMPLE_ATTRIBUTES}
        return {k: v for k, v in params.items() if isinstance(v, (int, float, str, bool))}


    with mlflow.start_run() as run:
        model = Prophet()

        model.fit(history_pd)

        params = extract_params(model)

        # Prepare future dataframe for prediction
        future_date = pd.DataFrame({'ds': ['2025-02-13 12:00:00']})
        future_date['ds'] = pd.to_datetime(future_date['ds'])

        # Predict
        forecast = model.predict(future_date)

        # Infer model signature
        signature = infer_signature(future_date, forecast)
        model_info = mlflow.prophet.log_model(
            model,
            artifact_path="prophet_model",
            signature=signature,
            input_example=history_pd[["ds"]].head(10)
        )
        mlflow.log_params(params)
        mlflow.set_tag(key="station_id", value=station_id)

        forecast_result = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        forecast_result.insert(0, 'station_id', station_id)
        return forecast_result
    
result_schema = 'station_id INTEGER, ds TIMESTAMP, yhat FLOAT, yhat_lower FLOAT, yhat_upper FLOAT'

bikes_per_station = (gold_df
    .selectExpr("ts AS ds", "bikes AS y", "station_id")
    .groupBy("station_id")
    .applyInPandas(forecast_station_bikes, schema=result_schema)
    .sort("station_id")
)

bikes_per_station.display()
