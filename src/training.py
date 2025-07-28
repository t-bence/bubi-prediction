# Databricks notebook source
from datetime import datetime

import mlflow
import mlflow.prophet
import pandas as pd
import pyspark.sql.functions as F
from mlflow.models import infer_signature
from prophet import Prophet, serialize
from pyspark import dbutils
from pyspark.sql import SparkSession

from includes.utilities import get_table_name

spark = SparkSession.builder.getOrCreate()

dbutils.widgets.text("catalog", "", "Catalog")
dbutils.widgets.text("schema", "", "Schema")
dbutils.widgets.text("experiment_name", "", "Experiment name")
dbutils.widgets.text("station_id", "", "Station ID")
dbutils.widgets.text("train_cutoff", "", "Train cutoff")
dbutils.widgets.text("model_name", "", "Model name")


catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
experiment_name = dbutils.widgets.get("experiment_name")
station_id = int(dbutils.widgets.get("station_id"))
train_cutoff = datetime.fromisoformat(dbutils.widgets.get("train_cutoff"))
model_name = dbutils.widgets.get("model_name")


if (
    not catalog
    or not schema
    or not experiment_name
    or not station_id
    or not train_cutoff
):
    raise ValueError("None of the parameters may be empty")


mlflow.set_registry_uri("databricks-uc")

# Load gold table
gold_df = (
    spark.read.table(get_table_name(catalog, schema, "gold"))
    .filter(F.col("station_id") == int(station_id))
    .selectExpr("ts AS ds", "bikes AS y")
)

train_df = gold_df.filter(F.col("ds") <= train_cutoff).toPandas()

test_df = gold_df.filter(F.col("ds") > train_cutoff).toPandas()

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

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(experiment_name)
mlflow.autolog(disable=True)


def extract_params(pr_model):
    params = {attr: getattr(pr_model, attr) for attr in serialize.SIMPLE_ATTRIBUTES}
    return {k: v for k, v in params.items() if isinstance(v, (int, float, str, bool))}


with mlflow.start_run() as run:
    model = Prophet()

    model.fit(train_df)

    params = extract_params(model)

    # Prepare future dataframe for prediction
    future_date = pd.DataFrame({"ds": ["2025-02-13 12:00:00"]})
    future_date["ds"] = pd.to_datetime(future_date["ds"])

    # Predict
    forecast = model.predict(future_date)

    # Infer model signature
    signature = infer_signature(future_date, forecast)
    model_info = mlflow.prophet.log_model(
        model,
        artifact_path="model",
        signature=signature,
        input_example=train_df[["ds"]].head(10),
    )
    mlflow.log_params(params)
    mlflow.set_tag(key="station_id", value=station_id)

    # Evaluate the model on training df
    train_evaluation_result = mlflow.evaluate(
        model=model_info.model_uri,
        data=train_df,
        targets="y",
        predictions="yhat",
        model_type="regressor",
        evaluator_config={
            "log_model_explainability": False,
            "metric_prefix": "train_",
            "pos_label": 1,
        },
    )

    # Evaluate the model on test df
    test_evaluation_result = mlflow.evaluate(
        model=model_info.model_uri,
        data=test_df,
        targets="y",
        predictions="yhat",
        model_type="regressor",
        evaluator_config={
            "log_model_explainability": False,
            "metric_prefix": "test_",
            "pos_label": 1,
        },
    )

    test_mae = test_evaluation_result.metrics["test_mean_absolute_error"]

    print(test_mae)

# forecast_result = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
# forecast_result.insert(0, 'station_id', station_id)

dbutils.jobs.taskValues.set(key="previous_run_id", value=run.info.run_id)
