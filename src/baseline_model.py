# Databricks notebook source
from datetime import datetime

import mlflow
import mlflow.sklearn
import pyspark.sql.functions as F
from pyspark.dbutils import dbutils
from pyspark.sql import SparkSession
from sklearn.dummy import DummyClassifier

from includes.utilities import get_table_name

spark = SparkSession.builder.getOrCreate()

dbutils.widgets.text("catalog", "", "Catalog")
dbutils.widgets.text("schema", "", "Schema")
dbutils.widgets.text("experiment_name", "", "Experiment name")
dbutils.widgets.text("station_id", "", "Station ID")
dbutils.widgets.text("train_cutoff", "", "Train cutoff")
dbutils.widgets.text("model_name", "", "Model name")
dbutils.widgets.text("force_retrain", "False", "Force retrain")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
experiment_name = dbutils.widgets.get("experiment_name")
station_id = int(dbutils.widgets.get("station_id"))
train_cutoff = datetime.fromisoformat(dbutils.widgets.get("train_cutoff"))
model_name = dbutils.widgets.get("model_name")
force_retrain = dbutils.widgets.get("force_retrain").lower() == "true"


if (
    not catalog
    or not schema
    or not experiment_name
    or not station_id
    or not train_cutoff
    or not model_name
):
    raise ValueError("None of the parameters may be empty")


mlflow.set_registry_uri("databricks-uc")

client = mlflow.MlflowClient()

model_fqn = f"{catalog}.{schema}.{model_name}"


try:
    baseline_model = client.get_model_version_by_alias(model_fqn, "Baseline")

except Exception as e:
    if "RESOURCE_DOES_NOT_EXIST" in e.message:
        print("Baseline model does not exist. Creating a new one...")

else:
    if not force_retrain:
        print("Baseline model already exists. Exiting...")
        dbutils.notebook.exit(0)

    else:
        print("Baseline model already exists. Retraining forced, continue...")


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

with mlflow.start_run() as run:
    mlflow.sklearn.autolog(False)

    train_X = train_df[["ds"]]
    train_y = train_df["y"]

    mean_bikes = round(train_y.mean())

    model = DummyClassifier(strategy="constant", constant=mean_bikes).fit(
        train_X, train_y
    )

    mlflow_model = mlflow.models.Model()
    mlflow.pyfunc.add_to_model(mlflow_model, loader_module="mlflow.sklearn")
    pyfunc_model = mlflow.pyfunc.PyFuncModel(model_meta=mlflow_model, model_impl=model)
    training_eval_result = mlflow.evaluate(
        model=pyfunc_model,
        data=train_df,
        targets="y",
        model_type="regressor",
        evaluator_config={
            "log_model_explainability": False,
            "metric_prefix": "train_",
            "pos_label": 1,
        },
    )

    # Log metrics for the test set
    test_eval_result = mlflow.evaluate(
        model=pyfunc_model,
        data=test_df,
        targets="y",
        model_type="regressor",
        evaluator_config={
            "log_model_explainability": False,
            "metric_prefix": "test_",
            "pos_label": 1,
        },
    )
    test_mae = test_eval_result.metrics["test_mean_absolute_error"]

    print(test_mae)


model_details = mlflow.register_model(f"runs:/{run.info.run_id}/model", f"{model_fqn}")

# COMMAND ----------

# Set this version as the Baseline model, using its model alias
mlflow.client.MlflowClient().set_registered_model_alias(
    name=model_fqn, alias="Baseline", version=model_details.version
)
