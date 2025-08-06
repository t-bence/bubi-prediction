import argparse
import logging
from datetime import datetime

import mlflow
import mlflow.prophet
import pandas as pd
import pyspark.sql.functions as F
from mlflow.models import infer_signature
from prophet import Prophet, serialize
from pyspark.sql import SparkSession
from sklearn.metrics import mean_absolute_error

from includes.utilities import get_table_name

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

spark = SparkSession.builder.getOrCreate()

parser = argparse.ArgumentParser(description="Training script arguments")
parser.add_argument("--catalog", type=str, required=True, help="Catalog name")
parser.add_argument("--schema", type=str, required=True, help="Schema name")
parser.add_argument(
    "--experiment_name", type=str, required=True, help="Experiment name"
)
parser.add_argument("--station_id", type=int, required=True, help="Station ID")
parser.add_argument(
    "--train_cutoff", type=str, required=True, help="Train cutoff (ISO format)"
)
parser.add_argument("--model_name", type=str, required=True, help="Model name")
args = parser.parse_args()

catalog = args.catalog
schema = args.schema
experiment_name = args.experiment_name
station_id = args.station_id
train_cutoff = datetime.fromisoformat(args.train_cutoff)
model_name = args.model_name
logger.info(
    f"Parameters: catalog={catalog}, schema={schema}, experiment_name={experiment_name}, "
    f"station_id={station_id}, train_cutoff={train_cutoff}, model_name={model_name}"
)

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

train_df = gold_df.filter(F.col("ts") <= train_cutoff).toPandas()

logger.info(f"Number of rows in training set: {train_df.shape[0]}")

test_df = gold_df.filter(F.col("ts") > train_cutoff).toPandas()

logger.info(f"Number of rows in test set: {test_df.shape[0]}")


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

    # Evaluate the model on training and test data
    train_mae = mean_absolute_error(
        train_df["y"], model.predict(train_df[["ds"]]).loc[:, "yhat"]
    )
    mlflow.log_metric("train_mae", train_mae)
    logger.info(f"Train MAE: {round(train_mae, 2)}")

    test_mae = mean_absolute_error(
        test_df["y"], model.predict(test_df[["ds"]]).loc[:, "yhat"]
    )
    mlflow.log_metric("test_mae", test_mae)
    logger.info(f"Test MAE: {round(test_mae, 2)}")


logger.info(f"Run ID: {run.info.run_id}")
