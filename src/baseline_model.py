import argparse
import logging
from datetime import datetime

import mlflow
import pyspark.sql.functions as F
from mlflow.models import infer_signature
from pyspark.sql import SparkSession
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error

from includes.utilities import get_table_name, model_version_with_alias_exists


def run_baseline(
    catalog: str,
    schema: str,
    experiment_name: str,
    station_id: int,
    train_cutoff: datetime,
    model_name: str,
    force_retrain: bool,
) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)
    spark = SparkSession.builder.getOrCreate()
    client = mlflow.MlflowClient()
    model_fqn = f"{catalog}.{schema}.{model_name}"

    if not model_version_with_alias_exists(model_fqn, "Baseline"):
        logger.info("Baseline model does not exist. Creating a new one...")
    elif force_retrain:
        logger.info("Baseline model already exists. Retraining forced, continue...")
    else:
        logger.info("Baseline model already exists. Exiting...")
        return

    gold_df = spark.read.table(get_table_name(catalog, schema, "gold")).filter(
        F.col("station_id") == int(station_id)
    )
    train_df = gold_df.filter(F.col("ts") <= train_cutoff).toPandas()
    logger.info(f"Number of rows in training set: {train_df.shape[0]}")
    test_df = gold_df.filter(F.col("ts") > train_cutoff).toPandas()
    logger.info(f"Number of rows in test set: {test_df.shape[0]}")
    mlflow.end_run()
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_experiment(experiment_name)
    mlflow.autolog(disable=True)
    with mlflow.start_run() as run:
        mlflow.sklearn.autolog(False)
        train_X = train_df[["ts"]]
        train_y = train_df["bikes"]
        model = DummyRegressor(strategy="mean").fit(train_X, train_y)
        signature = infer_signature(train_X, model.predict(train_X))
        mlflow.sklearn.log_model(model, "model", signature=signature)
        train_mae = mean_absolute_error(train_y, model.predict(train_X))
        mlflow.log_metric("train_mae", train_mae)
        logger.info(f"Train MAE: {round(train_mae, 2)}")
        test_X = test_df[["ts"]]
        test_y = test_df["bikes"]
        test_pred = model.predict(test_X)
        test_mae = mean_absolute_error(test_y, test_pred)
        mlflow.log_metric("test_mae", test_mae)
        logger.info(f"Test MAE: {round(test_mae, 2)}")
        model_details = mlflow.register_model(
            f"runs:/{run.info.run_id}/model", f"{model_fqn}"
        )
        client = mlflow.client.MlflowClient()
        client.set_registered_model_alias(
            name=model_fqn, alias="Baseline", version=model_details.version
        )
        client.set_model_version_tag(
            name=model_fqn,
            version=model_details.version,
            key="test_mae",
            value=str(round(test_mae, 2)),
        )
        client.set_model_version_tag(
            name=model_fqn,
            version=model_details.version,
            key="type",
            value="dummy",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline model training arguments")
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
    parser.add_argument("--force_retrain", action="store_true", help="Force retrain")
    args = parser.parse_args()
    train_cutoff = datetime.fromisoformat(args.train_cutoff)
    run_baseline(
        args.catalog,
        args.schema,
        args.experiment_name,
        args.station_id,
        train_cutoff,
        args.model_name,
        args.force_retrain,
    )
