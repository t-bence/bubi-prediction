import argparse
import logging

import mlflow
import pyspark.sql.functions as F
from mlflow.tracking import MlflowClient
from pyspark.sql import SparkSession

from includes.utilities import get_table_name


def run_challenger_validation(catalog: str, schema: str, model_name: str) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)
    spark = SparkSession.builder.getOrCreate()
    if not catalog or not schema or not model_name:
        logger.warning("None of the parameters may be empty. Exiting early.")
        return
    mlflow.set_registry_uri("databricks-uc")
    client = MlflowClient()
    model_fqn = f"{catalog}.{schema}.{model_name}"
    # get challenger version
    try:
        challenger_version = client.get_model_version_by_alias(model_fqn, "Challenger")
    except Exception:
        logger.info("No Challenger version exists")
        return
    challenger_run = client.get_run(challenger_version.run_id)
    challenger_mae = challenger_run.data.metrics.get("test_mean_absolute_error")
    logger.info(f"Challenger MAE: {challenger_mae}")
    # get champion version
    try:
        champion_version = client.get_model_version_by_alias(model_fqn, "Champion")
    except Exception:
        logger.info("No Champion exists, continue validation")
        metric_mae_passed = True
    else:
        champion_run = client.get_run(champion_version.run_id)
        champion_mae = champion_run.data.metrics.get("test_mean_absolute_error")
        logger.info(f"Champion MAE: {champion_mae}")
        metric_mae_passed = challenger_mae < champion_mae
        logger.info(f"MAE metric passed: {metric_mae_passed}")
    client.set_model_version_tag(
        model_fqn,
        challenger_version.version,
        key="metric_mae_passed",
        value=metric_mae_passed,
    )
    # determine if the challenger model can make predictions
    try:
        model_uri = f"models:/{model_name}@Challenger"
        challenger_model = mlflow.pyfunc.spark_udf(spark, model_uri)
        validation_df = spark.read.table(
            get_table_name(catalog, schema, "gold")
        ).select(F.col("ts").alias("ds"))
        feature_columns = [F.col(column) for column in validation_df.columns]
        preds = validation_df.withColumn(
            "prediction", challenger_model(F.struct(feature_columns))
        )
        preds.show()
        can_predict = True
    except Exception as e:
        logger.info("Unable to predict on features.")
        logger.info(e)
        can_predict = False
    client.set_model_version_tag(
        model_fqn,
        version=challenger_version.version,
        key="can_predict",
        value=can_predict,
    )
    if metric_mae_passed and can_predict:
        logger.info("Promoting version to Champion")
        client.set_registered_model_alias(
            name=model_fqn, version=challenger_version.version, alias="Champion"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Challenger validation arguments")
    parser.add_argument("--catalog", type=str, required=True, help="Catalog name")
    parser.add_argument("--schema", type=str, required=True, help="Schema name")
    parser.add_argument("--model_name", type=str, required=True, help="Model name")
    args = parser.parse_args()
    run_challenger_validation(args.catalog, args.schema, args.model_name)
