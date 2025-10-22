import argparse

import mlflow
from mlflow.tracking import MlflowClient
from pyspark.sql import SparkSession

from includes.utilities import configure_logger, get_table_name


def run_challenger_validation(
    catalog: str, schema: str, model_name: str, force_promotion_string: str = "False"
) -> None:
    spark = SparkSession.builder.getOrCreate()
    logger = configure_logger()

    force_promotion = force_promotion_string.lower() == "true"

    mlflow.set_registry_uri("databricks-uc")
    client = MlflowClient()
    model_fqn = f"{catalog}.{schema}.{model_name}"
    logger.info(f"Using model {model_fqn}")
    # get challenger version
    try:
        challenger_version = client.get_model_version_by_alias(model_fqn, "Challenger")
    except Exception:
        logger.info("No Challenger version exists")
        return
    challenger_run = client.get_run(challenger_version.run_id)
    logger.info(
        f"Found challenger run {challenger_run.info.run_name} ({challenger_run.info.run_id})"
    )
    challenger_mae = challenger_run.data.metrics.get("test_mae")
    logger.info(f"Challenger MAE: {challenger_mae}")
    # get champion version
    try:
        champion_version = client.get_model_version_by_alias(model_fqn, "Champion")
    except Exception:
        logger.info("No Champion exists, continue validation")
        metric_mae_passed = True
    else:
        champion_run = client.get_run(champion_version.run_id)
        champion_mae = champion_run.data.metrics.get("test_mae")
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
        model_uri = f"models:/{model_fqn}@Challenger"
        challenger_model = mlflow.pyfunc.load_model(model_uri)
        logger.info(f"Loaded model from {model_uri}")
        validation_df = spark.read.table(
            get_table_name(catalog, schema, "gold")
        ).toPandas()
        preds = challenger_model.predict(validation_df)

        logger.info(preds[:10])
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
    elif force_promotion:
        logger.info("Promotion to Champion forced")
        client.set_registered_model_alias(
            name=model_fqn, version=challenger_version.version, alias="Champion"
        )
    else:
        logger.info("No promotion this time.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Challenger validation arguments")
    parser.add_argument("--catalog", type=str, required=True, help="Catalog name")
    parser.add_argument("--schema", type=str, required=True, help="Schema name")
    parser.add_argument("--model_name", type=str, required=True, help="Model name")
    parser.add_argument(
        "--force_promotion", type=str, required=True, help="Force promotion"
    )
    args = parser.parse_args()
    run_challenger_validation(
        args.catalog, args.schema, args.model_name, args.force_promotion
    )
