import argparse
import datetime as dt

import mlflow
from mlflow.tracking import MlflowClient
from pyspark.sql import SparkSession

from includes.inference import (
    create_prediction_time_dataframe,
    create_predictions_spark_dataframe,
)
from includes.utilities import configure_logger


def run_inference(catalog: str, schema: str, model_name: str) -> None:
    spark = SparkSession.builder.getOrCreate()
    logger = configure_logger()

    mlflow.set_registry_uri("databricks-uc")
    client = MlflowClient()
    model_fqn = f"{catalog}.{schema}.{model_name}"
    logger.info(f"Using model {model_fqn}")

    # get champion version
    try:
        champion_version = client.get_model_version_by_alias(model_fqn, "Champion")
    except Exception:
        logger.info("No Champion exists, exiting")
        return

    model_uri = f"models:/{model_fqn}@Champion"
    model = mlflow.pyfunc.load_model(model_uri)
    logger.info(f"Loaded model from {model_uri}")

    pdf = create_prediction_time_dataframe()

    logger.info(f"Start of interval: {pdf['ts'].iloc[0]}")

    logger.info(f"End of interval: {pdf['ts'].iloc[-1]}")

    bikes = model.predict(pdf)

    preds_df = create_predictions_spark_dataframe(
        pdf, bikes, champion_version.version, dt.datetime.now(dt.UTC), spark
    )

    # create the inference table if it does not exist
    inference_table_fqn = f"{catalog}.{schema}.predictions"
    logger.info(f"Using predictions table {inference_table_fqn}")

    if not spark.catalog.tableExists(inference_table_fqn):
        logger.info("Inference table will be created...")
        preds_df.limit(0).write.saveAsTable(inference_table_fqn)

    else:
        logger.info("Inference table already exists")

    logger.info("Writing predictions")

    preds_df.createOrReplaceTempView("new")

    spark.sql(f"""
        MERGE INTO {inference_table_fqn}
        USING new
        ON new.ts = {inference_table_fqn}.ts
        WHEN MATCHED THEN UPDATE SET *
        WHEN NOT MATCHED THEN INSERT *
    """)

    logger.info(f"Written {preds_df.count()} rows.")

    logger.info("Sample data:")
    logger.info(preds_df.limit(10).show())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Challenger validation arguments")
    parser.add_argument("--catalog", type=str, required=True, help="Catalog name")
    parser.add_argument("--schema", type=str, required=True, help="Schema name")
    parser.add_argument("--model_name", type=str, required=True, help="Model name")
    args = parser.parse_args()
    run_inference(args.catalog, args.schema, args.model_name)
