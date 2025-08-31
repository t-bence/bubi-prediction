import argparse
import datetime as dt

import mlflow
import pyspark.sql.functions as F
from mlflow.tracking import MlflowClient
from pyspark.sql import SparkSession

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
    model = mlflow.prophet.load_model(model_uri)
    logger.info(f"Loaded model from {model_uri}")

    HOURS_IN_DAY = 24
    MINUTES_IN_HOUR = 60
    PREDICTION_PERIOD_MINUTES = 10
    periods = int(HOURS_IN_DAY * MINUTES_IN_HOUR / PREDICTION_PERIOD_MINUTES - 1)

    TEN_MINUTE_IN_SECONDS = "600s"

    pdf = model.make_future_dataframe(
        periods=periods, freq=TEN_MINUTE_IN_SECONDS, include_history=False
    )

    selected_day = str(pdf["ds"][0].date())
    logger.info(f"Selected day: {selected_day}")

    predictions = model.predict(pdf)

    preds_df = (
        spark.createDataFrame(predictions)
        .withColumn("model_version", F.lit(champion_version.version))
        .withColumn("prediction_date", F.lit(dt.datetime.now(dt.UTC)))
    )

    # create the inference table if it does not exist
    inference_table_fqn = model_fqn = f"{catalog}.{schema}.predictions"
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
        ON new.ds = {inference_table_fqn}.ds
        WHEN MATCHED THEN UPDATE SET *
        WHEN NOT MATCHED THEN INSERT *
    """)

    logger.info(f"Written {preds_df.count()} rows.")

    logger.info("Sample data:")
    logger.info(preds_df.select("ds", "yhat").limit(10).show())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Challenger validation arguments")
    parser.add_argument("--catalog", type=str, required=True, help="Catalog name")
    parser.add_argument("--schema", type=str, required=True, help="Schema name")
    parser.add_argument("--model_name", type=str, required=True, help="Model name")
    args = parser.parse_args()
    run_inference(args.catalog, args.schema, args.model_name)
