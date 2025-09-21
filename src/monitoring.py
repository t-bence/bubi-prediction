import argparse

from pyspark.sql import SparkSession

from includes.utilities import configure_logger


def run_monitoring(catalog: str, schema: str) -> None:
    logger = configure_logger()
    logger.info(f"Starting monitoring for catalog={catalog}, schema={schema}")
    spark = SparkSession.builder.getOrCreate()
    gold_table_fqn = f"{catalog}.{schema}.gold"
    inference_table_fqn = f"{catalog}.{schema}.predictions"
    monitoring_table_fqn = f"{catalog}.{schema}.monitoring"
    logger.info(
        f"Joining gold and predictions tables to create monitoring table: {monitoring_table_fqn}"
    )
    (
        spark.table(gold_table_fqn)
        .join(
            spark.table(inference_table_fqn).withColumnRenamed("bikes", "prediction"),
            how="left",
            on="ts",
        )
        .write.mode("overwrite")
        .saveAsTable(monitoring_table_fqn)
    )
    logger.info("Written monitoring data to table.")

    if (
        spark.sql(
            f"SHOW TBLPROPERTIES {monitoring_table_fqn} ('delta.enableChangeDataFeed');"
        )
        .collect()[0]
        .value
        != "true"
    ):
        logger.info("Enabling change data feed on monitoring table.")
        spark.sql(
            f"ALTER TABLE {monitoring_table_fqn} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)"
        )
    else:
        logger.info("Table already has change data feed enabled.")

    import os

    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.catalog import (
        MonitorInferenceLog,
        MonitorInferenceLogProblemType,
    )

    logger.info(f"Creating monitor for inference table {monitoring_table_fqn}")
    w = WorkspaceClient()
    try:
        info = w.quality_monitors.create(
            table_name=monitoring_table_fqn,
            inference_log=MonitorInferenceLog(
                problem_type=MonitorInferenceLogProblemType.PROBLEM_TYPE_REGRESSION,
                prediction_col="prediction",
                timestamp_col="prediction_date",
                granularities=["1 day", "1 week"],
                model_id_col="model_version",
                label_col="bikes",  # optional
            ),
            assets_dir=f"{os.getcwd()}/monitoring",  # Change this to another folder of choice if needed
            output_schema_name=f"{catalog}.{schema}",
            # baseline_table_name=f"{catalog}.{schema}.monitor_baseline",
            slicing_exprs=[],  # Slicing dimension
        )
        logger.info("Monitor creation request sent.")
    except Exception as exception:
        if "already exist" in str(exception).lower():
            logger.info(
                f"Monitor for {monitoring_table_fqn} already exists, retrieving monitor info."
            )
            info = w.quality_monitors.get(table_name=monitoring_table_fqn)
        else:
            logger.error(f"Error creating monitor: {exception}")
            raise exception

    import time

    from databricks.sdk.service.catalog import MonitorInfoStatus

    logger.info("Waiting for monitor to become active...")
    while info.status == MonitorInfoStatus.MONITOR_STATUS_PENDING:
        logger.info("Monitor status: PENDING. Sleeping 10 seconds...")
        info = w.quality_monitors.get(table_name=monitoring_table_fqn)
        time.sleep(10)

    if info.status == MonitorInfoStatus.MONITOR_STATUS_ACTIVE:
        logger.info("Monitor is now ACTIVE.")
    else:
        logger.error("Error creating monitor: status is not ACTIVE.")
        raise RuntimeError("Error creating monitor")

    logger.info("Monitor refresh started.")
    w.quality_monitors.run_refresh(table_name=monitoring_table_fqn)
    logger.info("Exiting.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitoring arguments")
    parser.add_argument("--catalog", type=str, required=True, help="Catalog name")
    parser.add_argument("--schema", type=str, required=True, help="Schema name")
    args = parser.parse_args()
    run_monitoring(args.catalog, args.schema)
