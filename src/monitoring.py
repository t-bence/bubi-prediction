import argparse

from pyspark.sql import SparkSession

from includes.utilities import configure_logger


def run_monitoring(catalog: str, schema: str) -> None:
    spark = SparkSession.builder.getOrCreate()
    logger = configure_logger()

    gold_table_fqn = f"{catalog}.{schema}.gold"

    # create the inference table if it does not exist
    inference_table_fqn = f"{catalog}.{schema}.predictions"
    logger.info(f"Using predictions table {inference_table_fqn}")

    monitoring_table_fqn = f"{catalog}.{schema}.monitoring"

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
    logger.info("Written monitoring data")

    import os

    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.catalog import (
        MonitorInferenceLog,
        MonitorInferenceLogProblemType,
    )

    print(f"Creating monitor for inference table {monitoring_table_fqn}")
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

    except Exception as exception:
        if "already exist" in str(exception).lower():
            print(
                f"Monitor for {monitoring_table_fqn} already exists, retrieving monitor info:"
            )
            info = w.quality_monitors.get(table_name=monitoring_table_fqn)
        else:
            logger.info(exception)
            raise exception

    import time

    from databricks.sdk.service.catalog import (
        MonitorInfoStatus,
        MonitorRefreshInfoState,
    )

    # Wait for monitor to be created
    while info.status == MonitorInfoStatus.MONITOR_STATUS_PENDING:
        info = w.quality_monitors.get(table_name=monitoring_table_fqn)
        time.sleep(10)

    assert (
        info.status == MonitorInfoStatus.MONITOR_STATUS_ACTIVE
    ), "Error creating monitor"

    def get_refreshes():
        return w.quality_monitors.list_refreshes(
            table_name=monitoring_table_fqn
        ).refreshes

    refreshes = get_refreshes()
    if len(refreshes) == 0:
        w.quality_monitors.run_refresh(table_name=monitoring_table_fqn)
        time.sleep(5)
        refreshes = get_refreshes()

    run_info = refreshes[0]
    while run_info.state in (
        MonitorRefreshInfoState.PENDING,
        MonitorRefreshInfoState.RUNNING,
    ):
        run_info = w.quality_monitors.get_refresh(
            table_name=monitoring_table_fqn, refresh_id=run_info.refresh_id
        )
        logger.info(f"Waiting for refresh to complete {run_info.state}...")
        time.sleep(30)

    assert run_info.state == MonitorRefreshInfoState.SUCCESS, "Monitor refresh failed"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitoring arguments")
    parser.add_argument("--catalog", type=str, required=True, help="Catalog name")
    parser.add_argument("--schema", type=str, required=True, help="Schema name")
    args = parser.parse_args()
    run_monitoring(args.catalog, args.schema)
