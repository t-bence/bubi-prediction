import argparse
import logging

import mlflow
from mlflow.tracking import MlflowClient


def run_promotion(
    catalog: str, schema: str, model_name: str, experiment_name: str, target: str
) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    mlflow.set_registry_uri("databricks-uc")
    client = MlflowClient()
    model_fqn = f"{catalog}.{schema}.{model_name}"

    try:
        baseline_version = client.get_model_version_by_alias(model_fqn, "Baseline")
    except Exception:
        logger.info("No Baseline found. Exiting.")
        return
    baseline_run = client.get_run(baseline_version.run_id)
    baseline_mae = baseline_run.data.metrics.get("test_mae")
    logger.info(f"Baseline test_mae: {baseline_mae}")
    exp = client.get_experiment_by_name(experiment_name)
    run = client.search_runs(
        experiment_ids=[exp.experiment_id], max_results=1, order_by=["start_time DESC"]
    )[0]
    last_run = client.get_run(run.info.run_id)
    test_mae = last_run.data.metrics.get("test_mae")
    logger.info(f"Last run MAE: {test_mae}")
    if test_mae <= baseline_mae:
        logger.info("Test MAE is <= than baseline MAE, so registering the model")
        version = mlflow.register_model(f"runs:/{run.info.run_id}/model", model_fqn)
        client.set_registered_model_alias(
            name=model_fqn, version=version.version, alias="Challenger"
        )
    elif target.startswith("dev"):
        logger.info("Target starts with 'dev', so registering the model")
        version = mlflow.register_model(f"runs:/{run.info.run_id}/model", model_fqn)
        client.set_registered_model_alias(
            name=model_fqn, version=version.version, alias="Challenger"
        )
    else:
        logger.info(
            "Test MAE is > than baseline MAE, so we are not registering the model"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Promotion script arguments")
    parser.add_argument("--catalog", type=str, required=True, help="Catalog name")
    parser.add_argument("--schema", type=str, required=True, help="Schema name")
    parser.add_argument("--model_name", type=str, required=True, help="Model name")
    parser.add_argument(
        "--experiment_name", type=str, required=True, help="Experiment name"
    )
    parser.add_argument("--target", type=str, required=True, help="Environment name")
    args = parser.parse_args()
    run_promotion(
        args.catalog, args.schema, args.model_name, args.experiment_name, args.target
    )
