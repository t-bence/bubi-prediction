import argparse
import logging

import mlflow
from mlflow.tracking import MlflowClient
from pyspark.sql import SparkSession

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


spark = SparkSession.builder.getOrCreate()

parser = argparse.ArgumentParser(description="Promotion script arguments")
parser.add_argument("--catalog", type=str, required=True, help="Catalog name")
parser.add_argument("--schema", type=str, required=True, help="Schema name")
parser.add_argument("--model_name", type=str, required=True, help="Model name")
parser.add_argument(
    "--experiment_name", type=str, required=True, help="Experiment name"
)
args = parser.parse_args()

catalog = args.catalog
schema = args.schema
model_name = args.model_name
experiment_name = args.experiment_name


if not catalog or not schema or not model_name or not experiment_name:
    raise ValueError("None of the parameters may be empty")


mlflow.set_registry_uri("databricks-uc")

# Evaluation
# If this model performs better (has a better test_mean_absolute_error) than the Baseline,
# register it and give it the Challenger label


# Retrieve the test_mean_absolute_error metric from the run with "Baseline" alias
client = MlflowClient()

model_fqn = f"{catalog}.{schema}.{model_name}"

try:
    baseline_version = client.get_model_version_by_alias(model_fqn, "Baseline")
except Exception:
    logger.info("No Baseline found. Exiting.")
    import sys

    sys.exit(1)


baseline_run = client.get_run(baseline_version.run_id)
baseline_mae = baseline_run.data.metrics.get("test_mae")
logger.info(f"Baseline test_mae: {baseline_mae}")


# get the last training run
exp = client.get_experiment_by_name(experiment_name)
run = client.search_runs(
    experiment_ids=[exp.experiment_id], max_results=1, order_by=["start_time DESC"]
)[0]

last_run = client.get_run(run.info.run_id)
test_mae = last_run.data.metrics.get("test_mae")
logger.info(f"Last run MAE: {test_mae}")

if test_mae <= baseline_mae:
    # register model and get version
    version = mlflow.register_model(f"runs:/{run.info.run_id}/model", model_fqn)
    # set challenger alias
    client.set_registered_model_alias(
        name=model_fqn, version=version.version, alias="Challenger"
    )
