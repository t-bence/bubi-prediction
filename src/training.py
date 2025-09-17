import argparse
from datetime import datetime

import holidays
import mlflow
import pandas as pd
import pyspark.sql.functions as F
from mlflow.models import infer_signature
from pyspark.sql import SparkSession
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from xgboost import XGBRegressor

from includes.utilities import configure_logger, get_table_name


def run_training(
    catalog: str,
    schema: str,
    experiment_name: str,
    station_id: int,
    train_cutoff: datetime,
    model_name: str,
) -> None:
    spark: SparkSession = SparkSession.builder.getOrCreate()

    logger = configure_logger()

    logger.info(
        f"Parameters: catalog={catalog}, schema={schema}, experiment_name={experiment_name}, "
        f"station_id={station_id}, train_cutoff={train_cutoff}, model_name={model_name}"
    )

    mlflow.set_registry_uri("databricks-uc")

    # Load gold table
    gold_df = spark.read.table(get_table_name(catalog, schema, "gold")).filter(
        F.col("station_id") == int(station_id)
    )

    train_pdf = gold_df.filter(F.col("ts") <= train_cutoff).toPandas()
    logger.info(f"Number of rows in training set: {train_pdf.count()}")

    test_pdf = gold_df.filter(F.col("ts") > train_cutoff).toPandas()
    logger.info(f"Number of rows in test set: {test_pdf.shape[0]}")

    mlflow.end_run()
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_experiment(experiment_name)
    mlflow.autolog(disable=True)

    def extract_time_features(X, country="HU"):
        df = X.copy()
        ts = pd.to_datetime(df["ts"])
        hu_holidays = holidays.country_holidays(country)
        df["hour"] = ts.dt.hour
        df["minute"] = ts.dt.minute
        df["day_of_week"] = ts.dt.dayofweek
        df["month"] = ts.dt.month
        df["is_holiday"] = ts.dt.date.astype(str).isin(hu_holidays).astype(int)
        return df[["hour", "minute", "day_of_week", "month", "is_holiday"]]

    pipeline = Pipeline(
        [
            (
                "time_features",
                FunctionTransformer(extract_time_features, kw_args={"country": "HU"}),
            ),
            ("xgb", XGBRegressor()),
        ]
    )

    X_train = train_pdf[["ts"]]
    y_train = train_pdf["bikes"]

    X_test = test_pdf[["ts"]]
    y_test = test_pdf["bikes"]

    with mlflow.start_run() as run:
        pipeline.fit(X_train, y_train)

        # Infer model signature
        signature = infer_signature(
            X_train.head(10), pipeline.predict(X_train.head(10))
        )
        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model",
            signature=signature,
            input_example=X_train.head(10),
        )
        mlflow.set_tag(key="station_id", value=station_id)

        # Evaluate the model on training and test data
        train_mae = mean_absolute_error(y_train, pipeline.predict(X_train))
        mlflow.log_metric("train_mae", train_mae)
        logger.info(f"Train MAE: {round(train_mae, 2)}")
        test_mae = mean_absolute_error(y_test, pipeline.predict(X_test))
        mlflow.log_metric("test_mae", test_mae)
        logger.info(f"Test MAE: {round(test_mae, 2)}")
        logger.info(f"Run ID: {run.info.run_id}")


if __name__ == "__main__":
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
    train_cutoff = datetime.fromisoformat(args.train_cutoff)
    run_training(
        args.catalog,
        args.schema,
        args.experiment_name,
        args.station_id,
        train_cutoff,
        args.model_name,
    )
