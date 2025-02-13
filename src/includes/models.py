import mlflow

def get_latest_model_version(model_name: str) -> str:
    versions = (mlflow.client.MlflowClient()
        .search_model_versions(f"name='{model_name}'")
    )
    # version is a str, convert it to int for finding the max,
    # but return the string!
    max_version = max(versions, lambda v: int(v.version))
    return max_version.version
