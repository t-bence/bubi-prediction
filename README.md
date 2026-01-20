# bubi_prediction

This is a project to predict the number of bikes at the Budapest public bike sharing stations called Bubi.

## TODO

- Add hyperparameter tuning.
- Monitoring table should only contain the single station I'm currently predicting for.
- The challenger - champion check should only run if a new challenger was created.

## Target environment

Serverless environment version 4 <https://docs.databricks.com/aws/en/release-notes/serverless/environment-version/four>

## Notes

The current logic is for a classical model, not for time series prediction. In this case, we should probably register a new model every day, and use that, no matter what. The fact that we registered a baseline model a long time ago should not mean that we don't overwrite that for example, because since then, we have a lot more data to train on.

I think this setup should be rewritten to use XGBoost for prediction with some features
