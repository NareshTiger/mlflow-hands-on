import mlflow # type: ignore

with mlflow.start_run(run_name="Model Training") as run:
    mlflow.log_param("model_type", "Linear Regression")
    mlflow.log_param("train_test_split", "80-20")
    mlflow.log_metric("rmse", 3.45)
    print(f"Model Training run-id: {run.info.run_id}")
