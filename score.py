import mlflow # type: ignore

with mlflow.start_run(run_name="Model Scoring") as run:
    mlflow.log_metric("r2_score", 0.85)
    print(f"Model Scoring run-id: {run.info.run_id}")
