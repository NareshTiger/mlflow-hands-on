import mlflow # type: ignore

mlflow.set_experiment("Housing_Price_Prediction")

with mlflow.start_run(run_name="Data Preparation") as run:
    mlflow.log_param("data_source", "housing.csv")
    mlflow.log_param("missing_value_handling", "median_imputation")
    print(f"Data Preparation run-id: {run.info.run_id}")
