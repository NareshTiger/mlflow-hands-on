import mlflow # type: ignore
import subprocess

mlflow.set_experiment("Housing_Price_Prediction")

with mlflow.start_run(run_name="Parent Run") as parent_run:
    parent_run_id = parent_run.info.run_id

    # Start Child Runs
    with mlflow.start_run(run_name="Data Preparation", nested=True) as child1:
        subprocess.run(["python", "mlflow-hands-on/data_prep.py"])

    with mlflow.start_run(run_name="Model Training", nested=True) as child2:
        subprocess.run(["python", "mlflow-hands-on/train.py"])

    with mlflow.start_run(run_name="Model Scoring", nested=True) as child3:
        subprocess.run(["python", "mlflow-hands-on/score.py"])
    print(f"Parent Run ID: {parent_run_id}")
