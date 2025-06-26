import mlflow
import json
import dagshub
from mlflow.tracking import MlflowClient

# Initialize DAGsHub tracking
dagshub.init(repo_owner='vinayak910', repo_name='mlops-mini', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/vinayak910/mlops-mini.mlflow")

client = MlflowClient()

# Get Experiment 3
experiment = mlflow.get_experiment_by_name("03 Hyperparameter Tuning")
experiment_id = experiment.experiment_id
all_runs = mlflow.search_runs([experiment_id])
parent_runs = all_runs[~all_runs["tags.mlflow.parentRunId"].notnull()]
latest_parent = parent_runs.sort_values(by="start_time", ascending=False).iloc[0]
parent_id = latest_parent.run_id
# === Step 1: Fetch the latest parent run ===
child_runs = mlflow.search_runs(
    experiment_ids=[experiment_id],
    filter_string=f"tags.mlflow.parentRunId = '{parent_id}'"
)
top_1 = child_runs.sort_values(by="metrics.f1_score", ascending=False).head(1)

top_models_dict = {}

for i, (_, row) in enumerate(top_1.iterrows(), start=1):
    rank_key = f"top_{i}"
    top_models_dict[rank_key] = {
        "model": row["params.model"],
        "vectorizer": row["params.vectorizer"],
        "f1_score": row["metrics.f1_score"],
        "run_id": row["run_id"]
    }

with open("best_model.json", "w") as f:
    json.dump(top_models_dict, f, indent=4)

print("✅ Saved best model config to best_run_config.json")
