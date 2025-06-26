from mlflow.tracking import MlflowClient
import mlflow
import json 
import dagshub

dagshub.init(repo_owner='vinayak910', repo_name='mlops-mini', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/vinayak910/mlops-mini.mlflow")
experiment = mlflow.get_experiment_by_name("02 Model Comparisons")
experiment_id = experiment.experiment_id
all_runs = mlflow.search_runs([experiment_id])

# Filter only parent runs (no parentRunId tag)
parent_runs = all_runs[~all_runs["tags.mlflow.parentRunId"].notnull()]

# Get the latest parent run
latest_parent = parent_runs.sort_values(by="start_time", ascending=False).iloc[0]
parent_id = latest_parent.run_id



# Get child runs of the latest parent
child_runs = mlflow.search_runs(
    experiment_ids=[experiment_id],
    filter_string=f"tags.mlflow.parentRunId = '{parent_id}'"
)

top_3 = child_runs.sort_values(by="metrics.f1_score", ascending=False).head(3)

top_models_dict = {}

for i, (_, row) in enumerate(top_3.iterrows(), start=1):
    rank_key = f"top_{i}"
    top_models_dict[rank_key] = {
        "model": row["params.Algorithm"],
        "vectorizer": row["params.vectorizer"],
        "f1_score": row["metrics.f1_score"],
        "run_id": row["run_id"]
    }

import json 
# Save to JSON
with open("top_models.json", "w") as f:
    json.dump(top_models_dict, f, indent=4)

print("✅ Saved top 3 ranked models in dictionary format to top_models.json")