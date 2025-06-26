https://dagshub.com/vinayak910/mlops-mini.mlflow


import dagshub
dagshub.init(repo_owner='vinayak910', repo_name='mlops-mini', mlflow=True)

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('meetric name', 1)