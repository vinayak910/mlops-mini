import unittest
import mlflow 
import os 


class TestModelLoading(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        
        dagshub_token = os.getenv('DAGSHUB_TOKEN')

        if not dagshub_token:
            raise EnvironmentError("DAGSHUB_TOKEN environment is not set")
        
        os.environ['MLFLOW_TRACKING_USERNAME'] = dagshub_token
        os.environ['MLFLOW_TRACKING_PASSWORD'] = dagshub_token 
        dagshub_url = "https://dagshub.com"
        repo_owner = "vinayak910"
        repo_name = "mlops-mini"

        mlflow.set_tracking_uri(f"{dagshub_token}/{repo_owner}/{repo_name}.mlflow")

        cls.model_name =  "my_model"

        cls.model_version = cls.get_latest_model_version(cls.model_name)

        cls.model_uri = f"models:/{cls.model_name}/{cls.model_version}"

        cls.model = mlflow.pyfunc.load_model(cls.model_uri)


    @staticmethod
    def get_latest_model_version(model_name):
        client = mlflow.MlflowClient()
        latest_version = client.get_latest_versions(model_name, stages=['Staging'])
        return latest_version[0].version if latest_version else None
    
    def test_model_loaded_properly(self):
        self.assertIsNotNone(self.model)

if __name__ == "__main__":
    unittest.main()