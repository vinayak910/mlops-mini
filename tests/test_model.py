import unittest
import mlflow 
import os 
import pickle
import pandas as pd 
import numpy as np 

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

        mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")

        cls.model_name =  "my_model"

        cls.model_version = cls.get_latest_model_version(cls.model_name)

        cls.model_uri = f"models:/{cls.model_name}/{cls.model_version}"

        cls.model = mlflow.pyfunc.load_model(cls.model_uri)

        cls.vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))



    @staticmethod
    def get_latest_model_version(model_name):
        client = mlflow.MlflowClient()
        latest_version = client.get_latest_versions(model_name, stages=['Staging'])
        return latest_version[0].version if latest_version else None
    
    def test_model_loaded_properly(self):
        self.assertIsNotNone(self.model)


    def test_model_signature(self):

        input_text = "hi how are you"
        input_data = self.vectorizer.transform([input_text])

        input_df = pd.DataFrame(input_data.toarray(), columns = [str[i] for i in range(input_data.shape[1])])

        prediction = self.model.predict(input_df)

        self.assertEqual(input_df.shape[1], len(self.vectorizer.get_feature_names_out()))

        self.assertEqual(len(prediction), input_df.shape[0])

        self.assertEqual(len(prediction.shape), 1)

if __name__ == "__main__":
    unittest.main()