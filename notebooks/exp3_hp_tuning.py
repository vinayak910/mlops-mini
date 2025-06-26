import numpy as np
import re
import mlflow
import pandas as pd
import string
import json
from sklearn.pipeline import Pipeline
from mlflow.tracking import MlflowClient
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import make_pipeline
import dagshub
from sklearn.model_selection import GridSearchCV
import joblib

client = MlflowClient()

# Initialize DAGsHub + MLflow tracking
dagshub.init(repo_owner='vinayak910', repo_name='mlops-mini', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/vinayak910/mlops-mini.mlflow")

# Load the data
df = pd.read_csv('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv')
df.drop(columns=['tweet_id'], inplace=True)

# ===== TEXT CLEANING FUNCTIONS =====
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def lemmatization(text):
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

def remove_stop_words(text):
    return " ".join([word for word in str(text).split() if word not in stop_words])

def removing_numbers(text):
    return ''.join([char for char in text if not char.isdigit()])

def lower_case(text):
    return " ".join([word.lower() for word in text.split()])

def removing_punctuations(text):
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = re.sub('\s+', ' ', text).strip()
    return text

def removing_urls(text):
    return re.sub(r'https?://\S+|www\.\S+', '', text)

def normalize_text(df):
    try:
        df['content'] = df['content'].apply(lower_case)
        df['content'] = df['content'].apply(remove_stop_words)
        df['content'] = df['content'].apply(removing_numbers)
        df['content'] = df['content'].apply(removing_punctuations)
        df['content'] = df['content'].apply(removing_urls)
        df['content'] = df['content'].apply(lemmatization)
        return df
    except Exception as e:
        print(f"Normalization error: {e}")
        raise

# Normalize
df = normalize_text(df)

# Filter for binary classification
df = df[df['sentiment'].isin(['happiness', 'sadness'])]
df['sentiment'] = df['sentiment'].replace({'sadness': 0, 'happiness': 1})

X_data = df['content']
y = df['sentiment']

# ===== Split dataset =====
from sklearn.model_selection import train_test_split
X_train_text, X_test_text, y_train, y_test = train_test_split(X_data, y, test_size=0.2, stratify=y, random_state=42)

# ===== Load Top 3 from Experiment 2 =====
with open("top_models.json", "r") as f:
    top_models_dict = json.load(f)  # No extra key access
model_map = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "MultinomialNB": MultinomialNB(),
    "RandomForest": RandomForestClassifier(),
    "GradientBoosting": GradientBoostingClassifier(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

vectorizer_map = {
    "BOW": CountVectorizer(),
    "TF-IDF": TfidfVectorizer()
}

# ===== Define hyperparameter grids =====
param_grids = {
    "LogisticRegression": {
        "params": {
            "model__C": [0.01, 0.1, 1, 10, 100],
            "model__solver": ["liblinear", "saga"]
        }
    },
    "MultinomialNB": {
        "params": {
            "model__alpha": [0.01, 0.1],
            "model__fit_prior": [True, False]
        }
    },
    "RandomForest": {
        "params": {
            "model__n_estimators": [10 , 50 , 100],
        }
    },
    "GradientBoosting": {
        "params": {
            "model__n_estimators": [100, 150, 200],

        }
    },
    "XGBoost": {
        "params": {
            "model__n_estimators": [100, 150, 200],

        }
    }
}

mlflow.set_experiment("03 Hyperparameter Tuning")


with mlflow.start_run() as parent_run:
    for key, config in top_models_dict.items():
        model_name = config["model"]
        vec_name = config["vectorizer"]

        model = model_map[model_name]
        vectorizer = vectorizer_map[vec_name]
        param_grid = param_grids[model_name]["params"]

        try:
            with mlflow.start_run(run_name=f"Tuning_{vec_name}_{model_name}", nested=True):
                pipeline = Pipeline([
                    ("vectorizer", vectorizer),
                    ("model", model)
                ])

                grid = GridSearchCV(
                    pipeline,
                    param_grid=param_grid,
                    scoring="f1",
                    cv=3,
                    n_jobs=-1,
                    verbose=1
                )

                grid.fit(X_train_text, y_train)
                best_model = grid.best_estimator_
                y_pred = best_model.predict(X_test_text)

                # Log base info
                mlflow.log_param("model", model_name)
                mlflow.log_param("vectorizer", vec_name)

                # Log hyperparameters
                for param, value in grid.best_params_.items():
                    mlflow.log_param(param, value)

                # Log metrics
                mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
                mlflow.log_metric("precision", precision_score(y_test, y_pred))
                mlflow.log_metric("recall", recall_score(y_test, y_pred))
                mlflow.log_metric("f1_score", f1_score(y_test, y_pred))
                joblib.dump(grid.best_estimator_, "model.pkl")
                mlflow.log_artifact("model.pkl")

                # Log model

                print(f"✅ Tuning complete for {model_name} with {vec_name}")

        except Exception as e:
            print(f"❌ Failed tuning {model_name} with {vec_name} — {e}")
            mlflow.set_tag("error", str(e))



