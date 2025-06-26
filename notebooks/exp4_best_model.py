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
with open("best_model.json", "r") as f:
    top_models_dict = json.load(f)  # No extra key access



mlflow.set_experiment("03 Best Model")

with mlflow.start_run() as parent_run:
    # Since there's only one best model, we fetch it directly
    config = top_models_dict["top_1"]  # Assuming the JSON uses "top_1" as key
    model_name = config["model"]
    vec_name = config["vectorizer"]


    if model_name == "LogisticRegression":
        model = LogisticRegression(max_iter=1000)
    elif model_name == "MultinomialNB":
        model = MultinomialNB()
    elif model_name == "RandomForest":
        model = RandomForestClassifier()
    elif model_name == "GradientBoosting":
        model = GradientBoostingClassifier()
    elif model_name == "XGBoost":
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    params = model.get_params()
    for param_name, param_value in params.items():
        try:
            mlflow.log_param(param_name, param_value)
        except:
            pass
    if vec_name == "BOW":
        vectorizer = CountVectorizer()
    elif vec_name == "TF-IDF":
        vectorizer = TfidfVectorizer()
    else:
        raise ValueError(f"Unsupported vectorizer: {vec_name}")

    # Create pipeline
    pipe = Pipeline([
        ("vectorizer", vectorizer),
        ("model", model)
    ])

    # Train
    pipe.fit(X_train_text, y_train)
    y_pred = pipe.predict(X_test_text)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Log everything
    mlflow.log_param("model", model_name)
    mlflow.log_param("vectorizer", vec_name)
    mlflow.log_param("source_run_id", config["run_id"])
    

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)

    # Save and log model
    joblib.dump(pipe, "best_model.pkl")
    mlflow.log_artifact("best_model.pkl")

    print(f"✅ Best model logged: {model_name} with {vec_name} — f1_score: {f1:.4f}")

