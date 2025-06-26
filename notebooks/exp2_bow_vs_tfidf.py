import mlflow.sklearn
import pandas as pd
import numpy as np
import re
import string
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

# ===== MLflow EXPERIMENT CONFIG =====
mlflow.set_experiment("02 Model Comparisons")



vectorizers = {
    'BOW' : CountVectorizer(),
    'TF-IDF': TfidfVectorizer()
}

models = {
    'LogisticRegression':LogisticRegression(),
    'MultinomialNB':MultinomialNB(),

}

with mlflow.start_run() as parent_run:
    for model_name , model in models.items():
        for vec_name,vectorizer in vectorizers.items():
            try:
                with mlflow.start_run(run_name = f"{vec_name}_{model_name}", nested=True) as child_run: 
                    X = vectorizer.fit_transform(df['content'])
                    y = df['sentiment']
                    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2 , stratify=y, random_state=42)

                    mlflow.log_param("vectorizer", vec_name)
                    mlflow.log_param("Algorithm" , model_name)
                    mlflow.log_param('test_size', 0.2)

                    ml = model
                    ml.fit(X_train,y_train)
                    y_pred = model.predict(X_test)

                    if model_name== "LogisticRegression":
                        mlflow.log_param("C", str(round(ml.C , 2)))

                    elif model_name =="MultinomialNB":
                        mlflow.log_param("alpha" , str(ml.alpha))
    
                    elif model_name == 'XGBoost':
                        mlflow.log_param("n_estimators", ml.n_estimators)
                        mlflow.log_param("learning_rate", ml.learning_rate)


                    elif model_name == 'GradientBoosting':
                        mlflow.log_param("n_estimators", ml.n_estimators)
                        mlflow.log_param("learning_rate", ml.learning_rate)
                        mlflow.log_param("max_depth", ml.max_depth)

                    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
                    mlflow.log_metric("precision", precision_score(y_test, y_pred))
                    mlflow.log_metric("recall", recall_score(y_test, y_pred))
                    mlflow.log_metric("f1_score", f1_score(y_test, y_pred))   
            except Exception as e:
                print(f"⚠️ Error with {model_name} + {vec_name}: {e}")
                mlflow.log_param("error", str(e))
