FROM python:3.10

WORKDIR /app

COPY flask_app/ /app/

COPY model/vectorizer.pkl /app/model/vectorizer.pkl 

RUN pip install -r requirements.txt 

RUN python -m nltk.downloader stopwords wordnet

EXPOSE 5000

CMD [ "gunicorn", "-b", "0.0.0.0:5000", "app:app" ]
