# lab8app/redditApp.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import mlflow

app = FastAPI(
    title="Reddit Comment Classifier",
    description="Classify Reddit comments as removed (1) or not removed (0)",
    version="0.1"
)

class Comment(BaseModel):
    reddit_comment: str

@app.on_event("startup")
def load_model():
    global model_pipeline
    model_pipeline = mlflow.pyfunc.load_model("runs:/fe76d8a852844408baae6adc40f1d443/model")

@app.get("/")
def read_root():
    return {"message": "Reddit Comment Classifier is up and running"}

@app.post("/predict")
def predict(comment: Comment):
    text = comment.reddit_comment
    proba = model_pipeline.predict_proba([text])[0].tolist()
    return {"predictions": proba}
