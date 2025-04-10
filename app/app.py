from fastapi import FastAPI
from pydantic import BaseModel
import joblib

#load model
model = joblib.load("model/sentiment_model.pkl")

#create fast api model
app = FastAPI()

#define input structure
class TextInput(BaseModel):
    text : str

#define endpoint
@app.post("/predict")
def predict_sentiment(txt : TextInput):
    pred = model.predict([txt.text])[0]
    return {"sentiment" : pred}
