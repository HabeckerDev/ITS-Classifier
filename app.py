from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import json
import ast

app = FastAPI()
templates = Jinja2Templates(directory="templates")

with open("conversions/id2label.txt", "r") as data:
    id2label = ast.literal_eval(data.read())

with open("conversions/label2id.txt", "r") as data:
    label2id = ast.literal_eval(data.read())

deberta_v3_large = 'models/itsclassifier'
tokenizer = AutoTokenizer.from_pretrained(deberta_v3_large)
model = AutoModelForSequenceClassification.from_pretrained(deberta_v3_large, id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True)

def prediction_list(text):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_class_id = torch.topk(logits, 3)
    predictions = []
    for i in range(3):
        pred = predicted_class_id[1][0][i].item()
        predictions.append(model.config.id2label[str(pred)])
    return predictions

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, content: str = Form(...)):
    prediction = prediction_list(content)
    return templates.TemplateResponse("index.html", {"request": request, "prediction": prediction, "email": content})

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=5001, reload=True)
