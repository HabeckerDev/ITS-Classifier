from flask import Flask, render_template, request
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import json
import ast

app = Flask(__name__)

with open("conversions/id2label.txt", "r") as data:
    id2label = ast.literal_eval(data.read())

with open("conversions/label2id.txt", "r") as data:
    label2id = ast.literal_eval(data.read())

deberta_v3_large = 'models/itsclassifier'
tokenizer = AutoTokenizer.from_pretrained(deberta_v3_large)
model = AutoModelForSequenceClassification.from_pretrained(deberta_v3_large,id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    def prediction_list(text):
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits
        predicted_class_id = logits
        predicted_class_id = torch.topk(logits, 3)
        predictions = []
        for i in range(3):
            pred = predicted_class_id[1][0][i].item()
            predictions.append(model.config.id2label[str(pred)])
        return predictions
    
    email = request.form.get('content')
    prediction = prediction_list(email)
    return render_template("index.html", prediction=prediction, email=email)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)