
---

# ITS-Classifier API Documentation

## Overview

The ITS-Classifier API is a web application built with FastAPI that uses a pre-trained DeBERTa model for sequence classification. The API provides an interface for users to input text and receive predicted classifications.

## Table of Contents

1. [Installation](#installation)
2. [Running the Server](#running-the-server)
3. [API Endpoints](#api-endpoints)
    - [Home Endpoint](#home-endpoint)
    - [Predict Endpoint](#predict-endpoint)
4. [Project Structure](#project-structure)
5. [Configuration](#configuration)
6. [Model Details](#model-details)

## Installation

1. **Clone the Repository:**

    ```bash
    git clone <repository-url>
    cd itsclassifier-main
    ```

2. **Create a Virtual Environment:**

    ```bash
    python -m venv env
    source env/bin/activate  # On macOS/Linux
    .\env\Scripts\activate   # On Windows
    ```

3. **Install Dependencies:**

    ```bash
    pip install fastapi uvicorn transformers torch
    ```

## Running the Server

To start the FastAPI server, run the following command:

```bash
uvicorn app:app --host 0.0.0.0 --port 5001 --reload
```

- `app:app` specifies the module and application instance.
- `--host 0.0.0.0` allows the server to be accessible from any network interface.
- `--port 5001` sets the port for the server.
- `--reload` enables automatic reloading of the server when code changes.

## API Endpoints

### Home Endpoint

- **URL:** `/`
- **Method:** `GET`
- **Description:** Renders the home page with a form for text input.
- **Response:** HTML page (`index.html`)

#### Example

Request:
```http
GET /
```

Response:
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ITS-Classifier</title>
</head>
<body>
    <form action="/predict" method="post">
        <textarea name="content"></textarea>
        <button type="submit">Predict</button>
    </form>
</body>
</html>
```

### Predict Endpoint

- **URL:** `/predict`
- **Method:** `POST`
- **Description:** Accepts text input and returns the top 3 predicted classifications.
- **Request Parameters:**
  - `content` (string): The text to be classified.
- **Response:** HTML page with predictions and the original input text.

#### Example

Request:
```http
POST /predict
Content-Type: application/x-www-form-urlencoded

content=Your+input+text+here
```

Response:
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ITS-Classifier</title>
</head>
<body>
    <p>Email: Your input text here</p>
    <p>Predictions:</p>
    <ul>
        <li>Prediction 1</li>
        <li>Prediction 2</li>
        <li>Prediction 3</li>
    </ul>
    <form action="/predict" method="post">
        <textarea name="content"></textarea>
        <button type="submit">Predict</button>
    </form>
</body>
</html>
```

## Project Structure

```
itsclassifier-main/
├── conversions/
│   ├── id2label.txt
│   └── label2id.txt
├── templates/
│   └── index.html
├── app.py
└── ...
```

- `conversions/`: Contains label mapping files.
- `templates/`: Contains HTML templates.
- `app.py`: The main FastAPI application file.

## Configuration

Ensure the following files are correctly set up:

1. **Label Mapping Files:**
   - `conversions/id2label.txt`: Contains a dictionary mapping IDs to labels.
   - `conversions/label2id.txt`: Contains a dictionary mapping labels to IDs.

2. **HTML Template:**
   - `templates/index.html`: The template for rendering the home and prediction pages.

## Model Details

- **Model:** DeBERTa V3 Large
- **Tokenizer and Model Loading:**
  ```python
  deberta_v3_large = 'models/itsclassifier'
  tokenizer = AutoTokenizer.from_pretrained(deberta_v3_large)
  model = AutoModelForSequenceClassification.from_pretrained(deberta_v3_large, id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True)
  ```
- **Prediction Logic:**
  ```python
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
  ```

## Conclusion

This documentation provides a comprehensive overview of the ITS-Classifier API, including installation instructions, how to run the server, API endpoints, project structure, configuration, and model details. For any further questions or issues, please refer to the FastAPI and Transformers documentation.

---
