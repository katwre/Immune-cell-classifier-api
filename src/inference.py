#!/usr/bin/env python
# coding: utf-8
import numpy as np
from PIL import Image
from tflite_runtime.interpreter import Interpreter
from io import BytesIO
import requests
import json

classes = ['basophil',
'eosinophil',
'erythroblast',
'ig',
'lymphocyte',
'monocyte',
'neutrophil',
'platelet']


interpreter = Interpreter(model_path="models/xception_v4_lr0.0001_drop0.5_27_0.911.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_index = input_details[0]["index"]
output_index = output_details[0]["index"]

def preprocess_input(x):
    x /= 127.5
    x -= 1.
    return x

def run_inference(url: str):

    # Download image
    resp = requests.get(url)
    resp.raise_for_status()

    # Open from bytes
    with Image.open(BytesIO(resp.content)) as img:
        img = img.convert("RGB")
        img = img.resize((150, 150), Image.NEAREST)
        x = np.array(img, dtype="float32")
        X = np.array([x])
        X = preprocess_input(X)

    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)[0]

    # Convert logits to probs
    exp = np.exp(preds - np.max(preds))
    probs = exp / exp.sum()
    rounded = {cls: round(float(p), 2) for cls, p in zip(classes, probs)}
    return rounded

def lambda_handler(event, context):
    
    # If called from the console or RIE, event may already be the JSON
    if "body" in event:
        payload = json.loads(event["body"])
    else:
        payload = event

    image_url = payload["url"]
    predictions = run_inference(image_url)

    return json.dumps({"predictions": predictions})

