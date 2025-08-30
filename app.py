from flask import Flask, render_template, request
import requests
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, decode_predictions, preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
from io import BytesIO
from PIL import Image
import random
import os

app = Flask(__name__)

# Load MobileNetV2 model with imagenet weights
model = MobileNetV2(weights="imagenet")

# Use NASA API key from environment (set in terminal or .env file)
NASA_API_KEY = os.environ.get("NASA_API_KEY")
NASA_URL = f"https://api.nasa.gov/planetary/apod?api_key={NASA_API_KEY}"

def get_random_nasa_image():
    """Fetch one random NASA APOD image (repeats until finds image media)."""
    while True:
        year = random.randint(2000, 2025)
        month = random.randint(1, 12)
        day = random.randint(1, 28)
        date = f"{year}-{month:02d}-{day:02d}"
        response = requests.get(f"{NASA_URL}&date={date}")
        data = response.json()
        if data.get("media_type") == "image":
            return data.get("url"), data.get("title", "NASA Image")

def classify_image(img):
    """Classify PIL image with MobileNetV2."""
    img = img.resize((224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    decoded = decode_predictions(preds, top=3)[0]
    return [(label, round(float(prob) * 100, 2)) for (_, label, prob) in decoded]

@app.route("/", methods=["GET", "POST"])
def index():
    uploaded_img_url = None
    uploaded_predictions = []
    nasa_images = []

    if request.method == "POST" and "upload" in request.files:
        # User uploaded an image
        file = request.files["upload"]
        img = Image.open(file.stream).convert("RGB")
        uploaded_predictions = classify_image(img)
        uploaded_img_url = None  # Can extend to serve uploaded image if needed

    # Always display multiple random NASA images
    for _ in range(3):
        url, title = get_random_nasa_image()
        response = requests.get(url)
        img = Image.open(BytesIO(response.content)).convert("RGB")
        preds = classify_image(img)
        nasa_images.append({"url": url, "title": title, "predictions": preds})

    return render_template(
        "index.html",
        nasa_images=nasa_images,
        uploaded_predictions=uploaded_predictions,
        uploaded_img_url=uploaded_img_url
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
