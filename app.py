import os
from flask import Flask, request, render_template
from PIL import Image
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import csv

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Carica il modello da TensorFlow Hub
model = tf.keras.Sequential([
    hub.KerasLayer("https://tfhub.dev/google/aiy/vision/classifier/plants_V1/1")
])

# Scarica le etichette del modello
labels_path = tf.keras.utils.get_file(
    'labels.csv',
    'https://storage.googleapis.com/download.tensorflow.org/data/aiy_plants_V1_labelmap.csv'
)

with open(labels_path, newline='') as f:
    reader = csv.reader(f)
    next(reader)  # Salta intestazione CSV
    labels = [row[1] for row in reader]

def predict_image(image_path):
    image = Image.open(image_path).resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0).astype(np.float32)
    prediction = model.predict(image)
    predicted_index = np.argmax(prediction)
    predicted_label = labels[predicted_index]
    confidence = float(np.max(prediction))
    return predicted_label, confidence

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    image_url = None

    if request.method == "POST":
        file = request.files["image"]
        if file:
            os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(file_path)
            prediction, confidence = predict_image(file_path)
            image_url = file_path

    return render_template("index.html", prediction=prediction, confidence=confidence, image_url=image_url)

if __name__ == "__main__":
    app.run(debug=True)
