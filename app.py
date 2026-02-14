from flask import Flask, render_template, request
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import json
import os

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load model
model = load_model("tyre_quality_model.keras")

# Load labels
with open("labels.json") as f:
    labels = json.load(f)

IMG_SIZE = (224, 224)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction_text = None
    confidence = None
    image_path = None

    if request.method == "POST":
        file = request.files["image"]

        if file:
            image_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(image_path)

            # Read & preprocess image
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, IMG_SIZE)
            img = img / 255.0
            img = np.expand_dims(img, axis=0)

            # Predict
            pred = model.predict(img)[0][0]

            if pred > 0.5:
                prediction_text = "GOOD TYRE"
                confidence = pred
            else:
                prediction_text = "DEFECTIVE TYRE"
                confidence = 1 - pred

            # Plot confidence graph
            labels_graph = ["Defective", "Good"]
            values = [1-pred, pred]

            plt.figure()
            plt.bar(labels_graph, values)
            plt.ylim(0, 1)
            plt.ylabel("Confidence")
            plt.title("Tyre Quality Prediction")
            plt.savefig("static/graph.png")
            plt.close()

    return render_template(
        "index.html",
        prediction=prediction_text,
        confidence=confidence,
        image=image_path
    )

if __name__ == "__main__":
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)
