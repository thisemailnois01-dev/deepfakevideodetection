from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
import os

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model("model.h5")

IMG_SIZE = 224

def preprocess_frame(frame):
    frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    frame = frame / 255.0
    return frame

@app.route("/")
def home():
    return "Deepfake API is running"

@app.route("/predict", methods=["POST"])
def predict():
    if 'video' not in request.files:
        return jsonify({"error": "No video uploaded"})

    file = request.files["video"]
    path = "temp.mp4"
    file.save(path)

    cap = cv2.VideoCapture(path)
    frames = []

    count = 0
    while count < 10:
        ret, frame = cap.read()
        if not ret:
            break

        frame = preprocess_frame(frame)
        frames.append(frame)
        count += 1

    cap.release()
    os.remove(path)

    if len(frames) == 0:
        return jsonify({"error": "No frames extracted"})

    frames = np.array(frames)

    # prediction
    preds = model.predict(frames)
    score = float(np.mean(preds))

    result = "FAKE" if score > 0.5 else "REAL"

    return jsonify({
        "prediction": result,
        "confidence": score
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)