import os
import json
import datetime
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from PIL import Image
from gtts import gTTS

app = Flask(__name__)

# --- Configuration ---
UPLOAD_FOLDER = "static/uploads"
HISTORY_FILE = "history.json"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- 1. Load AI Model & Metadata ---
# Ensure these files are in your main project folder
MODEL_PATH = "plant_disease_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)

with open("class_names.json", "r") as f:
    class_names = json.load(f)

with open("disease_info.json", "r", encoding="utf-8") as f:
    disease_info = json.load(f)

# --- 2. Helper Functions ---
def save_to_history(data):
    """Saves scan metadata to a local JSON file for the History Log."""
    history = []
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                history = json.load(f)
        except:
            history = []
    
    # Add new scan to the beginning of the list
    history.insert(0, data)
    
    # Keep only the last 20 scans to save space
    with open(HISTORY_FILE, "w") as f:
        json.dump(history[:20], f)

def predict_image(image_path):
    """Processes image and returns predicted class and confidence."""
    # Resize to 128x128 to match your model training
    img = Image.open(image_path).convert('RGB').resize((128, 128))
    img_array = np.array(img).astype('float32')
    img_array = np.expand_dims(img_array, axis=0)
    
    predictions = model.predict(img_array)
    confidence = float(np.max(predictions) * 100)
    index = int(np.argmax(predictions))
    
    return class_names[index], confidence

# --- 3. Main Routes ---

@app.route("/", methods=["GET", "POST"])
def index():
    # Initialize variables with default safe strings
    result = None
    prevention = "Maintain balanced N-P-K levels and monitor plant health."
    description = "Potential nutritional deficiency detected."
    confidence = 0
    audio_file = None
    uploaded_image_name = None
    selected_language = "en"
    error_message = None

    if request.method == "POST":
        file = request.files.get("file")
        selected_language = request.form.get("language", "en")

        if file and file.filename != '':
            # Create unique filename
            ts = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
            filename = secure_filename(f"{ts}_{file.filename}")
            path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(path)
            uploaded_image_name = filename

            # AI Inference
            res_class, conf_score = predict_image(path)
            confidence = round(conf_score, 2)

            # 🚨 Threshold Filter (85%)
            if conf_score < 85:
                error_message = "⚠️ Low Confidence: This doesn't look like a clear leaf scan. Please use a plain background and better lighting."
                result = "Unknown Analysis"
            else:
                result = res_class
                # Fetch data with .get() to prevent UndefinedError/NoneType crashes
                disease_data = disease_info.get(res_class, {})
                lang_data = disease_data.get(selected_language, {})
                
                description = lang_data.get("description", "Information on this specific species is currently being updated.")
                prevention = lang_data.get("prevention", "Consult a local agricultural expert for specialized treatment.")

                # Save valid scan to history
                save_to_history({
                    "date": datetime.datetime.now().strftime("%d %b %Y, %H:%M"),
                    "result": res_class.replace('_', ' '),
                    "confidence": confidence,
                    "image": filename # This key matches your history.html
                })

                # Audio Generation
                audio_filename = f"voice_{ts}.mp3"
                tts_text = f"The result is {res_class.replace('_', ' ')}. Advice: {prevention}"
                try:
                    tts = gTTS(text=tts_text, lang=selected_language)
                    tts.save(os.path.join(app.config["UPLOAD_FOLDER"], audio_filename))
                    audio_file = audio_filename
                except:
                    audio_file = None

    return render_template("index.html", 
                           result=result, 
                           prevention=prevention, 
                           description=description, 
                           confidence=confidence,
                           audio_file=audio_file, 
                           uploaded_image_name=uploaded_image_name,
                           selected_language=selected_language,
                           error_message=error_message)

@app.route("/history")
def history():
    """Displays the history of scans."""
    scans = []
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                scans = json.load(f)
        except:
            scans = []
    return render_template("history.html", scans=scans)

@app.route("/reset")
def reset():
    """Clears analysis and returns to home."""
    return redirect(url_for('index'))

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)