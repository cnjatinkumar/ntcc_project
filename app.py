from flask import Flask, request, jsonify
import os
import numpy as np
import librosa
import tensorflow as tf
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg'}
MAX_FILE_SIZE_MB = 50

# Set absolute path for the model
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'emotion_detection_model.h5')

app.config.from_mapping(
    UPLOAD_FOLDER=UPLOAD_FOLDER,
    MAX_CONTENT_LENGTH=MAX_FILE_SIZE_MB * 1024 * 1024,
    MODEL_PATH=MODEL_PATH
)

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load emotion detection model
try:
    print(f"Loading model from: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load model: {str(e)}") from e

# Supported emotions (aligned with model outputs)
EMOTION_LABELS = ['neutral', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise']

def allowed_file(filename):
    """Check if file has valid extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_features(audio_path, segment_duration=2.0, step=1.0):
    """Extract MFCC features from audio segments"""
    try:
        y, sr = librosa.load(audio_path, sr=22050)
    except Exception as e:
        return None, f"Error loading audio: {str(e)}"
    segment_length = int(segment_duration * sr)
    step_length = int(step * sr)
    if len(y) < segment_length:
        return None, "Audio is too short (minimum 2 seconds required)"
    features = []
    timestamps = []
    for i in range(0, len(y) - segment_length, step_length):
        segment = y[i:i + segment_length]
        mfccs = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=40)
        features.append(np.mean(mfccs, axis=1))
        timestamps.append(round(i / sr, 2))
    return np.array(features), timestamps, None

def detect_emotion(audio_path):
    """Analyze audio and detect emotions over time"""
    features, timestamps, error = extract_features(audio_path)
    if error:
        return {'error': error}, None
    if features.size == 0:
        return {'error': "No features could be extracted"}, None
    try:
        predictions = model.predict(features)
    except Exception as e:
        return {'error': f"Model prediction failed: {str(e)}"}, None
    results = []
    for i, pred in enumerate(predictions):
        emotion_idx = np.argmax(pred)
        results.append({
            'timestamp': timestamps[i],
            'emotion': EMOTION_LABELS[emotion_idx],
            'confidence': float(np.max(pred)),
            'emotion_breakdown': {label: float(pred[i]) for i, label in enumerate(EMOTION_LABELS)}
        })
    return None, results

# A simple GET endpoint to confirm the server is running
@app.route('/', methods=['GET'])
def home():
    return "Server is running!", 200

@app.route('/analyze', methods=['POST'])
def analyze_audio():
    """Endpoint for audio analysis"""
    if 'audio' not in request.files:
        return jsonify({'status': 'error', 'message': 'Please upload an audio file'}), 400
    file = request.files['audio']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No file selected'}), 400
    if not allowed_file(file.filename):
        return jsonify({'status': 'error', 'message': 'Invalid file type'}), 400
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    try:
        error, results = detect_emotion(file_path)
        if error:
            return jsonify({'status': 'error', 'message': error.get('error', 'Analysis failed')}), 400
        return jsonify({'status': 'success', 'analysis': results})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
