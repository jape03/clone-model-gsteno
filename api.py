import base64
import io
import os
import pickle
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model, Model
from tqdm import tqdm

app = FastAPI()

# ─── Enable CORS ──────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Config ───────────────────────────────────────────────────────────────────
cnn_name = "MobileNetV2"
h5_model_path = f"{cnn_name}_steno_model.h5"
pkl_path = f"{cnn_name}_embeddings_and_indices.pkl"
MIN_KNN_SIM = 0.93
TOP_K = 5

# ─── Lazy-loaded globals ──────────────────────────────────────────────────────
model = None
emb_model = None
records = []
class_indices = {}
all_labels = []

# ─── Lazy loader ──────────────────────────────────────────────────────────────
def load_assets():
    global model, emb_model, records, class_indices, all_labels

    if model is not None and emb_model is not None:
        return  # Already loaded

    print("🔍 Loading model and embeddings...")
    if not os.path.exists(h5_model_path):
        raise FileNotFoundError(f"Missing model file: {h5_model_path}")
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Missing embeddings file: {pkl_path}")

    model = load_model(h5_model_path, compile=False)
    emb_model = Model(inputs=model.inputs, outputs=model.get_layer("embedding_layer").output)

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    class_indices = data["class_indices"]
    records = data["records"]
    all_labels.extend(class_indices.keys())
    print("✅ Model and embeddings loaded")

# ─── Helper Functions ─────────────────────────────────────────────────────────
equivalents = {
    "a": ["a", "an"], "an": ["a", "an"],
    "are": ["are", "our", "hour"], "our": ["are", "our", "hour"], "hour": ["are", "our", "hour"],
    "at": ["at", "it"], "it": ["at", "it"],
    "be": ["be", "by"], "by": ["be", "by"],
    "correspond": ["correspond", "correspondence"], "correspondence": ["correspond", "correspondence"],
    "ever": ["ever", "every"], "every": ["ever", "every"],
    "important": ["important", "importance"], "importance": ["important", "importance"],
    "in": ["in", "not"], "not": ["in", "not"],
    "is": ["is", "his"], "his": ["is", "his"],
    "publish": ["publish", "publication"], "publication": ["publish", "publication"],
    "satisfy": ["satisfy", "satisfactory"], "satisfactory": ["satisfy", "satisfactory"],
    "their": ["their", "there"], "there": ["their", "there"],
    "thing": ["thing", "think"], "think": ["thing", "think"],
    "well": ["well", "will"], "will": ["well", "will"],
    "won": ["won", "one"], "one": ["won", "one"],
    "you": ["you", "your"], "your": ["you", "your"],
}

def is_equivalent(expected, predicted):
    if expected is None:
        return False
    return expected == predicted or predicted in equivalents.get(expected, [])

class PredictionRequest(BaseModel):
    image: str
    expected_word: str

def preprocess_base64(base64_str):
    image_data = base64.b64decode(base64_str)
    image = Image.open(io.BytesIO(image_data)).convert("L")
    image = ImageOps.invert(image)
    image = ImageOps.autocontrast(image)
    image = ImageOps.crop(image)
    image = ImageOps.pad(image, (224, 224), method=Image.LANCZOS, color=0)
    image = image.convert("RGB")
    arr = np.array(image).astype("float32") / 255.0
    return np.expand_dims(arr, 0)

def l2_normalize(vec):
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 1e-10 else vec

# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/")
def health_check():
    return {
        "status": "alive",
        "files": os.listdir(),
        "model_present": os.path.exists(h5_model_path),
        "pkl_present": os.path.exists(pkl_path),
    }

@app.post("/predict")
def predict(payload: PredictionRequest):
    try:
        load_assets()
        x = preprocess_base64(payload.image)
        emb_q = emb_model.predict(x, verbose=0)[0]
        emb_q_n = l2_normalize(emb_q)

        sims = [(float(np.dot(emb_q_n, rec["emb"])), rec["label"]) for rec in records]
        sims.sort(key=lambda t: t[0], reverse=True)

        top_matches = [{"word": label, "score": round(score, 4)} for score, label in sims[:5]]
        best_score, best_label = sims[0]

        return {
            "expected_word": payload.expected_word,
            "predicted_word": best_label,
            "similarity_score": round(best_score, 4),
            "top_5": top_matches
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
