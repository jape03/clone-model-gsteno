import base64
import io
import pickle
import os

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from PIL import Image
from tensorflow.keras.models import load_model, Model
import uvicorn

# ─── CONFIG ────────────────────────────────────────────────────────────────────
cnn_name             = "MobileNetV2"
h5_model_path        = f"{cnn_name}_steno_model.h5"
pkl_path             = f"{cnn_name}_embeddings_and_indices.pkl"
VALIDATION_THRESHOLD = 0.93  # 93%

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

def is_equivalent(expected: str, predicted: str) -> bool:
    return (
        predicted == expected
        or (expected in equivalents and predicted in equivalents[expected])
    )

# ─── IMAGE PREPROCESSING ────────────────────────────────────────────────────────
def composite_on_white(img: Image.Image) -> Image.Image:
    if img.mode == "RGB":
        return img
    img = img.convert("RGBA")
    bg  = Image.new("RGB", img.size, (255, 255, 255))
    bg.paste(img, mask=img.split()[3])
    return bg

def preprocess_image(img: Image.Image) -> np.ndarray:
    img = composite_on_white(img).convert("L")  # grayscale for cropping

    # Auto-crop to drawing content
    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)

    # Pad to square with white background
    w, h = img.size
    max_side = max(w, h)
    padded = Image.new("L", (max_side, max_side), color=255)
    padded.paste(img, ((max_side - w) // 2, (max_side - h) // 2))

    # Resize to 224×224 as required by the model
    final = padded.resize((224, 224), Image.LANCZOS)

    # Convert to RGB array and normalize
    arr = np.asarray(final, dtype="float32") / 255.0
    arr_rgb = np.stack([arr] * 3, axis=-1)
    return np.expand_dims(arr_rgb, 0)

def load_from_base64(b64: str) -> np.ndarray:
    raw = base64.b64decode(b64)
    img = Image.open(io.BytesIO(raw))
    return preprocess_image(img)

def l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 1e-10 else vec

# ─── FASTAPI SETUP ──────────────────────────────────────────────────────────────
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ─── GLOBAL OBJECTS ─────────────────────────────────────────────────────────────
embedding_model: Model = None
records: list = []

# ─── STARTUP EVENT ──────────────────────────────────────────────────────────────
@app.on_event("startup")
def load_model_and_records():
    global embedding_model, records
    print("[Startup] Loading model...")
    embedding_model_raw = load_model(h5_model_path, compile=False)
    embedding_model = Model(
        inputs=embedding_model_raw.inputs,
        outputs=embedding_model_raw.get_layer("embedding_layer").output
    )
    print("[Startup] Loading records...")
    with open(pkl_path, "rb") as f:
        records = pickle.load(f)["records"]
    print("[Startup] Finished loading.")

# ─── REQUEST & RESPONSE MODELS ─────────────────────────────────────────────────
class PredictionRequest(BaseModel):
    image: str
    expected_word: str

class PredictionResponse(BaseModel):
    correctness: str       # "Correct" or "Incorrect"
    expected_word: str
    detected_word: str
    accuracy: float
    reason: Optional[str] = None

# ─── PREDICT ENDPOINT ──────────────────────────────────────────────────────────
@app.post("/predict", response_model=PredictionResponse, response_model_exclude_none=True)
def predict(payload: PredictionRequest):
    try:
        x = load_from_base64(payload.image)
        emb_q = embedding_model.predict(x, verbose=0)[0]
        emb_q_n = l2_normalize(emb_q)

        exp_lower = payload.expected_word.lower()
        matched_records = [rec for rec in records if is_equivalent(exp_lower, rec["label"].lower())]

        if not matched_records:
            return {
                "correctness": "Incorrect",
                "expected_word": payload.expected_word,
                "detected_word": "",
                "accuracy": 0.0,
                "reason": f"No embeddings found for expected word '{payload.expected_word}'."
            }

        scores = [float(np.dot(emb_q_n, rec["emb"])) for rec in matched_records]
        best_score = max(scores)
        detected = matched_records[scores.index(best_score)]["label"]

        accuracy_pct = round(best_score * 100, 2)
        correct = best_score >= VALIDATION_THRESHOLD

        reason = None
        if not correct:
            reason = (
                f"Expected '{payload.expected_word}' scored "
                f"{accuracy_pct}%, below the {int(VALIDATION_THRESHOLD * 100)}% threshold"
            )

        return {
            "correctness": "Correct" if correct else "Incorrect",
            "expected_word": payload.expected_word,
            "detected_word": detected,
            "accuracy": round(best_score, 4),
            "reason": reason
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ─── HEALTH CHECK ──────────────────────────────────────────────────────────────
@app.get("/")
def health_check():
    return {"status": "alive"}

@app.head("/")
def health_check_head():
    return {"status": "alive"}

# ─── RUN ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "10000")))
