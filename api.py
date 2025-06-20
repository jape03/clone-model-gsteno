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
import uvicorn
from tqdm import tqdm

# Suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Load Model and Prototypes ────────────────────────────────────────────────
cnn_name = "MobileNetV2"
h5_model_path = f"{cnn_name}_steno_model.h5"
pkl_path = f"{cnn_name}_embeddings_and_indices.pkl"
MIN_KNN_SIM = 0.93
TOP_K = 5

model = load_model(h5_model_path, compile=False)
emb_model = Model(inputs=model.inputs, outputs=model.get_layer("embedding_layer").output)

with open(pkl_path, "rb") as f:
    data = pickle.load(f)

class_indices = data["class_indices"]
records = data["records"]
all_labels = list(class_indices.keys())

# ─── Equivalence Handling ──────────────────────────────────────────────────────
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

# ─── Request Model ─────────────────────────────────────────────────────────────
class PredictionRequest(BaseModel):
    image: str
    expected_word: str

# ─── Utility Functions ─────────────────────────────────────────────────────────
def preprocess_base64(base64_str):
    image_data = base64.b64decode(base64_str)
    image = Image.open(io.BytesIO(image_data)).convert("L")
    image = ImageOps.invert(image)
    image = ImageOps.autocontrast(image)
    image = ImageOps.crop(image)
    image = ImageOps.pad(image, (224, 224), method=Image.LANCZOS, color=0)
    image = image.convert("RGB")
    image.save("last_processed.png")
    arr = np.array(image).astype("float32") / 255.0
    return np.expand_dims(arr, 0)

def preprocess_path(path):
    image = Image.open(path).convert("RGB")
    image = image.resize((224, 224), resample=Image.LANCZOS)
    arr = np.array(image).astype("float32") / 255.0
    return np.expand_dims(arr, 0)

def l2_normalize(vec):
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 1e-10 else vec

# ─── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/predict")
def predict(payload: PredictionRequest):
    try:
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

@app.get("/batch_predict")
def batch_predict(
    images_dir: str = Query(..., description="Directory containing stroke images"),
    report_name: Optional[str] = Query(None, description="Name for the output report file")
):
    if not os.path.exists(images_dir):
        raise HTTPException(status_code=404, detail=f"Folder '{images_dir}' not found.")

    total = correct1 = correct_k = 0
    report_lines = []

    for fname in tqdm(sorted(os.listdir(images_dir)), desc="Batch Testing"):
        if not fname.lower().endswith(".png"):
            continue

        total += 1
        path = os.path.join(images_dir, fname)
        base = os.path.splitext(fname)[0].lower()
        expected = next((lbl for lbl in all_labels if lbl.lower() == base), None)

        x = preprocess_path(path)
        emb_q = emb_model.predict(x, verbose=0)[0]
        emb_q_n = l2_normalize(emb_q)

        raw = [(float(np.dot(emb_q_n, rec["emb"])), rec["label"]) for rec in records]
        best_map = {}
        for sim, lbl in raw:
            if lbl not in best_map or sim > best_map[lbl]:
                best_map[lbl] = sim
        sims = sorted([(sim, lbl) for lbl, sim in best_map.items()], key=lambda x: x[0], reverse=True)

        best_sim, best_lbl = sims[0]
        top_labels = [lbl for sim, lbl in sims[:TOP_K]]
        ok1 = (best_sim >= MIN_KNN_SIM) and is_equivalent(expected, best_lbl)
        okK = any(is_equivalent(expected, lbl) for lbl in top_labels)

        correct1 += int(ok1)
        correct_k += int(okK)

        rpt = [
            f"Stroke Assessment for '{fname}'",
            f"  Expected Word:     {expected or '❌ Not in classes'}",
            f"  Predicted (Top-1): {best_lbl}",
            f"  Cosine Sim:        {best_sim*100:.2f}%",
            f"  Correct (Top-1):   {'✅' if ok1 else '❌'}",
            f"  Correct (Top-{TOP_K}): {'✅' if okK else '❌'}",
            "  Nearest Neighbors:"
        ] + [f"    {lbl:<15} {sim*100:.2f}%" for sim, lbl in sims[:TOP_K]] + [""]

        report_lines.extend(rpt)

    summary = [
        f"Batch Test Summary: {cnn_name}",
        f"  Total images:        {total}",
        f"  Top-1 correct (≥{int(MIN_KNN_SIM*100)}%): {correct1} ({correct1/total*100:.2f}%)",
        f"  Top-{TOP_K} correct:  {correct_k} ({correct_k/total*100:.2f}%)",
        f"  Top-1 misses:        {total - correct1} ({(total - correct1)/total*100:.2f}%)",
        f"  Top-{TOP_K} misses:   {total - correct_k} ({(total - correct_k)/total*100:.2f}%)"
    ]
    report_lines.append("\n" + "\n".join(summary))

    report_path = report_name or f"{cnn_name}_batch_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    return {
        "status": "completed",
        "total_images": total,
        "top1_correct": correct1,
        "topK_correct": correct_k,
        "report_path": os.path.abspath(report_path)
    }

@app.api_route("/", methods=["GET", "HEAD"])
def health_check():
    return {"status": "alive"}

# ─── Local Run ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)