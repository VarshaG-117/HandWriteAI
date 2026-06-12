from __future__ import annotations

import base64
import io
import os
from dataclasses import dataclass

import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image, ImageOps

try:
    from flask_cors import CORS
except ImportError:  # pragma: no cover - same-origin app works without it
    CORS = None

try:
    import fitz
except ImportError:  # pragma: no cover - depends on local installation
    fitz = None

try:
    from tensorflow import keras
except ImportError:  # pragma: no cover - depends on local installation
    keras = None


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.keras")
ALLOWED_IMAGES = {"jpg", "jpeg", "png"}
LETTERS = [chr(ord("A") + i) for i in range(26)]


app = Flask(__name__)
if CORS is not None:
    CORS(app)


@dataclass
class ModelState:
    model: object | None = None
    error: str | None = None


state = ModelState()


def load_model() -> None:
    if keras is None:
        state.error = "TensorFlow is not installed. Run pip install -r requirements.txt."
        return
    if not os.path.exists(MODEL_PATH):
        state.error = "model.keras was not found. Run python train_model.py first."
        return
    try:
        state.model = keras.models.load_model(MODEL_PATH)
        state.error = None
    except Exception as exc:  # pragma: no cover - defensive startup path
        state.error = f"Could not load model.keras: {exc}"


load_model()


def allowed_image(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_IMAGES


def image_from_data_url(data_url: str) -> Image.Image:
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    raw = base64.b64decode(data_url)
    return Image.open(io.BytesIO(raw))


def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("L")
    image = ImageOps.autocontrast(image)
    arr = np.array(image).astype("uint8")

    # Normalize foreground to bright ink on a dark background.
    if arr.mean() > 127:
        arr = 255 - arr

    mask = arr > 25
    if mask.any():
        ys, xs = np.where(mask)
        pad = 8
        y1, y2 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, arr.shape[0])
        x1, x2 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, arr.shape[1])
        arr = arr[y1:y2, x1:x2]

    h, w = arr.shape
    scale = 20.0 / max(h, w)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    glyph = Image.fromarray(arr).resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas = Image.new("L", (28, 28), 0)
    canvas.paste(glyph, ((28 - new_w) // 2, (28 - new_h) // 2))

    x = np.array(canvas).astype("float32") / 255.0
    return x.reshape(1, 28, 28, 1)


def predict_array(x: np.ndarray) -> dict:
    if state.model is None:
        return {
            "error": state.error or "Model is not ready.",
            "modelReady": False,
        }

    probs = state.model.predict(x, verbose=0)[0]
    order = np.argsort(probs)[::-1][:3]
    top = [
        {
            "character": LETTERS[int(i)],
            "confidence": round(float(probs[int(i)]) * 100, 2),
        }
        for i in order
    ]
    return {
        "modelReady": True,
        "prediction": top[0]["character"],
        "confidence": top[0]["confidence"],
        "top3": top,
    }


def extract_components(page_image: Image.Image) -> list[Image.Image]:
    gray = page_image.convert("L")
    gray = ImageOps.autocontrast(gray)
    arr = np.array(gray)
    if arr.mean() > 127:
        binary = arr < 190
    else:
        binary = arr > 65

    visited = np.zeros(binary.shape, dtype=bool)
    h, w = binary.shape
    boxes: list[tuple[int, int, int, int]] = []

    for y in range(h):
        for x in range(w):
            if not binary[y, x] or visited[y, x]:
                continue
            stack = [(y, x)]
            visited[y, x] = True
            ys, xs = [], []
            while stack:
                cy, cx = stack.pop()
                ys.append(cy)
                xs.append(cx)
                for ny in (cy - 1, cy, cy + 1):
                    for nx in (cx - 1, cx, cx + 1):
                        if (
                            0 <= ny < h
                            and 0 <= nx < w
                            and not visited[ny, nx]
                            and binary[ny, nx]
                        ):
                            visited[ny, nx] = True
                            stack.append((ny, nx))
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)
            area = (x2 - x1 + 1) * (y2 - y1 + 1)
            if area >= 80 and (x2 - x1) >= 5 and (y2 - y1) >= 8:
                boxes.append((x1, y1, x2 + 1, y2 + 1))

    boxes.sort(key=lambda b: (b[1] // 40, b[0]))
    return [page_image.crop(box) for box in boxes[:120]]


@app.get("/")
def index():
    return render_template("index.html", model_ready=state.model is not None)


@app.get("/health")
def health():
    return jsonify({"ok": True, "modelReady": state.model is not None, "error": state.error})


@app.post("/predict")
def predict_canvas():
    data = request.get_json(silent=True) or {}
    image_data = data.get("image")
    if not image_data:
        return jsonify({"error": "Missing canvas image data."}), 400
    image = image_from_data_url(image_data)
    return jsonify(predict_array(preprocess_image(image)))


@app.post("/upload-image")
def upload_image():
    file = request.files.get("image")
    if not file or not allowed_image(file.filename):
        return jsonify({"error": "Upload a jpg, jpeg, or png image."}), 400
    image = Image.open(file.stream)
    return jsonify(predict_array(preprocess_image(image)))


@app.post("/upload-pdf")
def upload_pdf():
    file = request.files.get("pdf")
    if not file or not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Upload a PDF file."}), 400
    if fitz is None:
        return jsonify({"error": "PyMuPDF is not installed. Run pip install -r requirements.txt."}), 503

    doc = fitz.open(stream=file.read(), filetype="pdf")
    predictions = []
    for page_index, page in enumerate(doc):
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        page_image = Image.open(io.BytesIO(pix.tobytes("png")))
        components = extract_components(page_image)
        for item_index, component in enumerate(components):
            result = predict_array(preprocess_image(component))
            if "prediction" in result:
                result["page"] = page_index + 1
                result["item"] = item_index + 1
                predictions.append(result)
            elif "error" in result:
                return jsonify(result), 503

    text = "".join(item["prediction"] for item in predictions)
    return jsonify({"modelReady": True, "characters": predictions, "text": text})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
