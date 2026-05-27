from __future__ import annotations

import base64
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request, send_file


APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR.parent / "data"
IMAGE_PATH = DATA_DIR / "the_ambassadors.jpg"

# Точки, как в решении в ноутбуке Seminar_4.
DEFAULT_POINTS = np.array(
    [
        [250.0, 900.0],
        [650.0, 850.0],
        [820.0, 980.0],
        [380.0, 1080.0],
    ],
    dtype=np.float32,
)

RESULT_SIZE = (300, 400)  # width, height
PADDING_RATIO = 0.2

app = Flask(
    __name__,
    template_folder=str(APP_DIR / "templates"),
    static_folder=str(APP_DIR / "static"),
)


def load_image() -> np.ndarray:
    image = cv2.imread(str(IMAGE_PATH))
    if image is None:
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")
    return image


def warp_skull(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    out_w, out_h = RESULT_SIZE
    dst = np.array(
        [
            [0, 0],
            [out_w - 1, 0],
            [out_w - 1, out_h - 1],
            [0, out_h - 1],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(points.astype(np.float32), dst)
    return cv2.warpPerspective(image, matrix, (out_w, out_h))


@app.get("/")
def index():
    image = load_image()
    h, w = image.shape[:2]
    points = DEFAULT_POINTS.astype(np.float32).tolist()
    return render_template(
        "index.html",
        image_width=w,
        image_height=h,
        points=points,
        padding_ratio=PADDING_RATIO,
    )


@app.get("/image")
def image():
    return send_file(str(IMAGE_PATH))


@app.post("/warp")
def warp():
    payload = request.get_json(silent=True) or {}
    points_raw = payload.get("points")

    if not isinstance(points_raw, list) or len(points_raw) != 4:
        return jsonify({"error": "Expected 4 corner points"}), 400

    try:
        points = np.array(points_raw, dtype=np.float32)
        if points.shape != (4, 2):
            raise ValueError("Invalid shape")
    except Exception:
        return jsonify({"error": "Points must be [[x, y], ...]"}), 400

    image = load_image()
    skull = warp_skull(image, points)
    ok, encoded = cv2.imencode(".jpg", skull, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        return jsonify({"error": "Failed to encode image"}), 500

    result_b64 = base64.b64encode(encoded).decode("ascii")
    return jsonify({"image": f"data:image/jpeg;base64,{result_b64}"})


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5050)
