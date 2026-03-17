import json
from pathlib import Path

import numpy as np

from model import CNN, process_window


class GestureInferenceEngine:
    def __init__(self, bitstream_path: str = "cnn.bit", gesture_map_path: str = "gesture_map.json"):
        self.base_dir = Path(__file__).resolve().parent

        with open(self.base_dir / gesture_map_path, "r", encoding="utf-8") as f:
            raw_map = json.load(f)

        # gesture_map.json is gesture_name -> class_id
        # invert to class_id -> gesture_name
        self.id_to_gesture = {int(v): str(k) for k, v in raw_map.items()}

        self.cnn = CNN(str(self.base_dir / bitstream_path))
        self.window = np.zeros((CNN.IN_LEN, CNN.RAW_CH), dtype=np.float32)
        self.samples_seen = 0

    def reset(self) -> None:
        self.window[:] = 0.0
        self.samples_seen = 0

    def payload_to_row(self, payload: dict) -> np.ndarray:
        """
        Expected raw row layout for model.py:
        [t_ms, ax, ay, az, gx, gy, gz, flex_raw, press_raw, vib_duty]

        model.process_raw_signal() currently uses df[:, 1:9],
        so it uses:
        ax, ay, az, gx, gy, gz, flex_raw, press_raw
        and ignores t_ms + vib_duty during preprocessing.
        """
        return np.array([
            float(payload.get("t_ms", 0)),
            float(payload.get("ax", 0.0)),
            float(payload.get("ay", 0.0)),
            float(payload.get("az", 0.0)),
            float(payload.get("gx", 0.0)),
            float(payload.get("gy", 0.0)),
            float(payload.get("gz", 0.0)),
            float(payload.get("flex_raw", 0.0)),
            float(payload.get("press_raw", 0.0)),
            float(payload.get("vib_duty", 0.0)),
        ], dtype=np.float32)

    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        x = x - np.max(x)
        exp_x = np.exp(x)
        denom = np.sum(exp_x)
        if denom <= 0:
            return np.zeros_like(exp_x)
        return exp_x / denom

    def push_payload(self, payload: dict) -> dict:
        row = self.payload_to_row(payload)

        self.window = np.roll(self.window, -1, axis=0)
        self.window[-1, :] = row
        self.samples_seen += 1

        if self.samples_seen < CNN.IN_LEN:
            return {
                "status": "warming_up",
                "samples_seen": self.samples_seen,
                "samples_needed": CNN.IN_LEN,
                "gesture": "warming_up",
                "confidence": 0.0,
            }

        processed = process_window(self.window)   # shape: (25, 30)
        pred_id, logits, metrics = self.cnn.predict_timed(processed.T)

        pred_id = int(pred_id)
        logits = np.asarray(logits, dtype=np.float64)
        probs = self.softmax(logits)
        confidence = float(probs[pred_id]) if 0 <= pred_id < len(probs) else 0.0

        gesture = self.id_to_gesture.get(pred_id, f"class_{pred_id}")

        return {
            "status": "ok",
            "samples_seen": self.samples_seen,
            "samples_needed": CNN.IN_LEN,
            "gesture": gesture,
            "confidence": confidence,
            "pred_id": pred_id,
            "logits": logits.tolist(),
            "timing": metrics,
        }