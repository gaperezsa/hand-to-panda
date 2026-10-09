"""
The model used to extract the hand landmarks from RGB, and the two signals the retargeting
needs from them.

MediaPipe gives 21 landmarks per hand. I need exactly two things:

  * the GRASP POINT -- the midpoint of the thumb tip and index tip. Not the wrist. The
    Panda's end-effector site sits between its fingers. I assumed the human wrist is ~10 cm behind
    that and rotated away from it, so retargeting the wrist puts the gripper in the
    wrong place.
  * the PINCH APERTURE -- I need the thumb-tip to index-tip distance. LIBERO's gripper command is
    binary in every reference demo, so this gets thresholded to know if the gripper is closed or open.

This module stays in 2D. Lifting to metric 3D needs the LiDAR depth and the camera pose,
which belong with the capture, and I will do the lift there, not here.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

WRIST, THUMB_TIP, INDEX_TIP = 0, 4, 8
DEFAULT_MODEL = Path("data/models/hand_landmarker.task")


@dataclass
class HandFrame:
    ok: bool
    landmarks_px: np.ndarray | None = None   # (21, 2) pixel coordinates
    world: np.ndarray | None = None          # (21, 3) metres, hand-centred (MediaPipe's own)
    score: float = 0.0
    handedness: str = ""

    @property
    def grasp_px(self) -> np.ndarray:
        return (self.landmarks_px[THUMB_TIP] + self.landmarks_px[INDEX_TIP]) / 2.0

    @property
    def pinch_px(self) -> float:
        return float(np.linalg.norm(self.landmarks_px[THUMB_TIP] - self.landmarks_px[INDEX_TIP]))

    @property
    def pinch_m(self) -> float:
        """Thumb-index distance from MediaPipe's own metric hand model, in metres.

        Independent of depth, so it is a useful cross-check on the depth-lifted value --
        but it is a generic hand, not mine, so prefer the depth-derived number and
        use this to catch gross errors.
        """
        return float(np.linalg.norm(self.world[THUMB_TIP] - self.world[INDEX_TIP]))


class HandTracker:
    def __init__(self, model_path: str | Path = DEFAULT_MODEL, num_hands: int = 1,
                 min_detection: float = 0.3, min_tracking: float = 0.3):
        import mediapipe as mp
        from mediapipe.tasks import python as mpp
        from mediapipe.tasks.python import vision
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"{model_path} missing. Fetch it with:\n  curl -L -o {model_path} "
                "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                "hand_landmarker/float16/latest/hand_landmarker.task")
        self._mp = mp
        self._lm = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=mpp.BaseOptions(model_asset_path=str(model_path)),
                running_mode=vision.RunningMode.VIDEO, num_hands=num_hands,
                min_hand_detection_confidence=min_detection,
                min_tracking_confidence=min_tracking))

    def __call__(self, rgb: np.ndarray, timestamp_ms: int) -> HandFrame:
        img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB,
                             data=np.ascontiguousarray(rgb))
        res = self._lm.detect_for_video(img, int(timestamp_ms))
        if not res.hand_landmarks:
            return HandFrame(ok=False)
        h, w = rgb.shape[:2]
        lms = res.hand_landmarks[0]
        px = np.array([[l.x * w, l.y * h] for l in lms], dtype=np.float64)
        world = np.array([[l.x, l.y, l.z] for l in res.hand_world_landmarks[0]], dtype=np.float64)
        hd = res.handedness[0][0] if res.handedness else None
        return HandFrame(True, px, world, float(hd.score) if hd else 0.0,
                         hd.category_name if hd else "")

    def close(self):
        self._lm.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
