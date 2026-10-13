"""Synthetic stand-ins used ONLY by the smoke test (EMOTION_SMOKE=1).

No FER2013 data is downloaded. `make_csv` writes a tiny random icml_face_data.csv-shaped file and
`fake_landmarks` replaces `face_recognition.face_landmarks`, which finds no faces in random noise.
Numbers produced from this data say nothing about model quality.
"""
import numpy as np
import pandas as pd

_PARTS = {'chin': 17, 'left_eyebrow': 5, 'right_eyebrow': 5, 'nose_bridge': 4, 'nose_tip': 5,
          'left_eye': 6, 'right_eye': 6, 'top_lip': 12, 'bottom_lip': 12}


def make_csv(path, n_train=240, n_test=60, seed=0):
    rng = np.random.RandomState(seed)
    n = n_train + n_test
    usage = ['Training'] * n_train + ['PublicTest'] * (n_test // 2) + ['PrivateTest'] * (n_test - n_test // 2)
    pixels = [' '.join(map(str, rng.randint(0, 256, 48 * 48))) for _ in range(n)]
    labels = np.arange(n) % 7  # every class present in train and test
    pd.DataFrame({'emotion': labels, ' Usage': usage, ' pixels': pixels}).to_csv(path, index=False)


def fake_landmarks(img):
    """Return face_recognition-style landmarks (or [] for ~30% of images, like a failed detection)."""
    s = int(img.sum())
    if s % 10 >= 7:
        return []
    rng = np.random.RandomState(s % (2 ** 31))
    out = {}
    for name, k in _PARTS.items():
        pts = rng.randint(2, 46, size=(k, 2))
        out[name] = [(int(x), int(y)) for x, y in pts]
    return [out]
