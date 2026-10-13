"""Helpers that run the project's own notebook cells on tiny synthetic data.

The project has no importable module: all logic lives in cwANN_.ipynb. Instead of copying that logic
into a second place (where it could drift), the tests extract the real cells by a snippet of their source
and `exec` them in a namespace holding small synthetic DataFrames. No dataset, dlib or network is needed.
"""
import json
import math
import os
import pathlib

import numpy as np
import pandas as pd
import pytest

NOTEBOOK = pathlib.Path(__file__).resolve().parents[1] / "cwANN_.ipynb"

# face_recognition.face_landmarks() returns these parts, in this order, with these point counts.
PARTS = [("chin", 17), ("left_eyebrow", 5), ("right_eyebrow", 5), ("nose_bridge", 4), ("nose_tip", 5),
         ("left_eye", 6), ("right_eye", 6), ("top_lip", 12), ("bottom_lip", 12)]


@pytest.fixture(scope="session")
def cells():
    nb = json.load(open(NOTEBOOK))
    return ["".join(c["source"]) for c in nb["cells"]]


@pytest.fixture
def run(cells):
    """run(ns, (42, "text in cell 42"), ...) executes those notebook cells, in the order given, inside `ns`.

    The snippet guards against the notebook being re-ordered: if cell 42 no longer contains it, the test
    fails with a clear message instead of silently testing the wrong code.
    """
    def _run(ns, *steps):
        for idx, snippet in steps:
            assert snippet in cells[idx], f"cell {idx} no longer contains {snippet!r}"
            exec(compile(cells[idx], f"<cwANN_.ipynb cell {idx}>", "exec"), ns)
        return ns
    return _run


def make_landmarks(rng, scale=1.0, offset=0, jitter=True):
    """One face's landmarks in face_recognition format; all points land inside the 48x48 image."""
    out = {}
    for name, k in PARTS:
        pts = rng.randint(6, 40, size=(k, 2)) if jitter else np.full((k, 2), 20)
        out[name] = [(int(x * scale) + offset, int(y * scale) + offset) for x, y in pts]
    return [out]


def base_namespace():
    import scipy.stats
    return {"np": np, "pd": pd, "math": math, "os": os, "scipy": scipy}


@pytest.fixture
def faces():
    """A 6-row frame shaped like the notebook's `df` after the landmark step: 4 train rows, 2 test rows."""
    rng = np.random.RandomState(0)
    n = 6
    df = pd.DataFrame({"emotion": [0, 3, 4, 6, 3, 0],
                       "usage": ["Training"] * 4 + ["PublicTest", "PrivateTest"]})
    df["pixels"] = [rng.randint(0, 256, (48, 48)) for _ in range(n)]
    df["landmarks"] = [make_landmarks(rng) for _ in range(n)]
    df["distance"] = ""
    return df
