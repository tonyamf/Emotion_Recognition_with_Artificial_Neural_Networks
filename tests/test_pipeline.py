"""From landmark frame to the MLP's scaled 27-column matrix, and the image preprocessing helpers."""
import numpy as np
import pandas as pd
import pytest

from conftest import base_namespace

FEATURE_CELLS = [
    (20, 'df_ =pd.DataFrame(df[df[\'distance\'] != ""].copy())'),
    (21, 'NormalizedDistance'), (23, 'StabilityIndex_D'), (26, 'Skewness_D'),
    (29, 'df_["shape"][j] = []'), (31, 'NormalizedShape'), (33, 'StabilityIndex_S'), (35, 'Skewness_S'),
    (38, 'df_["texture"][j] = []'), (41, 'NormalizedTexture'), (43, 'StabilityIndex_T'), (45, 'Skewness_T'),
]


@pytest.fixture
def mlp_inputs(faces, run, tmp_path):
    """Run feature cells -> save CSV -> reload -> select, split, scale, one-hot (cells 17..59)."""
    ns = base_namespace()
    ns["df"] = faces
    ns["DATA_DIR"] = str(tmp_path)
    run(ns, (17, 'df["distance"][j] = []'), *FEATURE_CELLS,
        (47, "DataFrame.csv"), (50, "DataFrame.csv"), (51, "StabilityIndex_D"), (54, "df_train_Y"))
    from sklearn.preprocessing import StandardScaler
    import tensorflow as tf
    ns.update(StandardScaler=StandardScaler, tf=tf)
    run(ns, (56, "StandardScaler()"), (57, "float32"), (59, "to_categorical"))
    return ns


def test_there_are_27_feature_columns_and_none_is_missing(mlp_inputs):
    ns = mlp_inputs
    assert ns["df_train_X"].shape == (4, 27)
    assert ns["df_test_X"].shape == (2, 27)
    assert np.isfinite(ns["df_train_X"]).all() and np.isfinite(ns["df_test_X"]).all()
    assert ns["df_train_X"].dtype == np.float32


def test_split_follows_the_usage_column(mlp_inputs):
    ns = mlp_inputs
    assert list(ns["df_train_Y"]) == [0, 3, 4, 6]  # 'Training' rows
    assert list(ns["df_test_Y"]) == [3, 0]  # PublicTest + PrivateTest are both treated as "test"


def test_scaler_is_fitted_on_training_rows_only(mlp_inputs):
    ns = mlp_inputs
    col = ns["df_train_X"][:, 0]
    assert col.mean() == pytest.approx(0.0, abs=1e-5)  # train is centred ...
    assert abs(ns["df_test_X"][:, 0].mean()) > 1e-6  # ... the test rows are not re-centred on themselves


def test_labels_are_one_hot_over_seven_classes(mlp_inputs):
    ns = mlp_inputs
    assert ns["y_train"].shape == (4, 7) and ns["y_test"].shape == (2, 7)
    assert list(ns["y_train"].argmax(axis=1)) == [0, 3, 4, 6]


def test_image_helpers_turn_a_pixel_string_into_48x48(run):
    pytest.importorskip("cv2")
    ns = base_namespace()
    import cv2
    ns["cv2"] = cv2
    run(ns, (74, "def string_to_array"))
    s = " ".join(str(i % 256) for i in range(48 * 48))
    arr = ns["string_to_array"](s)
    assert arr.shape == (48, 48, 1) and arr.dtype == np.float32
    img = ns["resize"](arr)
    assert img.shape == (48, 48)
    assert np.allclose(img, arr[:, :, 0])  # resizing 48x48 to 48x48 is the identity
