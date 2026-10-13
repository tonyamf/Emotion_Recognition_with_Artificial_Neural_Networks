"""Geometric feature pipeline: distance / shape / texture signatures and their nine statistics each.

Every test runs the notebook's own cells (see conftest.run) on tiny synthetic landmarks.
"""
import math

import numpy as np
import pandas as pd
import pytest

from conftest import base_namespace, make_landmarks

# (cell index, text that identifies it)
DIST = (17, 'df["distance"][j] = []')
DF_ = (20, 'df_ =pd.DataFrame(df[df[\'distance\'] != ""].copy())')
NORM_D = (21, 'df_["NormalizedDistance"] = ""')
STAB_D = (23, 'StabilityIndex_D')
STATS_D = (26, 'Skewness_D')
SHAPE = (29, 'df_["shape"][j] = []')
TEXTURE = (38, 'df_["texture"][j] = []')


def selected_points(landmarks):
    """Independent re-statement of the 25-point selection the notebook performs (chin skipped)."""
    pts = []
    for name, v in landmarks[0].items():
        if name == "chin":
            continue
        elif name == "top_lip":
            pts += [v[0], v[3], v[6]]
        elif name == "bottom_lip":
            pts.append(v[3])
        elif name == "nose_tip":
            pts += [v[0], v[4]]
        elif name == "nose_bridge":
            pts.append(v[3])
        elif name in ("left_eyebrow", "right_eyebrow"):
            pts += [v[0], v[2], v[4]]
        else:
            pts += list(v)
    return pts


def distance_ns(faces, run):
    ns = base_namespace()
    ns["df"] = faces
    return run(ns, DIST)


def test_selects_25_points_and_300_pairwise_distances(faces, run):
    ns = distance_ns(faces, run)
    assert len(selected_points(faces["landmarks"][0])) == 25
    for d in ns["df"]["distance"]:
        assert len(d) == 25 * 24 // 2  # C(25, 2)


def test_distances_are_euclidean_in_selection_order(faces, run):
    ns = distance_ns(faces, run)
    pts = selected_points(faces["landmarks"][0])
    got = ns["df"]["distance"][0]
    assert got[0] == pytest.approx(math.dist(pts[0], pts[1]))
    assert got[1] == pytest.approx(math.dist(pts[0], pts[2]))
    assert got[-1] == pytest.approx(math.dist(pts[23], pts[24]))


def test_image_without_face_gets_no_signature(faces, run):
    faces.at[2, "landmarks"] = []
    ns = distance_ns(faces, run)
    assert ns["df"]["distance"][2] == ""  # later cells drop these rows with `distance != ""`
    assert len(ns["df"]["distance"][0]) == 300


def distance_stats(faces, run):
    ns = distance_ns(faces, run)
    return run(ns, DF_, NORM_D, STAB_D, STATS_D)["df_"]


def test_normalised_distance_columns_sum_to_one_over_the_dataset(faces, run):
    df_ = distance_stats(faces, run)
    nd = np.array(list(df_["NormalizedDistance"]))
    assert nd.shape == (6, 300)
    assert nd.sum(axis=0) == pytest.approx(np.ones(300))


def test_statistics_match_scipy_on_the_normalised_vector(faces, run):
    import scipy.stats as st
    df_ = distance_stats(faces, run)
    v = np.array(df_["NormalizedDistance"][0])
    assert df_["Range_D"][0] == pytest.approx(v.max() - v.min())
    assert df_["Skewness_D"][0] == pytest.approx(st.skew(v))
    assert df_["Entropy_D"][0] == pytest.approx(st.entropy(v))


def test_first_central_moment_is_zero_so_three_of_the_27_features_are_dead(faces, run):
    """Moment_*_1 is the first CENTRAL moment (mean minus mean): always 0, one dead column per signature."""
    df_ = distance_stats(faces, run)
    assert np.allclose(df_["Moment_D_1"].astype(float), 0.0, atol=1e-12)


def test_stability_index_follows_the_written_formula(faces, run):
    df_ = distance_stats(faces, run)
    v = np.array(df_["NormalizedDistance"][0])
    # |v[d-10] + v[d] - 10 * (v[d-1] + ... + v[d-9])| summed for d >= 10
    expected = sum(abs(v[d - 10] + v[d] - 10 * v[d - 9:d].sum()) for d in range(10, len(v)))
    assert df_["StabilityIndex_D"][0] == pytest.approx(expected)


def shape_frame(landmarks, run):
    ns = base_namespace()
    ns["df_"] = pd.DataFrame({"landmarks": landmarks})
    return run(ns, SHAPE)["df_"]


def test_shape_value_is_area_over_squared_perimeter(run):
    lm = make_landmarks(np.random.RandomState(1), jitter=False)
    # selected point order starts left_eyebrow[0], [2], [4]: make those a 3-4-5 right triangle
    lm[0]["left_eyebrow"] = [(10, 10), (0, 0), (13, 10), (0, 0), (10, 14)]
    df_ = shape_frame([lm], run)
    assert df_["shape"][0][0] == pytest.approx(6 / 12 ** 2)  # area 6, perimeter 12


def test_shape_signature_is_scale_invariant(run):
    lm = make_landmarks(np.random.RandomState(3))
    small = [{k: [(x // 2, y // 2) for x, y in v] for k, v in lm[0].items()}]
    double = [{k: [(2 * x, 2 * y) for x, y in v] for k, v in small[0].items()}]
    df_ = shape_frame([small, double], run)
    a, b = np.array(df_["shape"][0]), np.array(df_["shape"][1])
    assert len(a) == len(b)
    assert a == pytest.approx(b)


@pytest.mark.xfail(strict=True, reason="BUG: a degenerate (collinear) triangle is skipped, so the signature is "
                                       "shorter than 2300 and every later value lands in the wrong column")
def test_shape_signature_has_a_fixed_length_with_collinear_points(run):
    lm = make_landmarks(np.random.RandomState(5))
    lm[0]["left_eyebrow"] = [(5, 5), (0, 0), (6, 6), (0, 0), (7, 7)]  # three selected points on one line
    df_ = shape_frame([lm], run)
    assert len(df_["shape"][0]) == 2300


@pytest.mark.xfail(strict=True, reason="BUG: a landmark outside the 48x48 image drops its pairs instead of "
                                       "being clipped, so the vector is shorter than 300 and positions no "
                                       "longer line up between faces")
def test_distance_signature_has_a_fixed_length_when_a_landmark_is_off_image(faces, run):
    faces["landmarks"][0][0]["left_eye"][0] = (-3, 20)
    ns = distance_ns(faces, run)
    assert len(ns["df"]["distance"][0]) == 300


def texture_codes(image, landmarks, run):
    ns = base_namespace()
    ns["df_"] = pd.DataFrame({"landmarks": [landmarks], "pixels": [image]})
    return run(ns, TEXTURE)["df_"]["texture"][0]


def test_texture_of_a_flat_image_is_code_255_everywhere(run):
    lm = make_landmarks(np.random.RandomState(7))
    code = texture_codes(np.full((48, 48), 100), lm, run)
    assert len(code) == 25
    assert set(code) == {255}  # no neighbour is darker than the centre average -> all 8 bits are 1


def test_texture_codes_stay_in_byte_range(faces, run):
    code = texture_codes(faces["pixels"][1], faces["landmarks"][1], run)
    assert len(code) == 25
    assert min(code) >= 0 and max(code) <= 255
