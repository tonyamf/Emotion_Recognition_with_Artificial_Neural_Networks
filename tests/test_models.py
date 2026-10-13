"""The three Keras architectures, built from the notebook's own cells and run on tiny random arrays."""
import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
import keras  # noqa: E402
from keras import layers  # noqa: E402
from keras.layers import (Activation, BatchNormalization, Concatenate, Conv2D, Dense, Dropout, Flatten,  # noqa: E402
                          Input, MaxPooling2D)
from keras.models import Model, Sequential  # noqa: E402
from keras.optimizers import Adam  # noqa: E402

from conftest import base_namespace  # noqa: E402


@pytest.fixture
def ns():
    n = base_namespace()
    n.update(keras=keras, layers=layers, Sequential=Sequential, Model=Model, Input=Input, Dense=Dense,
             Dropout=Dropout, Flatten=Flatten, Conv2D=Conv2D, BatchNormalization=BatchNormalization,
             Activation=Activation, MaxPooling2D=MaxPooling2D, Adam=Adam, Concatenate=Concatenate)
    return n


def test_mlp_has_the_documented_layers_and_parameter_count(ns, run):
    run(ns, (61, "def DST_network"))
    net = ns["DST_network"]("adam", "categorical_crossentropy", "tanh", 0.0, 0.35)
    dense = [l for l in net.layers if isinstance(l, Dense)]
    assert [l.units for l in dense] == [512, 64, 16, 7]
    assert net.count_params() == 48327  # same total as the model.summary() recorded in the notebook
    assert net.input_shape == (None, 27)
    assert dense[-1].activation.__name__ == "softmax"


def test_mlp_outputs_a_probability_distribution_and_trains(ns, run):
    run(ns, (61, "def DST_network"))
    net = ns["DST_network"]("adam", "categorical_crossentropy", "relu", 0.01, 0.1)
    rng = np.random.RandomState(0)
    x = rng.randn(32, 27).astype("float32")
    y = tf.keras.utils.to_categorical(rng.randint(0, 7, 32), 7)
    p = net.predict(x, verbose=0)
    assert p.shape == (32, 7) and np.allclose(p.sum(axis=1), 1.0, atol=1e-5)
    hist = net.fit(x, y, epochs=2, batch_size=16, verbose=0)
    assert np.isfinite(hist.history["loss"]).all()


@pytest.mark.parametrize("loss", ["categorical_crossentropy", "categorical_hinge", "mean_squared_logarithmic_error"])
def test_mlp_accepts_every_loss_in_the_search_space(ns, run, loss):
    run(ns, (61, "def DST_network"))
    ns["DST_network"]("sgd", loss, "relu", 0.0, 0.0)  # compile() would raise on an unknown loss name


def test_cnn_maps_48x48_images_to_7_class_probabilities(ns, run):
    run(ns, (80, "model = Sequential()"))
    model = ns["model"]
    x = np.random.RandomState(1).rand(4, 48, 48, 1).astype("float32")
    p = model.predict(x, verbose=0)
    assert p.shape == (4, 7) and np.allclose(p.sum(axis=1), 1.0, atol=1e-5)
    assert model.count_params() == 4478727  # total recorded in the notebook's model.summary()
    flatten = [l for l in model.layers if isinstance(l, Flatten)][0]
    assert flatten.output.shape[-1] == 3 * 3 * 512  # 48 +6 padding -> 54 -> 27 -> 13 -> 6 -> 3 after 4 poolings


def test_cnn_takes_one_training_step(ns, run):
    run(ns, (80, "model = Sequential()"))
    model = ns["model"]
    rng = np.random.RandomState(2)
    x = rng.rand(8, 48, 48, 1).astype("float32")
    y = tf.keras.utils.to_categorical(rng.randint(0, 7, 8), 7)
    loss, acc = model.train_on_batch(x, y)
    assert np.isfinite(loss) and 0.0 <= acc <= 1.0


def test_combined_model_has_two_inputs_and_fuses_cnn_with_mlp_features(ns, run):
    run(ns, (104, "geo = Input(shape=(27,))"))
    model = ns["model"]
    assert [tuple(i.shape[1:]) for i in model.inputs] == [(48, 48, 1), (27,)]
    concat = [l for l in model.layers if isinstance(l, Concatenate)][0]
    assert concat.output.shape[-1] == 512 + 16  # CNN dense-512 branch + MLP's 16-unit last hidden layer
    rng = np.random.RandomState(3)
    img = rng.rand(4, 48, 48, 1).astype("float32")
    geo = rng.randn(4, 27).astype("float32")
    p = model.predict([img, geo], verbose=0)
    assert p.shape == (4, 7) and np.allclose(p.sum(axis=1), 1.0, atol=1e-5)


def test_combined_model_output_depends_on_the_geometric_input(ns, run):
    run(ns, (104, "geo = Input(shape=(27,))"))
    model = ns["model"]
    rng = np.random.RandomState(4)
    img = rng.rand(2, 48, 48, 1).astype("float32")
    a = model.predict([img, np.zeros((2, 27), "float32")], verbose=0)
    b = model.predict([img, 5 * np.ones((2, 27), "float32")], verbose=0)
    assert not np.allclose(a, b)
