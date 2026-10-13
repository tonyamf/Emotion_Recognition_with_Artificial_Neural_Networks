# Emotion Recognition with Artificial Neural Networks

This repository contains the code and resources for the research paper, "Geometric and Appearance approaches with Artificial Neural Network for Discrete Human Emotion Recognition from Static Face Images." The project explores and implements two primary approaches for recognizing human emotions from static facial images.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
  - [Geometric-Based Approach](#geometric-based-approach)
  - [Appearance-Based Approach](#appearance-based-approach)
- [Code Structure](#code-structure)
- [Setup and Installation](#setup-and-installation)
- [How to run](#how-to-run)
- [Changes to make it run today](#changes-to-make-it-run-today-hubrefine)
- [Results](#results-already-in-the-repo-produced-on-colab-by-the-original-author)
- [Contributing](#contributing)

## Project Overview

This project aims to recognize discrete human emotions from static face images using Artificial Neural Networks. It delves into two main feature extraction techniques:
- **Geometric-based:** Analyzing key facial features and their geometric properties.
- **Appearance-based:** Utilizing the entire facial image to learn features.

The project uses the Fer2013 dataset and implements both a Multi-Layer Perceptron (MLP) for geometric features and a Convolutional Neural Network (CNN) for appearance-based features.

## Dataset

The project uses the **Fer2013 dataset**, which was part of a Kaggle competition. The dataset consists of 48x48 pixel grayscale images of faces. The task is to categorize each face based on the emotion shown in the facial expression into one of seven categories:
- 0: Angry
- 1: Disgust
- 2: Fear
- 3: Happy
- 4: Sad
- 5: Surprise
- 6: Neutral

## Methodology

### Geometric-Based Approach

This approach involves:
1.  **Facial Landmark Detection:** Identifying key points on the face, such as the corners of the eyes and mouth.
2.  **Feature Extraction:** Calculating geometric properties from these landmarks:
    - **Distance Signature:** Distances between all pairs of keypoints.
    - **Shape Signature:** Signatures derived from triangles formed by triplets of keypoints.
    - **Texture Signature:** A custom Local Binary Pattern (LBP) around keypoints.
3.  **Statistical Analysis:** Computing statistical measures like skewness, kurtosis, moments, and entropy for these signatures.
4.  **Model Training:** Training a Multi-Layer Perceptron (MLP) on these extracted features.

### Appearance-Based Approach

This method leverages the power of deep learning:
1.  **Image Preprocessing:** The 48x48 pixel grayscale images are used directly as input.
2.  **Feature Extraction and Classification:** A Convolutional Neural Network (CNN) is used to automatically learn and extract relevant features from the images and classify them into the seven emotion categories.

## Code Structure

| File | What it is |
|---|---|
| `cwANN_.ipynb` | The project: geometric features -> MLP, CNN on pixels, and a combined MLP+CNN model (originally run on Google Colab). |
| `cwANN.pdf` | The coursework paper. |
| `requirements.txt` | Pinned dependencies (tested with Python 3.11). |
| `smoke_data.py`, `run_smoke_test.sh` | Smoke test on tiny synthetic data (no dataset needed). |
| `Thesis_seg.ipynb`, `Untitled-1.ipynb` | Not part of this project's pipeline: a Colab experiment on a breast-cancer-mammogram segmentation thesis (`Thesis_seg`) and early exploratory cells for this project (`Untitled-1`). Kept as uploaded, not fixed, they do not run. |

Notebook flow: load `icml_face_data.csv` -> face landmarks with `face_recognition` (dlib) -> distance / shape / texture
signatures and 9 statistics each (27 geometric features) -> MLP tuned with `BayesSearchCV` -> CNN on the 48x48 images ->
CNN + MLP combined. Cells after the `# Else` heading are abandoned scratch work and are expected to fail.

## Setup and Installation

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt     # dlib is compiled from source (cmake + C++ compiler needed; ~6 min)
```

## How to run

1. Get the **Fer2013** `icml_face_data.csv` from [Kaggle](https://www.kaggle.com/datasets/deadskull7/fer2013) (it is *not* in this repo)
   and put it in a folder, e.g. `data/`.
2. `export EMOTION_DATA_DIR=data` (default is the current folder; on Colab it is the original Drive folder
   `/content/drive/MyDrive/Colab Notebooks`), then open `cwANN_.ipynb` in Jupyter and run the cells down to `# Else`.
   Landmark detection over the whole dataset and the Bayesian search are slow (hours on CPU).
3. **Smoke test, no dataset needed:** `./run_smoke_test.sh` executes the notebook on 300 synthetic random images
   (stub landmarks, 1 epoch, 1 search iteration). It only shows that the code runs; its accuracy is chance-level and means nothing.

## Changes to make it run today (HubRefine)

- Colab-only pieces made optional (Drive mount, hard-coded `/content/drive/...` paths -> `DATA_DIR`).
- `keras.wrappers.scikit_learn.KerasClassifier` no longer exists -> `scikeras` (search keys now `model__*`).
- `from scipy.stats import *` shadowed the builtin `abs()` in current SciPy -> removed.
- `Adam(lr=)`, `ReduceLROnPlateau(model=)`, `.h5` checkpoint name, removed `keras.utils.vis_utils` import updated for Keras 3.
- Fixed a syntax error (`ImageDataGenerator(horizontal_flip=True, br)`), an undefined variable (`dfO_`), the CNN pixel column
  (`df['pixels']` was already an array), and `model.evaluate(...)` in the combined model, which was called without labels.
  The recorded `Test Accuracy: 0.0` output of that cell came from this bug; the notebook's own classification report gives the combined model 0.68.
- `n_jobs` of the Bayesian search is `-1` (all cores), as the old README claimed (the notebook used 25).
- Pinned `pandas==2.2.3`: pandas 3 silently ignores the chained assignments the notebook relies on.

## Results (already in the repo, produced on Colab by the original author)

These numbers are the recorded outputs in `cwANN_.ipynb` / `cwANN.pdf`. They were **not** reproduced here: the Fer2013 data
is not in the repo and was not downloaded, so treat them as the original claims.

| Model | Accuracy | Source |
|---|---|---|
| MLP, 27 geometric features (best of Bayesian search: tanh, adam, MSLE loss, dropout 0.35) | 0.365 CV score on train; 0.36 on test (4958 images with detected landmarks) | `cwANN_.ipynb` cells 64 and 69; `cwANN.pdf` ("37% training, 36% testing") |
| CNN on pixels | 0.666 on the (flip-augmented) test generator, 7178 images | `cwANN_.ipynb` cell 85; `cwANN.pdf` (72.5% train / 66.6% test) |
| CNN + MLP combined | 0.68 on test (4958 images with detected landmarks) | `cwANN_.ipynb` cell 108; `cwANN.pdf` (80.4% train / 68.1% test) |

Caveats: the three rows are not measured on the same test images (the geometric and combined models only see faces where
landmarks were found, 4958 of the 7178 test images), and the CNN row evaluates on a generator that applies random horizontal flips.
So "the combined model is better" (0.68 vs 0.666) is suggestive, not a controlled comparison. One recorded output
(`len(df_train_X)` = 39249) is larger than the 28709 Fer2013 training images and comes from a different run; its source is unknown.

## Contributing

Contributions are welcome! If you have any suggestions, improvements, or bug fixes, please feel free to open an issue or submit a pull request.
