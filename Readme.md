# Neural Networks and Deep Learning – Homework Challenges

University project for the **Neural Networks and Deep Learning** course, developed by group **TheDataArchitects**.

The project consists of two Kaggle/Codalab-style challenges:

1. **Challenge 1** – Image classification of healthy vs. unhealthy plants (CNNs, transfer learning, ensembling).
2. **Challenge 2** – Time series forecasting (recurrent networks, detrending / stationarization).

## Authors

- Antonio Napolitano
- Carlo Dell'Orco
- Federico Caroli
- Gabriele Farace

## Repository structure

```
.
├── Challenge_1/   # Plant health classification
├── Challenge_2/   # Time series forecasting
├── README.md
└── .gitignore
```

---

## Challenge 1 – Plant Health Classification

**Task.** Binary classification of plant images (`0 = healthy`, `1 = unhealthy`). The provided dataset contained outliers (images of Shrek and Trollface, always the same two pictures) as well as many duplicates.

### Approach

- **Data cleaning:** outliers and duplicates were removed with a custom script, reducing the dataset from 5200 to 4850 images (3060 healthy, 1790 unhealthy).
- **Class imbalance:** the training set was rebalanced by oversampling the unhealthy class with random duplicates (applied *after* the train/validation split).
- **Data augmentation:** rotation, translation and flips (zoom, contrast and crop were tried but hurt performance).
- **Models:** a ResNet34-style network trained from scratch overfitted heavily, so we moved to **transfer learning + fine-tuning** with ImageNet-pretrained models (VGG16, MobileNetV2, EfficientNet B0/B2/V2S, InceptionResNetV2, ConvNeXt Tiny/Small/Base/Large).
- **Classifier head:** 2 dense layers (64 and 32 units), batch normalization, ReLU, dropout 1/8.
- **Training:** two Adam optimizers (lr 0.01 for transfer learning, 0.001 for fine-tuning with ~60% of the backbone frozen), early stopping and LR scheduler.
- **Final submission:** an **ensemble of 5 ConvNeXtBase models** (original 96x96 images, plus inputs resized to 96, 128, 200 and 250 pixels), combined by **majority voting**.

Validation accuracy reached roughly 0.90 for the best individual networks.

### Contents

| File | Description |
|------|-------------|
| `model.py` | `model` class used for the Codalab submission; loads the 5 models and predicts by majority vote |
| `ConvNextBase.ipynb` | Creation and training of the 5 ConvNeXtBase models of the final ensemble |
| `EfficientNetV2S.ipynb` | Best EfficientNetV2S model from the development phase |
| `DeleteShrekAndTrololoDuplicatesFromDataset.ipynb` | Script that removes outliers and duplicates from the dataset |

The notebook used for the ResNet-from-scratch experiment (`ResNetFromZero`) and the EfficientNet experiments may also be included, depending on the version of the archive.

### Dataset

⚠️ **The dataset for this challenge is too big to be shared** thus is not included in this repository.

---

## Challenge 2 – Time Series Forecasting

**Task.** Forecast future samples of unrelated time series (input length 200, 6 categories A–F). The forecasting horizon (*telescope*) was 9 samples in the development phase and 18 in the final phase. Models are evaluated with MSE.

### Approach

- **Preprocessing:** padding removal using the provided valid periods, grouping by category, 90/10 train/validation split at the series level.
- **Sequence generation:** `build_sequences` and `compute_sequences_for_dataset` split each series into windows (input + target of length `telescope`) to enlarge the training set. A small stride is used for training and a stride equal to the window for validation, to avoid an optimistic validation MSE.
- **Network:** masking layer, three bidirectional LSTMs (256, 128, 64 units), and a dense output layer with `telescope` neurons. MSE loss, Adam, EarlyStopping and ReduceLROnPlateau.
- **Series handling approaches:**
  1. Original series.
  2. Stationary series (trend removed with a rolling mean, seasonality removed by differencing, period found via autocorrelation) – this performed *worse*, since not all series are actually seasonal.
  3. **Detrended series only** – slightly better than the original series and our best approach.
- **Per-category models vs. a single model:** a single model trained on all categories performed better (category F has too few series).
- **Other experiments:** autoregressive per-category models (poor results) and models combining 1D convolutions with bidirectional LSTMs (not submitted).
- **Final submission:** an **ensemble of two models** trained on detrended series with different train/validation splits.

### Contents

| File | Description |
|------|-------------|
| `Official_data_preprocessing.ipynb` | Dataset exploration; creation of detrended and stationary series |
| `Recurrent_network.ipynb` | Bidirectional LSTM model, trainable on original, detrended or stationary series (selected through a variable in the notebook) |
| `One_dimensional_convolutiol_network.ipynb` | Experimental model combining 1D convolutions and bidirectional LSTMs |

### Dataset

> ⚠️ **The dataset for this challenge has been lost** and is not included in this repository.

---

## Requirements

The notebooks were developed mainly on **Google Colab** with Python 3 and:

- TensorFlow / Keras
- NumPy
- Matplotlib
- scikit-learn (where used in the notebooks)

## Notes

This repository is an archive of a university project. The code is published as it was submitted, with no further refactoring.