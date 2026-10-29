# Molecular Prediction from EI Mass Spectra

A neuro-symbolic pipeline for predicting molecular structure-related features from **electron ionization mass spectra (EI-MS)** and combining neural predictions with **symbolic chemical constraints**.

The project explores how data-driven molecular prediction can be strengthened by integrating **multitask deep learning**, **uncertainty calibration**, and **Answer Set Programming (ASP)**. Neural models first infer molecular descriptors and structural properties from spectra; these predictions can then be translated into symbolic predicates and used to constrain chemically plausible molecular candidates.

## Project Overview

Inferring molecular structure from a mass spectrum is an underdetermined problem: different molecules can produce similar fragmentation patterns, while purely neural predictions may violate known chemical constraints.

This project addresses the problem through a hybrid architecture:

```text
EI mass spectrum
      ↓
Neural feature prediction
      ↓
Calibrated prediction sets / intervals
      ↓
Symbolic predicate compilation
      ↓
ASP chemical constraints
      ↓
Chemically consistent molecular candidates
```

The repository therefore combines two complementary forms of reasoning:

- **sub-symbolic learning**, used to extract structural information from spectra;
- **symbolic reasoning**, used to encode and enforce chemical knowledge.

## Neural Models

The core models are implemented in PyTorch and use a multitask formulation in which a shared representation of the spectrum feeds several prediction heads.

### Multitask MLP

`MultitaskMLP` uses a shared fully connected backbone with separate heads for:

- **binary molecular features**, including MACCS structural keys;
- **categorical molecular features**, represented as classification tasks.

### Multitask MLP with Quantile Regression

`MultitaskMLPWithQRegression` extends the base architecture with dedicated **quantile-regression heads** for continuous molecular quantities.

The quantile heads enforce monotonic outputs by predicting the first quantile directly and subsequent quantiles as positive offsets. This provides prediction intervals while avoiding quantile crossing.

### Multitask 1D CNN

The repository also contains a convolutional alternative, `MultitaskCNN`, which operates directly on the one-dimensional mass-spectrum signal using stacked **Conv1D layers**, global pooling and multitask output heads.

This allows comparison between dense and convolutional representations of spectral information.

## Uncertainty and Conformal Calibration

Prediction uncertainty is treated as part of the pipeline rather than as a post-processing detail.

The training workflow separates the data into:

- training;
- validation;
- calibration;
- test sets.

After model training, the calibration split is used to compute **conformal thresholds**. These thresholds are saved alongside the model checkpoints and support calibrated prediction sets or intervals that can subsequently be passed to the symbolic component.

For continuous targets, the project combines **quantile regression** with conformal calibration.

## Neuro-Symbolic Layer

The symbolic part of the system is implemented using **Answer Set Programming** with `clingo`.

The `src/predicates/` package provides the interface between model outputs and symbolic predicates, while `src/asp/` contains the ASP representation and the bridge to the solver.

The symbolic layer is designed to convert uncertain neural predictions into logical constraints describing admissible molecular properties. These constraints can then be combined with domain rules to restrict the search space to chemically consistent structures.

Key components include:

- predicate definitions and compilation;
- neural-to-symbolic prediction interfaces;
- an ASP bridge for solver interaction;
- chemical constraints encoded in `smiles.lp`;
- utilities for visualising generated SMILES structures.

## Training Pipeline

The main MLP training pipeline is implemented in:

```text
src/model/run_training.py
```

At a high level, it:

1. loads the configured spectrum dataset and feature schema;
2. constructs the appropriate multitask model from the dataset definition;
3. creates or reloads train/validation/calibration/test splits;
4. normalises continuous targets when required;
5. trains and checkpoints the neural model;
6. calibrates the trained model on the held-out calibration split;
7. stores conformal thresholds for downstream inference.

A separate training entry point is provided for the CNN architecture.

## Repository Structure

```text
.
├── src/
│   ├── asp/                  # ASP rules, solver bridge and SMILES utilities
│   ├── data/                 # dataset loading, preprocessing and splitting
│   ├── model/                # MLP/CNN models, training, prediction and calibration
│   ├── predicates/           # neural prediction → symbolic predicate interface
│   ├── tests/                # project tests
│   ├── config.py
│   └── filters.py
├── data/                     # datasets and generated data artifacts
├── notebooks/                # exploratory and experimental notebooks
├── scripts/                  # supporting scripts
├── docs/                     # project documentation
├── calibrate.py              # calibration utilities / entry point
├── evaluate.py               # model evaluation
├── evaluate_molecules.py     # molecule-level evaluation
├── infer.py                  # inference pipeline
├── config.yaml               # experiment configuration
├── requirements.txt
└── pyproject.toml
```

## Installation

The package requires **Python 3.10 or newer**.

Create and activate a virtual environment, then install the dependencies and the project in editable mode:

```bash
python -m venv env
source env/bin/activate
pip install -r requirements.txt
pip install -e .
```

The project is packaged as `molpred`, with source packages discovered from `src/`. Installing it in editable mode makes modules such as `model`, `predicates`, `asp` and `data` importable directly during development.

## Configuration

Experiment settings are defined in:

```text
config.yaml
```

The configuration controls dataset paths, model hyperparameters, training parameters, checkpoint locations and conformal-calibration settings.

## Running the Project

The exact workflow depends on the dataset and experiment configuration, but the main components are organised around the following stages:

```text
prepare data
    ↓
train neural model
    ↓
calibrate uncertainty
    ↓
evaluate / infer molecular features
    ↓
compile predictions into symbolic constraints
    ↓
run ASP-based molecular reasoning
```

For the main neural training pipeline:

```bash
python -m model.run_training
```

Additional top-level scripts provide dedicated entry points for calibration, inference and evaluation.

## Technology Stack

- **Python**
- **PyTorch**
- **pandas**
- **scikit-learn / scientific Python tooling**
- **clingo / Answer Set Programming**
- **multitask learning**
- **1D convolutional neural networks**
- **quantile regression**
- **conformal prediction**
- **neuro-symbolic reasoning**

## Research Perspective

The project is motivated by the idea that molecular identification benefits from combining statistical learning with explicit domain knowledge. Neural networks are effective at extracting patterns from spectral data, while symbolic constraints provide a mechanism for enforcing structural consistency and incorporating chemical rules that are difficult to learn reliably from data alone.

The resulting system is therefore best viewed as a **neuro-symbolic molecular prediction framework**, rather than as a standalone classifier or regressor.

## License

See [`LICENSE`](./LICENSE) for licensing information.
