# NYC Airbnb Room Type Predictor

Predicting the room type (Entire home/apt, Private room, Shared room, etc.) from Airbnb listing features in New York City — an end-to-end exploratory data analysis and machine learning project delivered as reproducible Jupyter Notebooks.

[![Notebook](https://img.shields.io/badge/Jupyter-Notebooks-orange)]() [![Status](https://img.shields.io/badge/status-experimental-yellow)]()

Table of contents
- [Overview](#overview)
- [Highlights](#highlights)
- [Dataset](#dataset)
- [What you'll find in the repo](#what-youll-find-in-the-repo)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Notebooks & Workflow](#notebooks--workflow)
- [Modeling approach](#modeling-approach)
- [Reproducibility](#reproducibility)
- [Evaluation & Results](#evaluation--results)
- [How to use the model / inference](#how-to-use-the-model--inference)
- [Tips & troubleshooting](#tips--troubleshooting)
- [Future work](#future-work)
- [Contributing](#contributing)
- [Acknowledgements & Data License](#acknowledgements--data-license)
- [Contact](#contact)

Overview
--------
This project demonstrates a pragmatic machine learning workflow for predicting the Airbnb room type for listings in New York City using public listing features (location, price, amenities, review stats, host info, etc.). It walks through data ingestion, cleaning, exploratory analysis, feature engineering, model training, evaluation, and basic deployment-ready export of predictions — all in reproducible Jupyter Notebooks.

Highlights
----------
- Clean, annotated Jupyter Notebooks for EDA, feature engineering, and modeling.
- End-to-end ML pipeline that can be adapted to other city datasets.
- Emphasis on interpretability: feature importance, partial dependence, and confusion analysis.
- Reproducible environment instructions and tips to run notebooks headlessly.

Dataset
-------
The repo is built around the NYC Airbnb public dataset (commonly available via NYC Open Data and Kaggle). The dataset typically includes:
- listing id, name, host information
- neighbourhood, neighbourhood_group
- latitude, longitude
- room_type (target)
- price, minimum_nights, availability, number_of_reviews
- last_review, reviews_per_month
- accommodations, bathrooms, bedrooms, beds
- amenities (if present) and host_response_rate

If the dataset is not included in the repo, download it from the original source (add the CSV(s) into a `data/` folder) and follow the filenames referenced in the notebooks.

What you'll find in the repo
---------------------------
- notebooks/
  - 01-EDA.ipynb         — data loading, cleaning, visual exploration
  - 02-feature-engineering.ipynb — transformations, encoding, feature creation
  - 03-modeling.ipynb    — model training, hyperparameter tuning, validation
  - 04-interpretation.ipynb — feature importance, error analysis
- data/ (gitignored)      — place raw and processed CSVs here
- requirements.txt        — Python dependencies
- environment.yml         — optional conda environment
- README.md               — this file

Installation
------------
1. Clone the repository:
   git clone https://github.com/mdzaheerjk/NYC-Airbnb-Room-Type-Predictor.git
   cd NYC-Airbnb-Room-Type-Predictor

2. Create and activate a virtual environment (recommended):
   python -m venv venv
   # macOS / Linux
   source venv/bin/activate
   # Windows (PowerShell)
   .\venv\Scripts\Activate.ps1

3. Install dependencies:
   pip install -r requirements.txt

If you prefer conda:
   conda env create -f environment.yml
   conda activate nyc-airbnb

Quickstart
----------
- Place the CSV dataset(s) under `data/` (e.g., `data/listings.csv`). Notebooks reference `data/` by default.
- Launch JupyterLab or Jupyter Notebook:
  jupyter lab
  Open and run the notebooks in the following order:
  1. notebooks/01-EDA.ipynb
  2. notebooks/02-feature-engineering.ipynb
  3. notebooks/03-modeling.ipynb
  4. notebooks/04-interpretation.ipynb

Run notebooks headlessly (optional):
  pip install nbconvert
  jupyter nbconvert --to notebook --execute notebooks/03-modeling.ipynb --ExecutePreprocessor.timeout=600 --output executed-03.ipynb

Notebooks & workflow
--------------------
1. 01-EDA.ipynb
   - Inspect missingness, distributions, geospatial plots (neighbourhood groups), price skew, and outlier detection.
2. 02-feature-engineering.ipynb
   - Clean text fields, parse amenities, encode categorical features (One-Hot / Target Encoding), scale numeric features, create interaction features (e.g., price per person), and engineer location-based features.
3. 03-modeling.ipynb
   - Baseline models (Logistic Regression), tree-based models (Random Forest, XGBoost/LightGBM), hyperparameter tuning (GridSearch / Randomized CV), cross-validation, and pipeline serialization.
4. 04-interpretation.ipynb
   - Confusion matrices, classification reports, SHAP or permutation importance, and error-case analysis.

Modeling approach
-----------------
- Problem type: multiclass classification (room_type)
- Typical pipeline:
  - Train/Validation split stratified by `room_type`
  - Preprocessing: imputation, encoding, scaling
  - Model candidates: LogisticRegression / RandomForestClassifier / XGBoost / LightGBM
  - Evaluation: accuracy, precision/recall/F1 per class, macro-F1, confusion matrix
  - Calibration & thresholding if needed for downstream use
- Export model artifacts using joblib or pickle for inference.

Reproducibility
---------------
- Set random seed in notebooks: e.g., SEED = 42
- Use the provided requirements.txt or environment.yml to reproduce package versions.
- Notebooks include explicit data preprocessing steps so you can re-generate training data and retrain models identically.

Evaluation & results
--------------------
- The notebooks compute and visualize:
  - Classification report (precision, recall, F1 by class)
  - Macro and weighted F1 scores for model comparison
  - Confusion matrix and normalized confusion matrix
  - Feature importances and SHAP summaries
- Exact metrics depend on dataset version and preprocessing choices — rerun `03-modeling.ipynb` to produce up-to-date numbers for your dataset.

How to use the model / inference
-------------------------------
- Option A: Run the inference cells in `03-modeling.ipynb` to load the serialized model and run predictions on a holdout/test CSV.
- Option B: Integrate the model artifact (e.g., `models/model.joblib`) into a microservice:
  - Load model: joblib.load("models/model.joblib")
  - Preprocess new rows using the same feature pipeline (serialized as `pipeline.joblib`)
  - Predict: model.predict(X_new) or model.predict_proba(X_new)

Tips & troubleshooting
----------------------
- If notebooks fail on missing columns: ensure your `data/listings.csv` has the expected column names or update the first cells that map/rename fields.
- Long notebook execution time: use a subset for fast iteration (`SAMPLE = 10000`) while developing.
- Memory issues with full dataset: use chunked processing or a smaller feature set during prototyping.
- If you get different results across runs, ensure the seed is set and that cross-validation splits are deterministic (use scikit-learn's random_state).

Future work
-----------
- Build a lightweight Flask/FastAPI endpoint for inference.
- Add geospatial models or spatial cross-validation to better capture neighborhood effects.
- Incorporate text embeddings from listing descriptions for improved accuracy.
- Experiment with class imbalanced techniques if any room type is rare.

Contributing
------------
Contributions are welcome! Suggested workflow:
1. Fork the repo.
2. Create a feature branch: git checkout -b feat/your-feature
3. Make changes and add tests (if applicable).
4. Open a pull request describing your change.

Please follow these guidelines:
- Keep notebooks tidy: clear outputs or use an executed output file for documentation.
- Add docstrings and comments where you add scripts or functions.
- If you add a new dependency, update requirements.txt and document why it is needed.

Acknowledgements & Data License
-------------------------------
- Dataset: NYC Airbnb datasets (ensure you comply with the dataset's license and terms of use).
- Example resources and model ideas inspired by common Kaggle kernels and public tutorials.
- If redistributing or using the dataset, cite the original data provider and follow their terms.

Contact
-------
Created by mdzaheerjk — for questions, issues, or collaboration ideas open an issue or contact via GitHub: https://github.com/mdzaheerjk

License
-------
If you want to add a license, consider MIT for open collaboration. Add a LICENSE file to the repository.
