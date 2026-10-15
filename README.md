# ECG & Machine Learning Case Study

This project explores three core machine-learning tasks:

1. **ECG classification** using the MIT-BIH Arrhythmia Database.
2. **Hospital length-of-stay regression** using the New York State Hospital Inpatient Discharges (SPARCS De-Identified): 2010 dataset.
3. **Credit card customer segmentation clustering** using the Kaggle Credit Card dataset, implementing **K-Means Clustering** and **Agglomerative Hierarchical Clustering**.

The notebooks contain exploratory data analysis, preprocessing, feature engineering, and model-ready outputs across the three tracks.

## Project Structure

```text
.
├── data/
│   ├── classification/get_dataset.py   # Download MIT-BIH records
│   ├── regression/get_dataset.py       # Download the 2010 SPARCS CSV
│   └── clustering/get_dataset.py       # Download CC GENERAL.csv dataset
├── notebooks/
│   ├── classification_preprocessing.ipynb
│   ├── classification.ipynb
│   ├── regression_preprocessing.ipynb
│   ├── regression_fast.ipynb
│   └── clustering.ipynb                # K-Means & Agglomerative Clustering
├── preprocessed_dataset/                # ECG NumPy/features used by notebooks
├── regression_dataset/                  # Downloaded SPARCS CSV
├── clustering_dataset/                  # CC GENERAL.csv & credit_card_clusters.csv
├── requirements.txt
└── README.md
```

## Datasets

### MIT-BIH Arrhythmia Database

The MIT-BIH Arrhythmia Database contains annotated two-channel ambulatory ECG recordings. The classification workflow uses ECG signal windows and rhythm annotations to create model inputs and labels. The raw database is downloaded as WFDB record files; it is not committed to the repository.

Source: [PhysioNet MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/)

The repository also contains derived files in `preprocessed_dataset/`:

- `X_signals.npy`: preprocessed ECG signal windows
- `y_classification.npy`: classification labels
- `y_regression.npy`: regression labels used by the ECG workflow
- `ecg_features_dataset.csv`: extracted ECG features

### NY SPARCS Hospital Inpatient Discharges: 2010

This dataset contains de-identified inpatient discharge records from New York State. The source contains 2,622,133 rows and 38 columns, including demographics, diagnoses, procedures, admission information, payer information, length of stay, total charges, and total costs.

The regression notebook uses **Length of Stay** as the target. It is converted from text to a non-negative numeric value. Identifier and provider-license fields are excluded from the baseline model because they have high cardinality, many missing values, or administrative meaning. `Total Charges` and `Total Costs` are excluded to avoid using post-discharge information as leakage when predicting length of stay.

Source: [New York Open Data](https://data.ny.gov/)

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

The repository currently includes an `ecg` virtual environment directory on the original development machine, but a new environment is recommended for portability.

## Download the Datasets

Run these commands from the repository root after installing dependencies.

### Regression dataset

```bash
python data/regression/get_dataset.py
```

The script searches the New York Open Data catalog for the exact 2010 SPARCS dataset and saves it as:

```text
regression_dataset/hospital-inpatient-discharges-sparcs-de-identified-2010-1.csv
```

You can choose another output path:

```bash
python data/regression/get_dataset.py --output regression_dataset/sparcs_2010.csv
```

### Classification dataset

```bash
python data/classification/get_dataset.py
```

The script downloads raw MIT-BIH WFDB records to `data/mitdb/`. The directory is ignored by Git because the raw recordings are downloaded data. To choose another location:

```bash
python data/classification/get_dataset.py --output data/mitdb
```

### Clustering dataset

```bash
python data/clustering/get_dataset.py
```

The script downloads the Credit Card dataset (`CC GENERAL.csv`) to `data/clustering/CC GENERAL.csv` and `clustering_dataset/CC GENERAL.csv`.

## Notebook Workflow

Open the `notebooks/` directory in VS Code or Jupyter and run cells from top to bottom.

### Classification

`classification_preprocessing.ipynb` loads the ECG arrays, inspects signal and label distributions, and prepares the classification data for modeling.

`classification.ipynb` imports the preprocessed dataset from `classification_preprocessing.ipynb`, splits the data into training and testing sets, and trains five classification algorithms with strict hyperparameter regularization to prevent model overfitting:

1. **Logistic Regression** (Baseline classifier; L2 regularization tuning over $C$; odds ratio analysis)
2. **K-Nearest Neighbors (KNN)** (Hyperparameter tuning enforcing $k \ge 3$; distance metrics & feature scaling analysis)
3. **Gaussian Naive Bayes** (`var_smoothing` tuning; check of conditional independence assumption)
4. **Decision Tree Classifier** (Constraining `max_depth`, `min_samples_split`, and `min_samples_leaf`; tree visualization & feature importances)
5. **Support Vector Machine (SVC)** (Tuning regularization parameter $C$ and `kernel`; feature scaling analysis)

---

### Regression

`regression_preprocessing.ipynb` follows this sequence:

1. Load and audit shape, data types, and missing values.
2. Convert `Length of Stay` to a numeric target.
3. Check and remove exact duplicate rows and invalid target rows.
4. Explore numeric and categorical feature distributions.
5. Plot the target distribution, correlation heatmap, and feature-target relationships.
6. Add `Age Group Midpoint` as an engineered numeric feature while retaining the original age category.
7. Split the data into training and testing sets.
8. Impute missing values, one-hot encode categorical columns, and standardize numeric columns.

The encoder and scaler are fitted on the training set only. The transformed matrices are sparse to keep memory usage manageable.

---

### Clustering

`clustering.ipynb` performs comprehensive customer segmentation using credit card transaction behavioral data:

1. **Exploratory Data Analysis (EDA):** Audit missing values (`MINIMUM_PAYMENTS`, `CREDIT_LIMIT`), inspect skewed behavioral distributions, and analyze the feature correlation matrix.
2. **Preprocessing & Transformation:** Drop account identifiers (`CUST_ID`), impute null values using robust medians, apply logarithmic transformation ($y = \ln(1+x)$) to stabilize variance across heavily skewed monetary columns, and standardize with `StandardScaler`.
3. **Dimensionality Reduction (PCA):** Compute Scree plot and project standardized 17-dimensional behavioral space into 2D principal components for spatial visualization.
4. **Algorithm 1 — K-Means Clustering:**
   - Multi-metric hyperparameter sweep ($k \in [2, 8]$): Elbow curve (WCSS / Inertia), Silhouette score, Davies-Bouldin index, and Calinski-Harabasz score.
   - Train final model with $k=4$, project centroids into 2D PCA space, and profile behavioral personas (Inactive users, Cash Advance revolvers, VIP premium spenders, Active installment transactors).
5. **Algorithm 2 — Agglomerative Hierarchical Clustering:**
   - Construct hierarchical dendrogram using Ward's minimum variance linkage with Euclidean distance.
   - Fit `AgglomerativeClustering(n_clusters=4, linkage='ward')` and map cluster boundaries in PCA space.
6. **Performance & Complexity Comparison:**
   - Quantitative evaluation across Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Score, and Adjusted Rand Index (ARI).
   - Analysis of time/space complexity ($O(N \cdot K \cdot d \cdot i)$ vs $O(N^2 \log N)$), memory footprint, and scalability.
7. **Actionable Marketing & Credit Risk Recommendations:** Tailored strategies for credit limit reviews, reward program targeting, and loan consolidation.

## Regression Outputs

The preprocessing workflow produces:

- `X_train_encoded`: sparse encoded training features
- `X_test_encoded`: sparse encoded test features
- `y_train`: training length-of-stay values
- `y_test`: test length-of-stay values

The notebook can evaluate regression models using MAE, RMSE, and R². MAE is measured in days for this target. RMSE gives additional weight to unusually large errors, while R² measures explained variance relative to a mean-target baseline.

Because the notebooks are exploratory, exact model scores depend on the cells run, package versions, split seed, and model hyperparameters. Run the regression notebook to generate the current results table rather than treating undocumented scores as fixed project results.

## Reproducibility and Data Notes

- Use `random_state=42` for the documented train/test split.
- Do not fit imputers, encoders, or scalers on the test set.
- Keep `Length of Stay` out of the feature matrix because it is the target.
- Keep `Total Charges` and `Total Costs` out of an early length-of-stay prediction model because they may only be known after discharge.
- Do not commit raw downloaded datasets or generated large matrices unless explicitly required.

## Attribution

The project uses the MIT-BIH Arrhythmia Database provided through PhysioNet and the New York State SPARCS hospital discharge dataset provided through New York Open Data. Follow each source's terms of use and citation requirements when distributing results.

## AI Assistance Disclosure

In accordance with Section 7.5 of the 23CSE301 Capstone Guidelines, generative AI tools were utilized exclusively for code scaffolding, repetitive boilerplate assistance, and Markdown table formatting. All exploratory data analysis, data quality investigations, feature selection decisions (including the exclusion of Production), model interpretations, and conclusions were independently conducted and reasoned by the project team. **Chatgpt** has been used for the error rectification and boilerplate generation. **Perplexity** for the research part of finding the correct dataset and its usage.
