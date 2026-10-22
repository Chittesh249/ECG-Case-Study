# ECG Case Study

This project explores two healthcare machine-learning tasks:

1. **ECG classification** using the MIT-BIH Arrhythmia Database.
2. **Hospital length-of-stay regression** using the New York State Hospital Inpatient Discharges (SPARCS De-Identified): 2010 dataset.

The notebooks contain exploratory data analysis, preprocessing, feature engineering, and model-ready outputs for the two tracks.

## Project Structure

```text
.
├── data/
│   ├── classification/get_dataset.py   # Download MIT-BIH records
│   └── regression/get_dataset.py       # Download the 2010 SPARCS CSV
├── notebooks/
│   ├── classification_preprocessing.ipynb
│   ├── classification.ipynb
│   ├── regression_preprocessing.ipynb
│   └── regression_fast.ipynb
├── preprocessed_dataset/                # ECG NumPy/features used by notebooks
├── regression_dataset/                  # Downloaded SPARCS CSV
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

# Credit Card Customer Segmentation: K-Means vs. Agglomerative Clustering

Unsupervised segmentation of credit card holders using **K-Means** and **Agglomerative Hierarchical Clustering (Ward linkage)**, with a quantitative comparison of the two.

**Dataset:** [Credit Card Dataset for Clustering (Kaggle, Arjun Bhasin)](https://www.kaggle.com/datasets/arjunbhasin2013/ccdata)
8,950 customers, 18 behavioral attributes over a 6-month window (balance, purchases, cash advances, payment behavior, credit limit, tenure).

## Objective

Group customers by transactional and payment behavior to support:
- Targeted marketing (cashback, installment offers, balance transfers)
- Credit risk flagging (e.g., heavy cash advance use with low payments)
- Product design based on empirical customer personas

## Pipeline

1. **Data audit and EDA**: dtypes, missing values, summary statistics, distributions, correlation heatmap
2. **Preprocessing**
   - Drop `CUST_ID`
   - Median imputation for `MINIMUM_PAYMENTS` (313 missing) and `CREDIT_LIMIT` (1 missing)
   - `log1p` transform on 10 right-skewed financial features
   - `StandardScaler` (zero mean, unit variance), giving a 17-feature matrix
3. **PCA**: explained-variance analysis and a 2D projection for visualization (PC1 + PC2 retain ~55.8% of variance)
4. **K-Means**: k searched from 2 to 8 using inertia (elbow), silhouette, Davies-Bouldin and Calinski-Harabasz; final model with **k = 4**
5. **Agglomerative clustering**: Ward linkage, dendrogram on a 1,500-point sample, final model fit on the full data with **k = 4**
6. **Comparison**: internal validity metrics, Adjusted Rand Index, and a contingency heatmap between the two label sets

## K-Means Personas (k = 4)

| Cluster | Persona | Behavior |
|---|---|---|
| 0 | Conservative / low-activity | Low balance, low purchases, infrequent use |
| 1 | Cash-advance / revolving borrowers | High balance and cash advances, very low purchases |
| 2 | High-volume spenders | Highest balance, purchases, credit limit and payments |
| 3 | Active installment purchasers | Frequent, installment-heavy purchases on moderate balances |

## Results

| Metric | K-Means (k=4) | Agglomerative (k=4) |
|---|---|---|
| Silhouette (↑) | 0.2174 | 0.1673 |
| Davies-Bouldin (↓) | 1.6876 | 1.7326 |
| Calinski-Harabasz (↑) | 2260.4 | 1870.1 |

- **Adjusted Rand Index between the two methods: 0.3497**, so the algorithms agree only partially.
- K-Means scores better on all three internal metrics and scales far better (roughly linear vs. O(n² log n) time and O(n²) memory).
- Absolute silhouette scores are low (~0.22 or less), which suggests the segments overlap and form a continuum rather than sharply separated groups. Treat the personas as useful behavioral summaries, not hard boundaries.

## Repository Structure

```
.
├── clustering.ipynb
└── data/
    └── clustering/
        ├── CC GENERAL.csv
        └── get_dataset.py
```

The notebook searches several relative paths for `CC GENERAL.csv`, including `../data/clustering/` and `data/clustering/`.

## Setup

Python 3.10 was used.

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy jupyter
```

1. Download the dataset (for example, with `data/clustering/get_dataset.py`) so that `CC GENERAL.csv` is in one of the paths above.
2. Launch Jupyter and run `clustering.ipynb` top to bottom:

```bash
jupyter notebook clustering.ipynb
```

## Notes and Limitations

- Silhouette scores during the k search use `sample_size=3000`, and the final comparison uses `sample_size=4000` (`random_state=42`), so values are sampled estimates.
- The dendrogram uses a random 1,500-sample subset, while the final Agglomerative model is fit on all 8,950 rows.
- Choosing k = 4 is a judgment call. Silhouette is highest at k = 2, and the elbow is not sharp.

## Possible Extensions

- Try Gaussian Mixture Models, DBSCAN/HDBSCAN, or K-Medoids
- Cluster on PCA-reduced features
- Add stability analysis across random seeds and bootstrap samples
