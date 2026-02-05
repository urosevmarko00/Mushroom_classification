# Mushroom Classification Machine Learning Project

A comprehensive machine learning project for classifying mushrooms as edible or poisonous using multiple classification algorithms with extensive data preprocessing, hyperparameter tuning, and performance evaluation.

## Table of Contents

- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Dataset Information](#dataset-information)
- [Installation Prerequisites](#installation-prerequisites)
- [Setup Instructions](#setup-instructions)
- [Usage Examples](#usage-examples)
- [API Documentation](#api-documentation)
- [Project Structure](#project-structure)
- [Model Performance](#model-performance)
- [Contribution Guidelines](#contribution-guidelines)
- [License Information](#license-information)

---

## Project Overview

This project implements a comparative analysis of various machine learning classification algorithms for mushroom edibility prediction. The system uses the Secondary Mushroom Dataset containing 61,069 hypothetical mushroom samples with 20 distinct features. The project demonstrates the complete machine learning pipeline including data preprocessing, feature engineering, model training, hyperparameter optimization, and comprehensive performance evaluation.

### Target Variable

| Class | Label | Description |
|-------|-------|-------------|
| `e` | Edible | Safe for consumption |
| `p` | Poisonous | Harmful, not recommended for consumption |

---

## Key Features

### Data Preprocessing
- **Missing Value Imputation**: KNN-based imputation for categorical features
- **Mode Imputation**: Probabilistic mode-based filling for remaining missing values
- **Outlier Removal**: Filtering of extreme stem-width values (>60mm)
- **Feature Encoding**: Label encoding for tree-based models, one-hot encoding for linear models
- **Feature Scaling**: StandardScaler normalization for linear models
- **Feature Selection**: SelectFromModel for identifying important features

### Machine Learning Models
| Model | Type | Hyperparameters Tuned |
|-------|------|----------------------|
| Decision Tree | Tree-based | max_depth, min_samples_split, min_samples_leaf |
| Random Forest | Ensemble | n_estimators, max_depth, min_samples_split, min_samples_leaf |
| Support Vector Classifier (SVC) | Linear | C, gamma |
| Logistic Regression | Linear | C, penalty, solver |
| K-Nearest Neighbors (KNN) | Instance-based | n_neighbors, weights, metric |

### Evaluation Metrics
- **Accuracy**: Overall classification correctness
- **Precision**: True positive rate among predicted positives
- **Recall**: True positive rate among actual positives
- **F1 Score**: Harmonic mean of precision and recall
- **F2 Score**: Weighted harmonic mean favoring recall (β=2)
- **ROC-AUC**: Area under the Receiver Operating Characteristic curve

### Visualization
- Boxplot comparisons of model performance
- Stripplot overlays for cross-validation score distribution
- Confusion matrices for each model
- ROC curves with AUC scores
- Histograms for numerical feature distributions

---

## Dataset Information

### Secondary Mushroom Dataset

| Attribute | Value |
|-----------|-------|
| **Total Samples** | 61,069 mushrooms |
| **Number of Features** | 20 variables |
| **Feature Types** | 17 nominal, 3 metrical |
| **Classes** | 2 (edible, poisonous) |
| **Source** | Dennis Wagner (2020) |
| **Reference Book** | "Mushrooms & Toadstools" by Patrick Hardin (1999) |

### Feature Description

| # | Feature | Type | Values |
|---|---------|------|--------|
| 1 | cap-diameter | Metrical | Float (cm) |
| 2 | cap-shape | Nominal | bell(b), conical(c), convex(x), flat(f), sunken(s), spherical(p), others(o) |
| 3 | cap-surface | Nominal | fibrous(i), grooves(g), scaly(y), smooth(s), shiny(h), leathery(l), silky(k), sticky(t), wrinkled(w), fleshy(e) |
| 4 | cap-color | Nominal | brown(n), buff(b), gray(g), green(r), pink(p), purple(u), red(e), white(w), yellow(y), blue(l), orange(o), black(k) |
| 5 | does-bruise-bleed | Nominal | bruises-or-bleeding(t), no(f) |
| 6 | gill-attachment | Nominal | adnate(a), adnexed(x), decurrent(d), free(e), sinuate(s), pores(p), none(f), unknown(?) |
| 7 | gill-spacing | Nominal | close(c), distant(d), none(f) |
| 8 | gill-color | Nominal | (same as cap-color) + none(f) |
| 9 | stem-height | Metrical | Float (cm) |
| 10 | stem-width | Metrical | Float (mm) |
| 11 | stem-root | Nominal | bulbous(b), swollen(s), club(c), cup(u), equal(e), rhizomorphs(z), rooted(r) |
| 12 | stem-surface | Nominal | (same as cap-surface) + none(f) |
| 13 | stem-color | Nominal | (same as cap-color) + none(f) |
| 14 | veil-type | Nominal | partial(p), universal(u) |
| 15 | veil-color | Nominal | (same as cap-color) + none(f) |
| 16 | has-ring | Nominal | ring(t), none(f) |
| 17 | ring-type | Nominal | cobwebby(c), evanescent(e), flaring(r), grooved(g), large(l), pendant(p), sheathing(s), zone(z), scaly(y), movable(m), none(f), unknown(?) |
| 18 | spore-print-color | Nominal | (same as cap-color) |
| 19 | habitat | Nominal | grasses(g), leaves(l), meadows(m), paths(p), heaths(h), urban(u), waste(w), woods(d) |
| 20 | season | Nominal | spring(s), summer(u), autumn(a), winter(w) |

---

## Installation Prerequisites

### System Requirements
- **Operating System**: Windows, macOS, or Linux
- **Python Version**: Python 3.7 or higher
- **RAM**: Minimum 4GB (8GB recommended)
- **Disk Space**: ~500MB for project and dependencies

### Required Python Packages

| Package | Minimum Version | Purpose |
|---------|----------------|---------|
| numpy | 1.19.0 | Numerical computing |
| pandas | 1.1.0 | Data manipulation and analysis |
| matplotlib | 3.3.0 | Plotting and visualization |
| seaborn | 0.11.0 | Statistical data visualization |
| scikit-learn | 0.24.0 | Machine learning algorithms |

---

## Setup Instructions

### Step 1: Clone or Download the Project

```bash
# If using Git
git clone <repository-url>
cd pati/1

# Or download and extract the project files
```

### Step 2: Create a Virtual Environment (Recommended)

```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

Create a `requirements.txt` file in the project root:

```txt
numpy>=1.19.0
pandas>=1.1.0
matplotlib>=3.3.0
seaborn>=0.11.0
scikit-learn>=0.24.0
```

Then install the packages:

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python -c "import numpy, pandas, matplotlib, seaborn, sklearn; print('All dependencies installed successfully!')"
```

### Step 5: Verify Dataset Location

Ensure the dataset files are in the correct location:

```
pati/1/
├── main.py
├── MushroomDataset/
│   ├── secondary_data.csv
│   ├── secondary_data_meta.txt
│   ├── primary_data.csv
│   └── primary_data_meta.txt
```

---

## Usage Examples

### Running the Main Script

Execute the complete analysis pipeline:

```bash
python main.py
```

This will:
1. Load the secondary mushroom dataset
2. Preprocess the data (impute missing values, remove outliers)
3. Split data into training and test sets
4. Train and evaluate 5 different classification models
5. Perform hyperparameter tuning for each model
6. Generate performance visualizations

### Using Individual Functions

#### 1. KNN Imputation

```python
from main import knn_impute
import pandas as pd

# Load your dataset
dataset = pd.read_csv("MushroomDataset/secondary_data.csv", sep=';')

# Impute missing values for a specific feature
target = "ring-type"
predictors = ["has-ring", "stem-height", "cap-shape", "gill-attachment"]
imputed_values = knn_impute(dataset, target, predictors)
dataset[target] = imputed_values
```

#### 2. Model Evaluation

```python
from main import evaluate_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Prepare your data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# Create and evaluate a model
model = RandomForestClassifier(max_depth=10, min_samples_leaf=5, random_state=42)
evaluate_model(model, X_train, X_test, y_train, y_test)
```

#### 3. Boxplot Comparison

```python
from main import boxplot_compare

# Prepare model scores from cross-validation
models_accuracy = {
    "Decision Tree": [0.95, 0.96, 0.94, 0.95, 0.96],
    "Random Forest": [0.98, 0.97, 0.98, 0.99, 0.98],
    "SVC": [0.92, 0.93, 0.91, 0.92, 0.93]
}

# Generate comparison plot
boxplot_compare(models_accuracy, "Accuracy")
```

### Custom Model Training Example

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, fbeta_score

# Load and prepare data
dataset = pd.read_csv("MushroomDataset/secondary_data.csv", sep=';')

# Preprocessing steps...
# (Remove outliers, impute missing values, encode features)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# Define hyperparameter grid
hyperparams = {
    'n_estimators': [200, 500],
    'max_depth': [None, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Perform grid search
grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    hyperparams,
    cv=5,
    scoring='f2',
    refit=True
)
grid.fit(X_train, y_train)

# Best model
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

print(f"Best parameters: {grid.best_params_}")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"F2 Score: {fbeta_score(y_test, y_pred, beta=2):.4f}")
```

---

## API Documentation

### Functions

#### `knn_impute(data_set, target, predictor)`

Performs K-Nearest Neighbors imputation for missing categorical values.

**Parameters:**
- `data_set` (pd.DataFrame): Original dataset with missing values
- `target` (str): Column name to impute
- `predictor` (list of str): List of predictor column names

**Returns:**
- `np.ndarray`: Imputed values in original categories

**Example:**
```python
imputed = knn_impute(dataset, "ring-type", ["has-ring", "stem-height"])
```

#### `evaluate_model(model, X_train, X_test, y_train, y_test)`

Trains and evaluates a classification model, printing metrics and displaying confusion matrix.

**Parameters:**
- `model`: Scikit-learn estimator object
- `X_train` (pd.DataFrame or np.ndarray): Training features
- `X_test` (pd.DataFrame or np.ndarray): Test features
- `y_train` (pd.Series or np.ndarray): Training labels
- `y_test` (pd.Series or np.ndarray): Test labels

**Returns:**
- `None`: Prints metrics and displays plot

**Example:**
```python
evaluate_model(RandomForestClassifier(), X_train, X_test, y_train, y_test)
```

#### `boxplot_compare(models_scores, metric)`

Creates boxplot and stripplot visualization for comparing model performances.

**Parameters:**
- `models_scores` (dict): Dictionary mapping model names to score lists
- `metric` (str): Name of the metric being compared

**Returns:**
- `None`: Displays visualization

**Example:**
```python
boxplot_compare({"Model A": [0.9, 0.91, 0.89], "Model B": [0.85, 0.86, 0.84]}, "Accuracy")
```

### Data Loading

```python
import pandas as pd

# Load secondary mushroom dataset
dataset = pd.read_csv("MushroomDataset/secondary_data.csv", sep=';')

# Load primary mushroom dataset (if needed)
primary_dataset = pd.read_csv("MushroomDataset/primary_data.csv", sep=';')
```

---

## Project Structure

```
pati/1/
│
├── main.py                          # Main analysis script with ML pipeline
├── README.md                        # Project documentation (this file)
│
├── MushroomDataset/                 # Dataset directory
│   ├── secondary_data.csv          # Secondary mushroom dataset (61,069 samples)
│   ├── secondary_data_meta.txt     # Secondary dataset metadata
│   ├── primary_data.csv            # Primary mushroom dataset (173 species)
│   └── primary_data_meta.txt       # Primary dataset metadata
│
└── .idea/                          # IntelliJ IDEA project configuration
    ├── secondaryMushroom.iml       # Module configuration
    ├── misc.xml
    ├── modules.xml
    └── inspectionProfiles/
```

### File Descriptions

| File | Description |
|------|-------------|
| [`main.py`](main.py:1) | Complete ML pipeline including data preprocessing, model training, hyperparameter tuning, and visualization |
| `MushroomDataset/secondary_data.csv` | Main dataset with 61,069 mushroom samples |
| `MushroomDataset/secondary_data_meta.txt` | Metadata describing secondary dataset features |
| `MushroomDataset/primary_data.csv` | Primary dataset with 173 mushroom species |
| `MushroomDataset/primary_data_meta.txt` | Metadata describing primary dataset features |

---

## Model Performance

The project evaluates five classification models using 5-fold stratified cross-validation. Key performance characteristics:

### Model Comparison Summary

| Model | Strengths | Best Use Case |
|-------|-----------|---------------|
| Random Forest | High accuracy, robust to overfitting, feature importance | General-purpose classification |
| Decision Tree | Interpretable, fast training | Quick prototyping, rule extraction |
| SVC | Good for high-dimensional spaces | Complex decision boundaries |
| Logistic Regression | Simple, interpretable, probabilistic | Baseline comparison, feature importance |
| KNN | No training required, intuitive | Small datasets, local patterns |

### Evaluation Metrics Used

- **Cross-Validation**: 5-fold stratified for robust performance estimation
- **Primary Metric**: F2 Score (emphasizes recall for safety-critical mushroom classification)
- **Secondary Metrics**: Accuracy, Precision, Recall, F1, ROC-AUC

---

## Contribution Guidelines

We welcome contributions to improve this project! Please follow these guidelines:

### Reporting Issues

When reporting bugs or issues, please include:
- Python version and operating system
- Package versions (use `pip list`)
- Complete error traceback
- Steps to reproduce the issue

### Pull Request Process

1. **Fork the repository** and create a feature branch
2. **Make your changes** following the existing code style
3. **Add tests** for new functionality (if applicable)
4. **Update documentation** for any API changes
5. **Submit a pull request** with a clear description of changes

### Code Style

- Follow PEP 8 style guidelines
- Use meaningful variable and function names
- Add docstrings to functions (following existing format)
- Keep functions focused and modular

### Suggested Improvements

Potential areas for contribution:
- Additional classification algorithms (XGBoost, LightGBM, Neural Networks)
- Advanced feature engineering techniques
- Ensemble methods and stacking
- Interactive visualizations with Plotly or Dash
- Model deployment as a web service
- Additional dataset support

---

## License Information

### Dataset License

The mushroom datasets are provided by Dennis Wagner (2020) and are based on:

> **Source Book**: Patrick Hardin. *Mushrooms & Toadstools*. Zondervan, 1999

The datasets are available at: https://mushroom.mathematik.uni-marburg.de/files/

### Code License

This project is provided as-is for educational and research purposes. Please ensure compliance with the original dataset licensing terms when using or redistributing the data.

### Third-Party Libraries

This project uses the following open-source libraries:

| Library | License |
|---------|---------|
| NumPy | BSD License |
| Pandas | BSD License |
| Matplotlib | PSF License |
| Seaborn | BSD License |
| Scikit-learn | BSD License |

Please refer to each library's documentation for specific licensing terms.

---

## Acknowledgments

- **Dennis Wagner** for creating and publishing the mushroom datasets
- **Patrick Hardin** for the original "Mushrooms & Toadstools" reference book
- **Jeff Schlimmer** for the original UCI Mushroom Dataset (1987)
- The scikit-learn community for providing excellent machine learning tools

---

## References

1. Wagner, D. (2020). *Secondary Mushroom Dataset*. University of Marburg. https://mushroom.mathematik.uni-marburg.de/files/
2. Hardin, P. (1999). *Mushrooms & Toadstools*. Zondervan.
3. Schlimmer, J. (1987). *Mushroom Data Set*. UCI Machine Learning Repository. https://archive.ics.uci.edu/ml/datasets/Mushroom
4. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.

---

## Contact

For questions, issues, or suggestions regarding this project, please open an issue in the repository or contact the maintainers.

---

*Last Updated: February 2026*
