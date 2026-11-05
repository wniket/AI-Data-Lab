
# 🧠 AI DataLab

### End-to-End Machine Learning and Data Analysis Platform

AI DataLab is an interactive data science platform built with Python and Streamlit.

It allows users to upload datasets, perform exploratory data analysis, train machine learning models, compare their performance, and interpret model predictions using Explainable AI.

## ✨ Features

### 📊 Exploratory Data Analysis (EDA)
- CSV dataset upload
- Dataset overview and statistics
- Missing values detection
- Duplicate detection
- Correlation heatmap
- Feature distribution visualization

### 🤖 Machine Learning
- Linear Regression
- Random Forest Regressor
- Automated preprocessing pipeline
- Missing value imputation
- Numerical feature scaling
- Categorical feature encoding

### 📈 Model Evaluation
- Train/Test Split
- 5-Fold Cross-Validation
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score
- Generalization Gap analysis

### ⚙️ Hyperparameter Optimization
- GridSearchCV
- Automated Random Forest parameter tuning
- Cross-validation based model selection

### 🔍 Explainable AI
- Feature Importance
- SHAP Global Explanation
- SHAP Individual Prediction Explanation
- Prediction vs Actual visualization

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data manipulation |
| NumPy | Numerical computing |
| Scikit-learn | Machine Learning |
| Streamlit | Interactive web interface |
| Plotly | Interactive visualizations |
| SHAP | Model explainability |
| Matplotlib | Visualization |

## 📁 Project Structure

```text
AIDataLab/
│
├── app.py
├── main.py
├── requirements.txt
├── .gitignore
│
├── .streamlit/
│   └── config.toml
│
└── src/
    ├── __init__.py
    ├── analysis.py
    ├── ml_engine.py
    └── explainability.py
```

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/wniket/AI-Data-Lab.git
cd AI-Data-Lab
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
```

### 3. Activate the environment

macOS / Linux:

```bash
source .venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the application

```bash
streamlit run app.py
```

The application will open in your browser.

## 📂 Dataset

The application supports CSV datasets with a numerical target variable for regression tasks.

The project was developed and tested using the California Housing dataset.

Dataset files are not included in this repository.

## 🧪 Machine Learning Workflow

1. Upload a dataset.
2. Explore and analyze the data.
3. Select a target variable.
4. Preprocess numerical and categorical features.
5. Train machine learning models.
6. Evaluate model performance.
7. Optimize hyperparameters.
8. Interpret predictions using SHAP.

## 🎯 Project Goals

This project was developed as part of my journey toward becoming an AI / Machine Learning Engineer.

Its main goals are to explore:
- Data Science workflows
- Machine Learning algorithms
- Model evaluation and optimization
- Explainable Artificial Intelligence
- Python software development

---

**Author:** wniket

**GitHub:** https://github.com/wniket
