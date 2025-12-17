# Retail Marketing Analytics: Income Classification & Customer Segmentation

## Project Overview

This project simulates a real-world data science project for a **retail client** seeking actionable marketing insights from population data. It demonstrates end-to-end machine learning workflows on **U.S. Census Bureau Current Population Survey (1994–1995)** data.

### Key Deliverables

1. **Income Classification Model** – Predicts whether an individual earns **>$50K or ≤$50K annually** using 40 demographic and employment-related features
2. **Customer Segmentation Model** – Performs **unsupervised clustering** to identify distinct socio-demographic segments for targeted marketing campaigns


---

## Repository Structure

```
.
├── classification.ipynb           # Income classification analysis & models
├── segmentation.ipynb             # Customer segmentation clustering
├── census-bureau.data             # Raw census data (CSV format)
├── census-bureau.columns          # Column names for the dataset
├── ML-TakehomeProject.pdf         # Original project specification
├── Project Report.pdf             # Final analysis report
├── README.md                      # This file
└── requirements.txt               # Python dependencies
```

---

## Model Performance

### Classification Results (Test Set)

All models optimized for **F1 score** on the minority class (income >$50K):

| Model | Best F1 Threshold | Precision (minority class)| Recall (minority class)| F1 Score (minority class)| Accuracy | ROC AUC |
|-------|-------------------|-----------|--------|----------|----------|---------|
| **Logistic Regression** | 0.224 | 0.49 | 0.63 | 0.55 | 0.93 | 0.937 |
| **XGBoost** | 0.840 | 0.56 | 0.63 | 0.59 | 0.94 | 0.949 |
| **CatBoost** ⭐ | 0.298 | 0.60 | 0.62 | **0.61** | 0.95 | **0.952** |

**Best Model:** CatBoost achieves the highest F1 score (0.61) and ROC AUC (0.952), making it the recommended production model.


---

## Getting Started


### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. **Create virtual environment**
   ```bash
   python -m venv env
   source env/bin/activate       # macOS/Linux
   env\Scripts\activate          # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

##  Usage

### Running the Notebooks

1. **Launch Jupyter**
   ```bash
   jupyter notebook
   ```

2. **Execute analyses:**
   - `classification.ipynb` – Income prediction models
   - `segmentation.ipynb` – Customer clustering

3. **Run all cells:** Use `Cell → Run All` to reproduce full results

### Data Loading

Ensure both data files are in the project root:
```python
# Column names
columns_file = 'census-bureau.columns'
data_file = 'census-bureau.data'
columns = pd.read_csv(columns_file, header=None).iloc[:, 0].tolist()

# Load dataset
df = pd.read_csv(data_file, delimiter=',', header=None, names=columns)
```

---
## Dataset Information

**Source:** U.S. Census Bureau Current Population Survey (1994-1995)  
**Size:** ~200K records  
**Features:** 40 demographic, employment, and geographic variables  
**Target:** Binary income classification (≤$50K vs >$50K)

---

## Documentation

- `Project Report.pdf` – Comprehensive analysis findings and recommendations
- Notebook outputs include inline visualizations and detailed commentary
