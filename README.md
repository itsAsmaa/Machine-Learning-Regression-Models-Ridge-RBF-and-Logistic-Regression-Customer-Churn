# Machine Learning Regression Models: Ridge, RBF, and Logistic Regression

A comprehensive implementation of non-linear regression techniques and logistic regression for classification, demonstrating regularization methods, basis function expansion, and polynomial feature engineering for machine learning applications.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Assignment Structure](#assignment-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Part 1: Non-Linear Regression](#part-1-non-linear-regression)
- [Part 2: Logistic Regression](#part-2-logistic-regression)
- [Results](#results)
- [Key Findings](#key-findings)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project implements and compares various regression and classification techniques for machine learning:

**Part 1: Non-Linear Regression**
- Ridge Regression with multiple regularization parameters
- Radial Basis Function (RBF) regression with varying numbers of basis functions
- Analysis of overfitting, underfitting, and generalization performance

**Part 2: Logistic Regression for Classification**
- Binary classification on customer churn dataset
- Linear and polynomial feature transformations
- Model evaluation using accuracy, precision, recall, and ROC-AUC metrics

**Course**: Machine Learning and Data Science (ENCS5341)  
**Institution**: Electrical and Computer Engineering Department  
**Assignment**: Non-Linear Regression & Logistic Regression

## Features

### Non-Linear Regression Analysis
- Synthetic dataset generation with controlled noise
- Ridge regression with regularization (λ = 0, 0.05, 0.2, 2, 15)
- RBF basis function implementation (1, 5, 10, 50 basis functions)
- Visualization of model complexity vs generalization trade-offs
- Comparison of regularization techniques

### Logistic Regression for Classification
- Reusable data preprocessing pipeline from Assignment 1
- Train/validation/test split (2500/500/500 samples)
- Linear decision boundary model
- Polynomial feature expansion (degree 2, 5, 9)
- Comprehensive model evaluation metrics
- ROC curve analysis with AUC computation

### Evaluation & Visualization
- Performance metrics: Accuracy, Precision, Recall, AUC
- Model comparison across different complexities
- Overfitting and underfitting analysis
- Best model selection based on validation performance

## Assignment Structure

### Part 1: Non-Linear Regression

#### A) Ridge Regression
Generate synthetic dataset with 25 points where:
- x values uniformly distributed in [0, 1]
- y = sin(5πx) + ε, where ε ∈ [-0.3, 0.3]

Apply ridge regression with different λ values and analyze generalization.

#### B) RBF Regression
Use Radial Basis Functions with evenly spaced centers:
- 1 RBF basis function
- 5 RBF basis functions
- 10 RBF basis functions
- 50 RBF basis functions

Compare results with true function sin(5πx).

### Part 2: Logistic Regression

Apply logistic regression to customer churn prediction:
- Preprocess data (standardization, missing values, outliers)
- Split dataset: 2500 train / 500 validation / 500 test
- Train models with linear and polynomial features
- Evaluate and compare model performance
- Generate ROC curves for best model

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ml-regression-models.git
cd ml-regression-models
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

### Required Libraries
```
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
jupyter>=1.0.0
```

## Usage

### Running the Complete Assignment

1. **Execute the main script**:
```bash
python assignment2_encs5341.py
```

2. **Or use Jupyter Notebook** (recommended):
```bash
jupyter notebook Assignment2_ENCS5341.ipynb
```

### Running Individual Parts

#### Part 1: Non-Linear Regression
```python
# Ridge Regression
python ridge_regression.py

# RBF Regression
python rbf_regression.py
```

#### Part 2: Logistic Regression
```python
# Logistic Regression with preprocessing
python logistic_regression_churn.py
```

### Input Data

For Part 2, you need the customer churn dataset:
- Place `customer_data.csv` in the project root directory
- The dataset should contain: CustomerID, Age, Gender, Income, Tenure, ProductType, SupportCalls, ChurnStatus

## Part 1: Non-Linear Regression

### A) Ridge Regression

**Objective**: Analyze the effect of regularization parameter λ on model generalization.

**Implementation**:
- Generate 25 data points with noise
- Apply polynomial features (degree 10)
- Train Ridge Regression with λ ∈ {0, 0.05, 0.2, 2, 15}
- Visualize fitted curves

**Key Insights**:
- λ = 0 or 0.05: Overfitting (follows noise)
- λ = 0.2 or 2: Best generalization (smooth, captures pattern)
- λ = 15: Underfitting (too flat, misses sine pattern)

### B) RBF Regression

**Objective**: Compare different numbers of RBF basis functions for function approximation.

**Implementation**:
- Create Gaussian RBF basis functions
- Centers evenly spaced across [0, 1]
- Width σ = 0.15 (chosen based on center spacing)
- Train with 1, 5, 10, and 50 RBFs

**Key Insights**:
- 1 RBF: Underfitting (too simple)
- 5 RBFs: Good balance (captures pattern, avoids noise)
- 10 RBFs: Moderate fit (more detail, slight overfitting risk)
- 50 RBFs: Overfitting (fits noise, poor generalization)

**Best Configuration**: 5 RBF basis functions provide optimal generalization.

## Part 2: Logistic Regression

### Data Preprocessing

Reused preprocessing pipeline from Assignment 1:
- Missing value handling (deletion for <5%, median imputation for 5-30%)
- Outlier detection using IQR method
- Outlier treatment via winsorizing
- Feature standardization (Z-score normalization)

**Preprocessing Results**:
- Original: 3,500 samples
- After preprocessing: 3,165 samples (90.4% retained)
- 100% complete data (no missing values)

### Model Training & Evaluation

**Dataset Split**:
- Training: 2,500 samples
- Validation: 500 samples
- Test: 500 samples

**Models Trained**:

1. **Linear Decision Boundary**
   - Standard logistic regression
   - Original features only

2. **Polynomial Features (Degree 2)**
   - 27 features after transformation
   - Captures quadratic interactions

3. **Polynomial Features (Degree 5)**
   - 461 features after transformation
   - More complex decision boundary

4. **Polynomial Features (Degree 9)**
   - 5,004 features after transformation
   - Very high complexity

### Performance Metrics

| Model | Validation Accuracy | Test Accuracy |
|-------|---------------------|---------------|
| Linear | 0.9700 | 0.9640 |
| Polynomial (Degree 2) | 0.9700 | 0.9640 |
| Polynomial (Degree 5) | 0.9840 | 0.9760 |
| Polynomial (Degree 9) | 0.9540 | 0.9540 |

**Best Model**: Polynomial Degree 5 (Validation: 98.40%, Test: 97.60%)

## Results

### Part 1: Non-Linear Regression

**Ridge Regression Findings**:
- Optimal λ value: 0.2 - 2.0 (balances bias-variance trade-off)
- Too small λ (0, 0.05): Model overfits to training noise
- Too large λ (15): Model underfits, loses important patterns
- Regularization is crucial for polynomial regression

**RBF Regression Findings**:
- Optimal number of RBFs: 5 basis functions
- 1 RBF: Insufficient complexity (high bias)
- 5-10 RBFs: Good generalization
- 50 RBFs: Excessive complexity (high variance)
- RBF provides localized control over function approximation

### Part 2: Logistic Regression

**Key Findings**:

1. **Model Complexity vs Performance**:
   - Degree 2: Underfitting (too simple, lower accuracy)
   - Degree 5: Optimal balance (best validation and test accuracy)
   - Degree 9: Overfitting (high complexity, reduced test accuracy)

2. **Generalization Analysis**:
   - Polynomial degree 5 achieves best generalization
   - Degree 9 shows signs of overfitting (5,004 features vs 2,500 samples)
   - Higher degree ≠ better performance

3. **Feature Dimensionality Impact**:
   - Original features: 6 dimensions
   - Degree 2: 27 dimensions (4.5x increase)
   - Degree 5: 461 dimensions (77x increase)
   - Degree 9: 5,004 dimensions (834x increase)

4. **ROC-AUC Analysis**:
   - Best model (Degree 5) selected for ROC curve generation
   - High AUC indicates excellent discrimination ability
   - Model effectively separates churned vs non-churned customers

## Key Findings

### Bias-Variance Trade-off

**Ridge Regression**:
- Small λ: Low bias, high variance (overfitting)
- Optimal λ: Balanced bias and variance
- Large λ: High bias, low variance (underfitting)

**RBF Regression**:
- Few basis functions: High bias (underfitting)
- Optimal number: Balanced complexity
- Many basis functions: High variance (overfitting)

**Polynomial Features**:
- Low degree: High bias (underfitting)
- Optimal degree: Best generalization
- High degree: High variance (overfitting)

### Practical Insights

1. **Regularization is Essential**: Without proper regularization or complexity control, models either overfit or underfit
2. **Validation Set Importance**: Using a separate validation set helps select optimal hyperparameters
3. **Feature Engineering Impact**: Polynomial features can significantly improve model performance when chosen appropriately
4. **Curse of Dimensionality**: Very high-degree polynomials create excessive features that may harm generalization

## Technologies Used

- **Python 3.8+**: Core programming language
- **NumPy**: Numerical computations and array operations
- **Pandas**: Data manipulation and preprocessing
- **Scikit-learn**: Machine learning models and evaluation
- **Matplotlib**: Data visualization and plotting
- **Seaborn**: Statistical data visualization

## Project Structure

```
ml-regression-models/
│
├── assignment2_encs5341.py              # Main implementation script
├── Assignment2_ENCS5341.ipynb           # Jupyter notebook version
├── assignment_2.pdf                     # Assignment specifications
│
├── data/
│   ├── customer_data.csv                # Customer churn dataset
│   └── preprocessing_output.csv         # Preprocessed data
│
├── models/
│   ├── ridge_regression.py              # Ridge regression implementation
│   ├── rbf_regression.py                # RBF regression implementation
│   └── logistic_regression_models.py    # Logistic regression models
│
├── utils/
│   └── data_preprocessor.py             # Reusable preprocessing class
│
├── visualizations/
│   ├── ridge_comparison.png             # Ridge λ comparison plot
│   ├── rbf_comparison.png               # RBF basis function comparison
│   ├── polynomial_accuracy.png          # Accuracy vs polynomial degree
│   └── roc_curve.png                    # ROC curve for best model
│
├── requirements.txt                     # Python dependencies
├── README.md                            # Project documentation
└── LICENSE                              # MIT License
```

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


## Acknowledgments

- Course: Machine Learning and Data Science (ENCS5341)
- Electrical and Computer Engineering Department
- Assignment guidance provided by course instructors
- Preprocessing pipeline adapted from Assignment 1

## Contact

For questions or feedback, please open an issue on GitHub.

---

**Note**: This is an educational project demonstrating non-linear regression techniques and logistic regression for classification. The synthetic dataset in Part 1 and customer churn dataset in Part 2 are used for learning purposes.

## References

- Scikit-learn Documentation: [Ridge Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)
- Scikit-learn Documentation: [Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- Scikit-learn Documentation: [Polynomial Features](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html)
- Understanding the Bias-Variance Trade-off
- Radial Basis Function Networks
