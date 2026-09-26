# German Credit Risk Analysis

This project performs statistical analysis and credit risk prediction on the **German Credit dataset**.

It calculates **Pearson, Spearman, and Kendall correlation matrices**, generates correlation heatmaps, checks multicollinearity using **Variance Inflation Factor (VIF)**, and trains a **Logistic Regression** model to predict bad credit risk.

## Project Structure

```text
project/
│
├── data/
│   └── german_credit.csv
│
├── src/
│   ├── main.py
│   ├── helperFunctions.py
│   └── graph.py
│
├── output/
│   ├── pearson_heatmap.png
│   ├── spearman_heatmap.png
│   └── kendall_heatmap.png
│
└── README.md
```

## Features

- Load the German Credit dataset
- Analyze target distribution
- Calculate feature means and covariance
- Calculate Pearson correlation
- Calculate Spearman rank correlation
- Calculate Kendall correlation
- Generate correlation heatmaps
- Calculate Variance Inflation Factor (VIF)
- Prepare the target variable for binary classification
- Split data into training and testing sets
- Standardize features
- Train a Logistic Regression model
- Evaluate the model using accuracy, confusion matrix, and classification report
- Generate bad credit risk probability scores

## Correlation Analysis

### Pearson Correlation

Pearson correlation measures the strength of the linear relationship between two numerical variables.

The project calculates Pearson correlation using covariance and variance.

### Spearman Correlation

Spearman correlation measures the relationship between the ranks of two variables.

The feature values are first converted into ranks, and Pearson correlation is then calculated on the ranked values.

### Kendall Correlation

Kendall correlation measures the relationship between two variables by comparing pairs of observations.

The implementation calculates:

- Concordant pairs
- Discordant pairs
- Ties in X
- Ties in Y

These values are then used to calculate the Kendall correlation coefficient.

## Heatmaps

Heatmaps are generated for all three correlation methods:

```text
output/
├── pearson_heatmap.png
├── spearman_heatmap.png
└── kendall_heatmap.png
```

Correlation values range from:

```text
-1  -> Strong negative correlation
 0  -> No correlation
+1  -> Strong positive correlation
```

## Variance Inflation Factor (VIF)

VIF is used to check for multicollinearity between input features.

The project calculates VIF values using the inverse of the Pearson correlation matrix.

In the current dataset, no features were removed based on VIF.

## Logistic Regression

The target variable is converted into a binary classification target:

```text
1 -> 0
2 -> 1
```

The data is then divided into:

```text
80% Training Data
20% Testing Data
```

A stratified split is used to maintain the target class distribution in both training and testing sets.

## Feature Scaling

The features are standardized using `StandardScaler`.

```python
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

The scaler is fitted only on the training data and then applied to the test data.

## Model Training

A Logistic Regression model is trained using the scaled training data.

```python
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)
```

## Model Evaluation

The model is evaluated using:

- Accuracy
- Confusion Matrix
- Precision
- Recall
- F1-score
- Support

The model also generates probability scores for bad credit risk:

```python
y_prob = model.predict_proba(X_test_scaled)[:, 1]
```

These probabilities represent the model's estimated probability that a sample belongs to the bad credit risk class.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Installation

Install the required Python libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Running the Project

Navigate to the `src` directory:

```bash
cd src
```

Run the program:

```bash
python main.py
```

## Output

The program prints:

- Pearson correlation matrix
- Spearman correlation matrix
- Kendall correlation matrix
- VIF values
- Logistic Regression accuracy
- Confusion matrix
- Classification report
- First 10 bad credit propensity scores

The generated heatmaps are stored inside the `output` directory.

## Workflow

```text
Load Dataset
      ↓
Analyze Target Distribution
      ↓
Calculate Pearson Correlation
      ↓
Calculate Spearman Correlation
      ↓
Calculate Kendall Correlation
      ↓
Generate Heatmaps
      ↓
Calculate VIF
      ↓
Prepare Target Variable
      ↓
Train/Test Split
      ↓
Standardize Features
      ↓
Train Logistic Regression
      ↓
Evaluate Model
      ↓
Generate Credit Risk Probabilities
```
